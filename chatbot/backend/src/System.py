import os
import re
import json
import asyncio
import signal
import inspect
import logging
from dataclasses import dataclass, field
from contextlib import suppress
from openai import OpenAI
from dotenv import load_dotenv
import Tools

load_dotenv()
logger = logging.getLogger(__name__)

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '1024'))
    LLM_TIMEOUT = float(os.getenv('LLM_TIMEOUT', '30.0'))
    MAX_CONTEXT_TURNS = int(os.getenv('MAX_CONTEXT_TURNS', '3'))
    CHROMA_TOP_K = int(os.getenv('CHROMA_TOP_K', '3'))         
    CHROMA_MIN_SCORE = float(os.getenv('CHROMA_MIN_SCORE', '0.5'))
    SHUTDOWN_TIMEOUT = float(os.getenv('SHUTDOWN_TIMEOUT', '10.0'))

config = Config()

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    timeout=config.LLM_TIMEOUT,
)

# ── Graceful shutdown tracking ──
_shutting_down = False
_inflight_tasks: set[asyncio.Task] = set()
_shutdown_event = asyncio.Event()


def _track_task(task: asyncio.Task) -> None:
    """Theo dõi task đang chạy để có thể cancel khi shutdown."""
    _inflight_tasks.add(task)
    task.add_done_callback(_inflight_tasks.discard)


def is_shutting_down() -> bool:
    return _shutting_down


async def wait_for_inflight(timeout: float = 5.0) -> int:
    """Chờ các task đang chạy hoàn tất, trả về số task còn lại nếu timeout."""
    if not _inflight_tasks:
        return 0
    remaining = list(_inflight_tasks)
    done, pending = await asyncio.wait(remaining, timeout=timeout)
    for t in pending:
        t.cancel()
        with suppress(asyncio.CancelledError):
            await t
    return len(pending)

_FALLBACK_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_qa_retriever",
            "description": "CHỈ dùng khi người dùng hỏi về ký hiệu của một từ cụ thể (ví dụ: 'ký hiệu con chó', 'từ cha trong VSL', 'thể hiện từ mẹ thế nào'). Không dùng cho câu hỏi chào hỏi, giới thiệu, hay câu thông thường.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Câu hỏi của người dùng về ký hiệu từ vựng"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vsl_restruct",
            "description": "CHỈ dùng khi người dùng yêu cầu chuyển đổi một câu tiếng Việt sang cấu trúc ngữ pháp VSL (S-P-O). Ví dụ: 'chuyển câu tôi đi ăn sang VSL', 'sắp xếp câu này theo VSL'. KHÔNG dùng cho câu hỏi thông thường, giới thiệu, hay hỏi về ký hiệu từ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Câu tiếng Việt cần chuyển đổi sang cấu trúc VSL"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_web_search",
            "description": "CHỈ dùng khi người dùng hỏi về tin tức, thông tin mới, số liệu cập nhật, sự kiện hiện tại. KHÔNG dùng cho câu hỏi về VSL hay chào hỏi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Từ khóa tìm kiếm trên internet"}
                },
                "required": ["query"]
            }
        }
    },
]

TOOL_FUNCS = Tools.TOOLS_MAPPING_TO_FUNC_ASYNC


TOOL_DISPLAY_MAP = {
    "get_qa_retriever": "📖 Tra cứu tri thức",
    "vsl_restruct": "🔄 Chuyển đổi cấu trúc",
    "get_web_search": "🌐 Tìm kiếm web",
}

def map_tools_display(tools: list[str]) -> list[str]:
    return [TOOL_DISPLAY_MAP.get(t, t) for t in tools]


@dataclass
class IntentResult:
    tools: list[str] = field(default_factory=list)
    needs_extraction: bool = False
    extracted_sentence: str | None = None
    confidence: float = 1.0  # 0.0 → 1.0


@dataclass
class ToolResult:
    content: str = ""
    links: list = field(default_factory=list)


@dataclass
class OrchestratedResult:
    fused_context: str = ""
    tools_used: list[str] = field(default_factory=list)
    all_links: list = field(default_factory=list)


class QueryAnalyzer:
    """Phân tích query bằng regex/keywords — zero LLM cost."""

    # Patterns cho từng tool
    _VSL_SIGN_WORDS = [
        "ký hiệu", "miêu tả", "biểu diễn", "cách đánh", "cách ký",
        "dấu hiệu", "ngôn ngữ ký hiệu", "vsl", "động tác", "thể hiện.*tay",
        "ký tự", "đánh vần", "làm thế nào.*ký",
    ]
    _VSL_RESTRUCT_WORDS = [
        "chuyển.*câu", "sắp xếp.*câu", "cấu trúc.*vsl", "ngữ pháp.*vsl",
        "sang.*vsl", "viết lại.*vsl", "đổi.*cấu trúc", "chuyển đổi.*vsl",
        "sap xep", "cau truc",
    ]
    _WEB_SEARCH_WORDS = [
        "tin tức", "mới nhất", "hiện nay", "năm 202[5-9]",
        "thị trường", "báo giá", "xu hướng", "google", "tìm kiếm.*web",
        "thông tin mới", "cập nhật",
    ]
    _GREETING_WORDS = [
        "xin chào", "chào", "hello", "hi", "bạn là ai",
        "giới thiệu", "có thể làm gì", "bạn tên gì", "làm quen",
    ]

    @classmethod
    def analyze(cls, query: str) -> IntentResult:
        q = query.lower().strip()
        tools = []
        needs_extraction = False

        # 1. Kiểm tra greeting nhanh — không cần tool
        if cls._is_greeting(q):
            return IntentResult(tools=[], confidence=1.0)

        # 2. Kiểm tra vsl_restruct (ưu tiên cao nhất vì cần extract)
        if cls._match_any(q, cls._VSL_RESTRUCT_WORDS):
            tools.append("vsl_restruct")
            needs_extraction = True

        # 3. Kiểm tra tra cứu ký hiệu
        if cls._match_any(q, cls._VSL_SIGN_WORDS) or cls._likely_sign_query(q):
            tools.append("get_qa_retriever")

        # 4. Kiểm tra web search
        if cls._match_any(q, cls._WEB_SEARCH_WORDS):
            tools.append("get_web_search")

        # 5. Default: nếu không match tool nào và không phải greeting
        if not tools and len(q) > 2:
            # Câu hỏi chung về VSL → tra từ điển
            if cls._likely_vsl_related(q):
                tools.append("get_qa_retriever")
            else:
                # Query dài, không rõ intent → fallback về LLM routing
                return IntentResult(tools=[], confidence=0.3, needs_extraction=False)

        # Dedup + giữ thứ tự ưu tiên
        seen = set()
        unique_tools = []
        for t in tools:
            if t not in seen:
                seen.add(t)
                unique_tools.append(t)

        return IntentResult(
            tools=unique_tools,
            needs_extraction=needs_extraction,
            confidence=0.9 if unique_tools else 0.3,
        )

    @classmethod
    def _is_greeting(cls, q: str) -> bool:
        return cls._match_any(q, cls._GREETING_WORDS)

    @classmethod
    def _match_any(cls, q: str, patterns: list[str]) -> bool:
        for p in patterns:
            if re.search(p, q):
                return True
        return False

    @classmethod
    def _likely_sign_query(cls, q: str) -> bool:
        short_words = [w for w in q.split() if len(w) <= 5]
        return len(short_words) >= 1 and any(
            w in q for w in ["là gì", "trong vsl", "từ", "chữ"]
        )

    @classmethod
    def _likely_vsl_related(cls, q: str) -> bool:
        vsl_hints = ["vsl", "ký hiệu", "ngôn ngữ", "người câm", "điếc",
                     "thủ ngữ", "câm", "ra dấu", "tay"]
        for hint in vsl_hints:
            if hint in q:
                return True
        # Nếu query có dạng câu hỏi về 1 từ đơn
        if len(q.split()) <= 5 and ("?" in q or q.endswith("?")):
            return True
        return False

    @staticmethod
    def extract_sentence(query: str) -> str | None:
        patterns = [
            # "chuyển câu 'tôi đi ăn' sang VSL"
            r"['\"](.+?)['\"]",
            # "chuyển câu tôi đi ăn sang VSL"
            r"(?:chuyển|đổi)\s+(?:câu\s+)?(.+?)(?:\s+sang|\s+thành|\s+theo|\s*$)",
            # "câu: tôi đi ăn"
            r"(?:câu|câu sau|đoạn sau)[:\s]+(.+?)(?:\s*$|\s*sang)",
        ]
        for p in patterns:
            m = re.search(p, query, re.IGNORECASE)
            if m:
                extracted = m.group(1).strip().rstrip(".,;")
                if len(extracted) >= 3:  # câu phải có ít nhất 3 ký tự
                    return extracted
        return None



class MiniExtractor:
    _PROMPT = (
        "Trích xuất CHÍNH XÁC câu tiếng Việt cần chuyển đổi cấu trúc VSL "
        "(sắp xếp lại thứ tự từ theo ngữ pháp VSL) từ câu hỏi dưới đây.\n"
        "CHỈ trả về câu đó, không thêm bất kỳ ký tự hay giải thích nào.\n\n"
        "Câu hỏi: {query}\n\n"
        "Câu cần chuyển đổi:"
    )

    @classmethod
    async def extract(cls, query: str) -> str | None:
        try:
            resp = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": cls._PROMPT.format(query=query)}],
                temperature=0,
                max_tokens=50,
            )
            extracted = (resp.choices[0].message.content or "").strip().strip("\"'")
            if extracted and len(extracted) >= 2:
                logger.info(f"MiniExtractor: '{query[:50]}...' → '{extracted[:50]}...'")
                return extracted
            return None
        except Exception as e:
            logger.warning(f"MiniExtractor failed: {e}")
            return None



class ToolOrchestrator:
    @staticmethod
    async def execute(
        tools: list[str],
        query: str,
        extracted_sentence: str | None = None,
    ) -> OrchestratedResult:
        if not tools:
            return OrchestratedResult()

        # Map tool name → coroutine
        coros: dict = {}
        for tool in tools:
            if tool == "get_qa_retriever":
                coros[tool] = _execute_tool_rich(tool, {"query": query})
            elif tool == "vsl_restruct":
                text = extracted_sentence or query
                # Nếu câu extract quá ngắn (< 3 từ) hoặc giống hệt query → dùng query
                if text and len(text.split()) < 3 and text != query:
                    text = query
                coros[tool] = _execute_tool_rich(tool, {"text": text})
            elif tool == "get_web_search":
                coros[tool] = _execute_tool_rich(tool, {"query": query})

        if not coros:
            return OrchestratedResult()

        # Song song: tất cả tool chạy đồng thời
        results = await asyncio.gather(*coros.values(), return_exceptions=True)

        context_parts = []
        tools_used = []
        all_links = []

        for (tool_name, _coro), result in zip(coros.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_name} execution error: {result}")
                context_parts.append(f"[{tool_name}]: Lỗi: {result}")
                continue

            tools_used.append(tool_name)
            content = (result.get("content") or "").strip()
            if content:
                # Prefix rõ nguồn để LLM biết thông tin từ tool nào
                label = {
                    "get_qa_retriever": "TRA CỨU TỪ ĐIỂN VSL",
                    "vsl_restruct": "CHUYỂN ĐỔI CẤU TRÚC VSL",
                    "get_web_search": "TÌM KIẾM WEB",
                }.get(tool_name, tool_name.upper())
                context_parts.append(f"=== {label} ===\n{content}")

            links = result.get("links") or []
            all_links.extend(links)

        return OrchestratedResult(
            fused_context="\n\n".join(context_parts),
            tools_used=tools_used,
            all_links=all_links,
        )


async def _execute_tool_rich(name: str, args: dict) -> dict:
    """Execute tool với logging + error handling. Trả về dict {content, links}."""
    func = TOOL_FUNCS.get(name)
    if not func:
        logger.warning(f"Tool not found: {name}")
        return {"content": "", "links": []}

    logger.info(f"Executing tool: {name}({json.dumps(args, ensure_ascii=False)})")
    try:
        if inspect.iscoroutinefunction(func):
            result = await func(**args)
        else:
            result = func(**args)

        content = result.get("context", "") if isinstance(result, dict) else str(result)
        links = result.get("links", []) if isinstance(result, dict) else []

        # Lọc kết quả ChromaDB theo score threshold
        if name == "get_qa_retriever" and isinstance(result, dict):
            results_list = result.get("results", [])
            high_quality = [r for r in results_list if r.get("score", 0) >= config.CHROMA_MIN_SCORE]
            if high_quality and len(high_quality) < len(results_list):
                # Rebuild context chỉ với kết quả đạt threshold
                context_parts = [
                    f"[{i+1}] Score: {r['score']:.3f} | Source: {r['source']}\n{r['text']}"
                    for i, r in enumerate(high_quality)
                ]
                content = "\n\n".join(context_parts)
                logger.info(f"Filtered {len(results_list)} → {len(high_quality)} results (threshold={config.CHROMA_MIN_SCORE})")

        return {"content": content, "links": links}
    except Exception as e:
        logger.error(f"Tool {name} execution error: {e}")
        return {"content": "", "links": []}



SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên về Ngôn ngữ Ký hiệu Việt Nam (VSL). "
    "Luôn trả lời bằng tiếng Việt, tự nhiên, không dùng ký tự đặc biệt, "
    "không tiêu đề, không markdown, không trích link hay ghi chú nguồn.\n\n"

    "QUY TẮC CHỌN NHÁNH TRẢ LỜI (chỉ dùng nhánh tương ứng với dữ liệu được cung cấp):\n\n"

    "1) Nếu có === TRA CỨU TỪ ĐIỂN VSL ===:\n"
    "   - Dùng nguyên văn mô tả ký hiệu tìm được, kèm Loại và Khu vực nếu có.\n"
    "   - Nếu không có kết quả phù hợp, trả lời: 'Chưa có dữ liệu về ký hiệu này'.\n"
    "   - Giải thích thêm cho người học: paraphrase lại mô tả bằng lời tự nhiên, nói rõ cách thực hiện ký hiệu.\n\n"

    "2) Nếu có === CHUYỂN ĐỔI CẤU TRÚC VSL ===:\n"
    "   - Đây là nhánh CHỈ dùng khi người dùng yêu cầu chuyển một câu tiếng Việt "
    "sang cấu trúc ngữ pháp VSL (ví dụ: 'câu này viết theo VSL thế nào', "
    "'chuyển sang ngữ pháp ký hiệu', 'làm thế nào để nói ...').\n"
    "   - KHÔNG dùng nhánh này cho câu giới thiệu bản thân đơn thuần như 'tôi tên là X', "
    "'tôi là Y', vì đó là giao tiếp thông thường, không phải yêu cầu chuyển đổi cấu trúc.\n"
    "   - Dùng kết quả được cung cấp, đối chiếu với ngữ pháp VSL dưới đây, "
    "rồi giải thích NGẮN GỌN nhưng ĐẦY ĐỦ: đưa ra câu VSL, nói rõ vì sao sắp xếp như vậy "
    "(chủ ngữ - tân ngữ - động từ), từ nào bị lược bỏ và tại sao.\n"
    "   - Ví dụ: 'Câu \"tôi đi ăn\" trong VSL được sắp xếp là: tôi / ăn / đi. "
    "Trong VSL, thứ tự là Chủ ngữ - Tân ngữ - Động từ, nên \"tôi\" đứng trước, "
    "\"ăn\" ở giữa, \"đi\" ở cuối. Động từ \"đi\" được chuyển xuống cuối câu.'\n\n"

    "3) Nếu có === TÌM KIẾM WEB ===:\n"
    "   - Tổng hợp lại bằng lời văn của bạn, không kèm link hay nói 'theo nguồn...'.\n\n"

    "4) Nếu KHÔNG có bất kỳ nhãn === ... === nào ở trên (ví dụ: chào hỏi, "
    "hỏi bạn là ai, giới thiệu bản thân, hỏi kiến thức chung về VSL, "
    "trò chuyện thông thường):\n"
    "   - Trả lời trực tiếp bằng tiếng Việt tự nhiên như một trợ lý bình thường.\n"
    "   - TUYỆT ĐỐI không áp cấu trúc ngữ pháp VSL (S-O-P, bỏ từ nối...) vào câu trả lời "
    "trong trường hợp này. Cấu trúc VSL chỉ áp dụng cho nhánh (2).\n\n"

    "NGỮ PHÁP VSL (chỉ tham chiếu khi xử lý nhánh 2):\n"
    "- Cấu trúc câu: Chủ ngữ → Tân ngữ → Động từ (S → O → P).\n"
    "- Phủ định: thêm 'không' ở cuối câu.\n"
    "- Câu hỏi: từ hỏi đặt ở cuối câu.\n"
    "- Số đứng sau danh từ. Thời gian đứng ở đầu câu.\n"
    "- Lược bỏ các hư từ: là, của, ở, những, các, đã, sẽ, rất, đang.\n"
)


def _build_messages(query: str, conversation_history: list | None = None, tool_context: str | None = None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject context từ tools (nếu có) — ngay sau system prompt
    if tool_context:
        messages.append({
            "role": "system",
            "content": (
                "Kết quả tra cứu từ hệ thống:\n"
                f"{tool_context}\n\n"
                "Hãy dùng thông tin trên để trả lời. "
                "Nếu kết quả trống hoặc lỗi, hãy trả lời dựa trên kiến thức của bạn."
            ),
        })

    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})
    return messages



def _safe_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."
    return text



async def run_query(query: str, conversation_history: list = None) -> dict:
    """Single-pass: heuristic → parallel tools → 1 LLM call. Không tool schemas."""
    if is_shutting_down():
        return {
            "answer": "Hệ thống đang tắt, vui lòng thử lại sau.",
            "tools_used": [],
            "links": [],
        }
    try:
        # ── Phase 1: Analyze intent ──
        intent = QueryAnalyzer.analyze(query)

        # ── Phase 2: Extract sentence cho vsl_restruct (nếu cần) ──
        extracted = None
        if "vsl_restruct" in intent.tools:
            extracted = QueryAnalyzer.extract_sentence(query)
            if not extracted:
                extracted = await MiniExtractor.extract(query)

        # ── Phase 3: Execute tools song song ──
        tool_results = await ToolOrchestrator.execute(
            tools=intent.tools,
            query=query,
            extracted_sentence=extracted,
        )

        # ── Phase 4: Build messages + single LLM call ──
        messages = _build_messages(
            query=query,
            conversation_history=conversation_history,
            tool_context=tool_results.fused_context or None,
        )

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

        final_answer = response.choices[0].message.content or ""

    except Exception as e:
        logger.error(f"run_query error: {e}")
        final_answer = "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."
        tool_results = OrchestratedResult()

    return {
        "answer": _safe_answer(final_answer),
        "tools_used": map_tools_display(tool_results.tools_used),
        "links": tool_results.all_links,
    }



async def run_query_streaming(query: str, conversation_history: list = None):
    """
    Hybrid routing: heuristic pre-filter → LLM quyết định tool → stream.
    2-pass: (1) non-streaming xác định tool → (2) streaming câu trả lời.
    """
    if is_shutting_down():
        yield {"type": "token", "content": "Hệ thống đang tắt, vui lòng thử lại sau."}
        yield {"type": "_done", "tools_used": [], "links": []}
        return

    tool_results = OrchestratedResult()
    try:
        yield {"type": "info", "content": "Đang phân tích câu hỏi..."}

        # ── Bước 0: Heuristic pre-filter ──
        # Nếu heuristic chắc chắn là greeting (confidence=1.0) → skip tool, stream luôn
        # Còn lại → để LLM quyết định tool
        heuristic_intent = QueryAnalyzer.analyze(query)
        needs_tool_call = not (heuristic_intent.confidence == 1.0 and not heuristic_intent.tools)

        messages = _build_messages(query, conversation_history)

        if needs_tool_call:
            # ── Bước 1: LLM quyết định tool (chỉ khi heuristic nghi ngờ cần tool) ──
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                tools=_FALLBACK_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=config.MAX_OUTPUT_TOKENS,
            )

            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                yield {"type": "info", "content": "Đang tra cứu thông tin..."}

                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        func_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    tool_result = await _execute_tool_rich(func_name, func_args)
                    tool_results.tools_used.append(func_name)
                    tool_results.all_links.extend(tool_result.get("links", []))

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result["content"],
                    })

                yield {"type": "info", "content": "Đang tổng hợp câu trả lời..."}
        else:
            # Không cần tool → stream trực tiếp, không có initial content
            initial_content = None

        # ── Bước 2: Stream câu trả lời ──
        stream = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        has_content = False
        if needs_tool_call and msg.content:
            has_content = True
            yield {"type": "token", "content": msg.content}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                has_content = True
                yield {"type": "token", "content": delta.content}

        if not has_content:
            yield {"type": "token", "content": "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."}

    except Exception as e:
        logger.error(f"run_query_streaming error: {e}")
        yield {"type": "token", "content": "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."}

    finally:
        yield {
            "type": "_done",
            "tools_used": map_tools_display(tool_results.tools_used),
            "links": tool_results.all_links,
        }



def preload_embedding_model():
    Tools.preload()


async def cleanup():
    """Giải phóng toàn bộ tài nguyên hệ thống."""
    global _shutting_down
    _shutting_down = True
    logger.info("=== System cleanup started ===")

    # 1. Chờ in-flight tasks hoàn tất (có timeout)
    remaining = await wait_for_inflight(config.SHUTDOWN_TIMEOUT / 2)
    if remaining > 0:
        logger.warning(f"{remaining} in-flight tasks cancelled on shutdown.")

    # 2. Đóng OpenAI HTTP client
    try:
        client.close()
        logger.info("OpenAI client closed.")
    except Exception as e:
        logger.warning(f"OpenAI client close error: {e}")

    # 3. Cleanup Tools (HTTP clients, ChromaDB, caches)
    await Tools.cleanup()

    # 4. Đánh dấu shutdown hoàn tất
    _shutdown_event.set()
    _inflight_tasks.clear()
    logger.info("=== System cleanup done ===")



