import os
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
    MAX_CONTEXT_TURNS = int(os.getenv('MAX_CONTEXT_TURNS', '5'))
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

VSL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_qa_retriever",
            "description": "Tra cứu thông tin về ngôn ngữ ký hiệu Việt Nam (VSL) từ cơ sở tri thức. Dùng cho: (1) hỏi về ký hiệu / cách đánh / cách mô tả một từ cụ thể (vd: 'ký hiệu con chó', 'từ mẹ trong VSL'), (2) hỏi kiến thức chung về VSL, người khiếm thính, văn hóa điếc (vd: 'ngôn ngữ ký hiệu là gì', 'cộng đồng khiếm thính Việt Nam'), (3) bất kỳ câu hỏi nào liên quan đến VSL và người khiếm thính.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Toàn bộ câu hỏi của người dùng (dùng để vector search)"
                    },
                    "search_word": {
                        "type": "string",
                        "description": "[TÙY CHỌN] TỪ/CỤM TỪ CHÍNH XÁC cần tra ký hiệu, ví dụ: 'mẹ', 'con chó', 'anh trai'. Chỉ cung cấp nếu người dùng hỏi về ký hiệu của một từ cụ thể. Nếu là câu hỏi chung, không cần tham số này."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vsl_restruct",
            "description": "Dùng khi người dùng yêu cầu chuyển đổi một câu tiếng Việt sang cấu trúc ngữ pháp VSL (S-O-P). Ví dụ: 'chuyển câu tôi đi ăn sang VSL', 'sắp xếp câu này theo VSL'.",
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
            "description": "Tìm kiếm thông tin trên internet. Dùng khi người dùng hỏi về: (1) tin tức, sự kiện hiện tại, số liệu cập nhật, (2) thông tin thực tế, số liệu thống kê, dữ liệu mới nhất (vd: 'quy mô cộng đồng khiếm thính', 'bao nhiêu người điếc ở Việt Nam'), (3) bất kỳ câu hỏi nào cần tra cứu thông tin ngoài kiến thức của bạn.",
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
class ToolResult:
    content: str = ""
    links: list = field(default_factory=list)


@dataclass
class OrchestratedResult:
    fused_context: str = ""
    tools_used: list[str] = field(default_factory=list)
    all_links: list = field(default_factory=list)


def _wrap_tool_result(tool_name: str, content: str) -> str:
    """Wrap content với label để LLM nhận biết nhánh trả lời."""
    label = {
        "get_qa_retriever": "TRA CỨU TỪ ĐIỂN VSL",
        "vsl_restruct": "CHUYỂN ĐỔI CẤU TRÚC VSL",
        "get_web_search": "TÌM KIẾM WEB",
    }.get(tool_name, tool_name.upper())
    return f"=== {label} ===\n{content}"


class ToolOrchestrator:
    @staticmethod
    async def execute(
        tools_args: dict[str, dict],
    ) -> OrchestratedResult:
        """Execute tools với args được LLM cung cấp.
        
        Args:
            tools_args: Dict {tool_name: args_dict} do LLM quyết định
        """
        if not tools_args:
            return OrchestratedResult()

        # Map tool name → coroutine
        coros: dict = {}
        for tool_name, args in tools_args.items():
            func = TOOL_FUNCS.get(tool_name)
            if func:
                coros[tool_name] = _execute_tool_rich(tool_name, args)

        if not coros:
            return OrchestratedResult()

        results = await asyncio.gather(*coros.values(), return_exceptions=True)

        context_parts = []
        tools_used = []
        all_links = []

        for (tool_name, _coro), result in zip(coros.items(), results):
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_name} execution error: {result}")
                context_parts.append(f"=== LỖI ===\n[{tool_name}]: {result}")
                continue

            tools_used.append(tool_name)
            content = (result.get("content") or "").strip()
            if content:
                context_parts.append(_wrap_tool_result(tool_name, content))

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
    "   - Nếu tìm thấy kết quả phù hợp:\n"
    "       + Dùng nguyên văn mô tả ký hiệu tìm được, kèm Loại và Khu vực nếu có.\n"
    "       + Giải thích thêm cho người học: paraphrase lại mô tả bằng lời tự nhiên, nói rõ cách thực hiện ký hiệu.\n"
    "   - Nếu KHÔNG có kết quả phù hợp: CHỈ trả lời 'Chưa có dữ liệu về ký hiệu này.' "
    "KHÔNG tự ý mô tả ký hiệu dựa trên kiến thức riêng của bạn.\n\n"

    "2) Nếu có === CHUYỂN ĐỔI CẤU TRÚC VSL ===:\n"
    "   - Đây là nhánh CHỈ dùng khi người dùng yêu cầu chuyển một câu tiếng Việt "
    "sang cấu trúc ngữ pháp VSL (ví dụ: 'câu này viết theo VSL thế nào', "
    "'chuyển sang ngữ pháp ký hiệu', 'làm thế nào để nói ...').\n"
    "   - KHÔNG dùng nhánh này cho câu giới thiệu bản thân đơn thuần như 'tôi tên là X', "
    "'tôi là Y', vì đó là giao tiếp thông thường, không phải yêu cầu chuyển đổi cấu trúc.\n"
    "   - Bước 1 — TRÍCH XUẤT: Lấy nguyên văn câu VSL từ kết quả tool, "
    "hiển thị ở đầu câu trả lời dạng: 'Câu VSL: tôi / ăn / cơm'.\n"
    "   - Bước 2 — KIỂM TRA: Đối chiếu với ngữ pháp VSL bên dưới. "
    "Xác minh cấu trúc S-O-P, hư từ đã lược bỏ chưa, phủ định/câu hỏi xử lý đúng chưa. "
    "Nếu tool bị lỗi → thông báo dịch vụ tạm không khả dụng và tự phân tích.\n"
    "   - Bước 3 — GIẢI THÍCH: Giải thích ngắn gọn, tự nhiên: vì sao sắp xếp như vậy, "
    "từ nào bị lược bỏ và tại sao.\n"
    "   - Định dạng đầu ra: Xuống dòng giữa các bước, KHÔNG dùng dấu [] hay ký tự đặc biệt. "
    "Viết liền mạch, tự nhiên như đang giảng giải cho người học.\n"
    "   - Ví dụ:\n"
    "Câu VSL: tôi / ăn / cơm\n\n"
    "→ Câu này tuân theo cấu trúc Chủ ngữ - Tân ngữ - Động từ (S-O-P) của VSL.\n\n"
    "→ Hư từ 'sẽ' và 'đang' đã được lược bỏ vì VSL không dùng từ nối, thời gian thể hiện qua ngữ cảnh.\n\n"

    "3) Nếu có === TÌM KIẾM WEB ===:\n"
    "   - Tổng hợp lại bằng lời văn của bạn, không kèm link hay nói 'theo nguồn...'.\n\n"

    "4) Nếu KHÔNG có bất kỳ nhãn === ... === nào ở trên (ví dụ: chào hỏi, "
    "hỏi bạn là ai, giới thiệu bản thân, trò chuyện thông thường):\n"
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
    """LLM quyết định tool → execute → trả lời (dùng chuẩn OpenAI tool messages)."""
    if is_shutting_down():
        return {
            "answer": "Hệ thống đang tắt, vui lòng thử lại sau.",
            "tools_used": [],
            "links": [],
        }

    tool_results = OrchestratedResult()
    try:
        # ── Phase 1: LLM quyết định tool + tham số ──
        messages = _build_messages(query, conversation_history)

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=VSL_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

        choice = response.choices[0]
        msg = choice.message

        # ── Phase 2: Execute tool nếu LLM yêu cầu ──
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                # Execute tool
                tool_result = await _execute_tool_rich(tc.function.name, func_args)
                tool_results.tools_used.append(tc.function.name)
                tool_results.all_links.extend(tool_result.get("links", []))

                # Wrap content với label để LLM nhận biết nhánh
                labeled_content = tool_result["content"]
                if labeled_content.strip():
                    labeled_content = _wrap_tool_result(tc.function.name, labeled_content)

                # Gắn assistant message + tool message theo đúng chuẩn OpenAI
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": labeled_content,
                })

        # ── Phase 3: Generate final answer ──
        if msg.tool_calls:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=config.MAX_OUTPUT_TOKENS,
            )
            final_answer = response.choices[0].message.content or ""
        else:
            # LLM trả lời trực tiếp (greeting, chat thông thường)
            final_answer = msg.content or ""

    except Exception as e:
        logger.error(f"run_query error: {e}")
        final_answer = "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."

    return {
        "answer": _safe_answer(final_answer),
        "tools_used": map_tools_display(tool_results.tools_used),
        "links": tool_results.all_links,
    }



async def run_query_streaming(query: str, conversation_history: list = None):
    """
    LLM quyết định tool (với search_word) → execute → stream câu trả lời.
    2-pass: (1) non-streaming xác định tool → (2) streaming câu trả lời.
    """
    if is_shutting_down():
        yield {"type": "token", "content": "Hệ thống đang tắt, vui lòng thử lại sau."}
        yield {"type": "_done", "tools_used": [], "links": []}
        return

    tool_results = OrchestratedResult()
    has_content = False
    try:
        yield {"type": "info", "content": "Đang phân tích câu hỏi..."}

        messages = _build_messages(query, conversation_history)

        # ── Bước 1: LLM quyết định tool + tham số ──
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=VSL_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            yield {"type": "info", "content": "Đang tra cứu thông tin..."}

            # Execute tools với args từ LLM (search_word được LLM cung cấp trực tiếp)
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                tool_result = await _execute_tool_rich(func_name, func_args)
                tool_results.tools_used.append(func_name)
                tool_results.all_links.extend(tool_result.get("links", []))

                # Wrap với label để khớp SYSTEM_PROMPT
                labeled_content = tool_result["content"]
                if labeled_content.strip():
                    labeled_content = _wrap_tool_result(func_name, labeled_content)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": labeled_content,
                })

            yield {"type": "info", "content": "Đang tổng hợp câu trả lời..."}

        # ── Bước 2: Stream câu trả lời ──
        stream = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=config.MAX_OUTPUT_TOKENS,
            stream=True,
        )

        # KHÔNG yield msg.content khi có tool_calls!
        # Nếu LLM vừa gọi tool vừa trả lời (hiếm), đó là phản hồi thiếu context tool.
        # Chỉ dùng nếu KHÔNG có tool (trả lời trực tiếp).
        if not msg.tool_calls and msg.content:
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



