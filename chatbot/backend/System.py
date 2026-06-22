import os
import json
import inspect
import logging
from openai import OpenAI
from dotenv import load_dotenv
import Tools

load_dotenv()
logger = logging.getLogger(__name__)


class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '512'))
    LLM_TIMEOUT = float(os.getenv('LLM_TIMEOUT', '30.0'))
    MAX_CONTEXT_TURNS = int(os.getenv('MAX_CONTEXT_TURNS', '3'))


config = Config()

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL,
    timeout=config.LLM_TIMEOUT,
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_qa_retriever",
            "description": (
                "Tìm kiếm và trích xuất thông tin liên quan từ cơ sở tri thức PDF nội bộ (ChromaDB local). "
                "Sử dụng khi câu hỏi liên quan đến VSL, ngôn ngữ ký hiệu, lịch sử, định nghĩa, thuật ngữ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu hỏi hoặc từ khóa cần tìm kiếm trong cơ sở tri thức"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_web_search",
            "description": (
                "Tìm kiếm thông tin trên internet. "
                "Sử dụng khi câu hỏi cần thông tin mới, cập nhật, tin tức, thị trường, số liệu thực tế."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Từ khóa tìm kiếm trên web"
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
            "description": "Tái cấu trúc câu tiếng Việt theo ngữ pháp VSL (Ngôn ngữ ký hiệu Việt Nam)",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Văn bản tiếng Việt cần chuyển đổi sang cấu trúc VSL"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên về Ngôn ngữ Ký hiệu Việt Nam (VSL). Trả lời bằng tiếng Việt.\n\n"
    "QUY TẮC:\n"
    "- Hỏi về VSL, ngôn ngữ ký hiệu → dùng get_qa_retriever\n"
    "- Hỏi về thị trường, tin tức, số liệu mới → dùng get_web_search\n"
    "- Cần chuyển đổi cấu trúc VSL → dùng vsl_restruct\n"
    "- Câu chào hỏi đơn giản → trả lời trực tiếp, không cần dùng công cụ\n"
    "- KHÔNG tự ý trả lời từ kiến thức riêng nếu có công cụ phù hợp\n"
    "- Nếu không biết, hãy dùng công cụ thay vì trả lời bừa.\n\n"
    "ĐỊNH DẠNG TRẢ LỜI:\n"
    "- Trả lời chi tiết, đầy đủ thông tin, tự nhiên như đang trò chuyện.\n"
    "- KHÔNG dùng ký tự đặc biệt như **, ###, --- trong câu trả lời.\n"
    "- Nếu liệt kê, dùng dấu gạch đầu dòng đơn giản (-) hoặc số (1. 2. 3.).\n"
    "- KHÔNG thêm tiêu đề, KHÔNG thêm ghi chú kiểu (lưu ý, chú thích).\n"
    "- Trả lời như đang chat tự nhiên, không format cầu kỳ, nhưng phải đầy đủ ý và có chiều sâu."
)

TOOL_FUNCS = Tools.TOOLS_MAPPING_TO_FUNC_ASYNC


async def _execute_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCS.get(name)
    if not func:
        logger.warning(f"Tool not found: {name}")
        return f"Lỗi: không tìm thấy công cụ {name}"

    try:
        logger.info(f"Executing tool: {name}({json.dumps(args, ensure_ascii=False)})")
        if inspect.iscoroutinefunction(func):
            result = await func(**args)
        else:
            result = func(**args)
        return result.get("context", str(result))
    except Exception as e:
        logger.error(f"Tool {name} error: {e}")
        return f"Lỗi thực thi công cụ {name}: {str(e)}"


def _build_messages(query: str, conversation_history: list = None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
    messages = _build_messages(query, conversation_history)
    tools_used = []

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=config.MAX_OUTPUT_TOKENS,
        )

        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}
                    logger.warning(f"Failed to parse args for {func_name}: {tc.function.arguments}")

                tool_result = await _execute_tool(func_name, func_args)
                tools_used.append(func_name)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=config.MAX_OUTPUT_TOKENS,
            )
            final_answer = response.choices[0].message.content or ""
        else:
            final_answer = msg.content or ""

    except Exception as e:
        logger.error(f"run_query error: {e}")
        final_answer = "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."

    return {
        "answer": _safe_answer(final_answer),
        "tools_used": tools_used,
    }


async def run_query_streaming(query: str, conversation_history: list = None):
    messages = _build_messages(query, conversation_history)
    tools_used = []

    try:
        yield {"type": "info", "content": "Đang phân tích câu hỏi..."}

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
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

                tool_result = await _execute_tool(func_name, func_args)
                tools_used.append(func_name)

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })

            yield {"type": "info", "content": "Đang tổng hợp câu trả lời..."}

            stream = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=config.MAX_OUTPUT_TOKENS,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield {"type": "token", "content": delta.content}
        else:
            final_answer = msg.content or ""
            if final_answer:
                yield {"type": "token", "content": final_answer}
            else:
                yield {"type": "token", "content": _safe_answer(final_answer)}

    except Exception as e:
        logger.error(f"run_query_streaming error: {e}")
        yield {"type": "token", "content": "Xin lỗi, hệ thống AI tạm thời không khả dụng. Vui lòng thử lại sau."}

    finally:
        yield {"type": "_done", "tools_used": tools_used}


def preload_embedding_model():
    Tools.preload()


async def cleanup():
    logger.info("Cleaning up resources...")
    await Tools.cleanup()

if __name__ == '__main__':
    import asyncio
    print("Testing run_query...")
    result = asyncio.run(run_query("Ngôn ngữ ký hiệu là gì?"))
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {result['tools_used']}")
