import os
import torch
from openai import OpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
import json
import Tools
import ChatHistory

import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.vilao.ai/v1')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'ts/gpt-5.4-mini')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', '1024'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f'Loading embedding model "{Config.EMBEDDING_MODEL_NAME}" on {Config.DEVICE} ...')
        _embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME, device=Config.DEVICE)
        logger.info('Embedding model loaded successfully.')
    return _embedding_model


def preload_embedding_model():
    get_embedding_model()
    logger.info('Embedding model is warm and ready.')


class OpenAIClient:
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            api_key=self.config.OPENAI_API_KEY,
            base_url=self.config.OPENAI_BASE_URL,
        )
        
    def invoke(self, prompt: str, temperature: float = 0) -> str:
        max_retries = self.config.MAX_RETRIES
        last_error = None
        try:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=self.config.MAX_OUTPUT_TOKENS,
                    )
                    return (response.choices[0].message.content or "").strip()
                except Exception as e:
                    last_error = e
                    logger.error(f'OpenAI ERROR (attempt {attempt}/{max_retries}): {str(e)}')
            return ""
        except Exception as e:
            logger.error(f'OpenAI ERROR: {str(e)}')
            return ""
    
    
def build_tools_list() -> str:
    tools = Tools.AGENT_TOOLS_LIST.get('TOOLS', [])
    tool_line = ['Available tools:']
    for i, tool in enumerate(tools, 1):
        tool_line.append(
            f"[{i}]: {tool['name']}\n"
            f"    Description: {tool['description']}\n"
            f"    Arguments: {tool['args']}"
        )
    return '\n'.join(tool_line)

AGENT_INSTRUCTION = """
Vai trò: Trợ lý AI cung cấp thông tin về ngôn ngữ ký hiệu Việt Nam (VSL) và trả lời bằng tiếng Việt

QUY TẮC QUAN TRỌNG:
1. LUÔN kiểm tra PAST TOOL OBSERVATIONS trước
2. Nếu thông tin trong observations đã đủ → TRẢ LỜI NGAY
3. Chỉ gọi công cụ khi THỰC SỰ thiếu thông tin
4. KHÔNG gọi lại cùng một công cụ với truy vấn tương tự

QUY TRÌNH:
Bước 1: Đọc kỹ PAST TOOL OBSERVATIONS
Bước 2: Phân loại câu hỏi:
   - Nếu liên quan VSL (ngôn ngữ ký hiệu), tài liệu VSL, hoặc cần tái cấu trúc theo VSL →
     ưu tiên get_qa_retriever; nếu cần tái cấu trúc cụ thể thì dùng vsl_restruct
   - Các chủ đề còn lại (không liên quan VSL) → dùng get_web_search
   - Nếu đã đủ thông tin → Viết READY

ĐỊNH DẠNG ĐẦU RA (BẮT BUỘC):

Nếu đã đủ thông tin:
THOUGHT: Giải thích ngắn gọn vì sao có thể trả lời dựa trên dữ liệu đã có
READY: <tóm tắt ngắn các dữ liệu/kết quả sẽ dùng để trả lời>

Nếu cần sử dụng công cụ:
THOUGHT: Giải thích thông tin nào đang thiếu và công cụ nào sẽ dùng
ACTION: get_qa_retriever | get_web_search | vsl_restruct
ARGUMENTS: <JSON hợp lệ>

ARGUMENTS THEO TỪNG CÔNG CỤ:
- get_qa_retriever  → {"query": "câu truy vấn bằng tiếng Việt"}
- get_web_search    → {"query": "câu truy vấn bằng tiếng Việt"}
- vsl_restruct      → {"text": "câu tiếng Việt cần tái cấu trúc theo VSL"}

QUY TẮC BẮT BUỘC:
- ARGUMENTS phải là JSON hợp lệ, dùng dấu nháy kép
- Nếu PAST TOOL OBSERVATIONS đã có thông tin liên quan → PHẢI DÙNG READY, KHÔNG gọi tool nữa
- Mỗi công cụ chỉ được gọi TỐI ĐA 1 LẦN cho mỗi câu hỏi
- Sau khi nhận đầu ra từ công cụ, kiểm tra tính logic và xác thực của đầu ra đó.
"""

ANSWER_INSTRUCTION = """
Bạn là trợ lý AI thân thiện, chuyên về Ngôn ngữ Ký hiệu Việt Nam (VSL).
Dựa trên câu hỏi của người dùng và dữ liệu đã thu thập, hãy viết câu trả lời cuối cùng.

Yêu cầu về phong cách:
- Tự nhiên, thân thiện như đang trò chuyện trực tiếp
- Không cứng nhắc, không liệt kê máy móc
- Có thể dùng ngôn ngữ gần gũi, ví dụ minh họa nếu phù hợp
- Trả lời bằng tiếng Việt
- KHÔNG đề cập đến "tool", "công cụ", "observations", hay bất kỳ chi tiết kỹ thuật nội bộ nào
"""


class AgentState(TypedDict):
    query: str
    last_agent_response: str
    tool_observations: list
    num_steps: int
    final_answer: str
    conversation_history: list  # List of {"role": "user"|"assistant", "content": str}


config = Config()
openai_model = OpenAIClient(config)

Tools.TOOLS_MAPPING_TO_FUNC["get_qa_retriever"] = (
    lambda query, top_k=3: Tools.get_qa_retriever(query, top_k, llm_client=openai_model)
)
    
def _format_history(history: list, max_turns: int = 10) -> str:
    """Format conversation history thành chuỗi để đưa vào prompt."""
    if not history:
        return "Chưa có lịch sử hội thoại."
    # Giữ tối đa max_turns lượt gần nhất để tránh prompt quá dài
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý"
        lines.append(f"{role_label}: {msg['content']}")
    return '\n'.join(lines)


def call_agent(state: AgentState) -> AgentState:
    observations = '\n\n'.join(state.get('tool_observations', []))
    if not observations:
        observations = 'None yet - first turn'
    
    tools_list = build_tools_list()
    
    tool_calls_count = {}
    for obs in state.get('tool_observations', []):
        if 'TOOL:' in obs:
            tool_name = obs.split('TOOL:')[1].split('\n')[0].strip()
            tool_calls_count[tool_name] = tool_calls_count.get(tool_name, 0) + 1
    
    tools_used = ', '.join([f"{k}: {v}x" for k, v in tool_calls_count.items()]) if tool_calls_count else "None"
    
    last_response = state.get('last_agent_response', '')
    is_incomplete = last_response and 'READY:' not in last_response.upper() and 'ACTION:' not in last_response.upper()
    incomplete_hint = (
        f"\nLƯU Ý: Phản hồi trước của bạn bị ngắt giữa chừng:\n"
        f"'''{last_response}'''\n"
        f"Hãy tiếp tục và hoàn thành với định dạng READY: đầy đủ.\n"
    ) if is_incomplete and state.get('num_steps', 0) > 0 else ""

    history_text = _format_history(state.get('conversation_history', []))

    prompt = f"""
{AGENT_INSTRUCTION}

{tools_list}

LỊCH SỬ HỘI THOẠI (các lượt trước):
{history_text}

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:
{state.get('query')}

CÔNG CỤ ĐÃ SỬ DỤNG:
{tools_used}

QUAN SÁT TỪ CÁC CÔNG CỤ TRƯỚC ĐÓ:
{observations}
{incomplete_hint}
QUAN TRỌNG:
- Tham khảo LỊCH SỬ HỘI THOẠI nếu câu hỏi hiện tại liên quan đến nội dung trước đó.
- Nếu các quan sát ở trên đã chứa thông tin liên quan và đủ để trả lời,
  HÃY TRẢ LỜI NGAY.
- KHÔNG gọi thêm công cụ trừ khi thực sự cần thiết.

Hãy phản hồi ngay bây giờ theo đúng định dạng đã quy định:
"""
    
    response = openai_model.invoke(prompt=prompt)

    # Fix: nếu Groq vẫn cố gọi tool dù đã có observations (lỗi tool_choice=none),
    # kiểm tra xem tool/query này đã được thực thi chưa → force READY
    if response and 'ACTION:' in response:
        observations_list = state.get('tool_observations', [])
        has_real_data = any(
            obs for obs in observations_list
            if 'RESULT:' in obs and 'Error' not in obs
        )
        if has_real_data:
            # Trích tool name từ response để kiểm tra duplicate
            proposed_tool = ''
            for line in response.split('\n'):
                if line.strip().startswith('ACTION:'):
                    proposed_tool = line.split('ACTION:')[1].strip()
                    break
            already_used = any(f'TOOL: {proposed_tool}' in obs for obs in observations_list)
            if already_used:
                logger.warning(
                    f'Groq cố gọi lại tool "{proposed_tool}" đã dùng (lỗi tool_choice). '
                    f'Force READY để tránh hallucinate.'
                )
                response = (
                    "THOUGHT: Đã có đủ thông tin từ công cụ đã chạy trước đó.\n"
                    "READY: Sử dụng kết quả từ tool observations để trả lời câu hỏi của người dùng."
                )

    state['last_agent_response'] = response
    state['num_steps'] = state.get('num_steps', 0) + 1

    print(f'\n--- AGENT STEP {state["num_steps"]} ---')
    print(response)
    print('-'*50)

    return state


def call_tool(state: AgentState) -> AgentState:
    action_text = state.get('last_agent_response', '')
    
    if 'ACTION:' not in action_text:
        state.setdefault('tool_observations', []).append('No ACTION found')
        return state
    
    try:
        tool_name = None
        for line in action_text.split('\n'):
            if line.strip().startswith('ACTION:'):
                tool_name = line.split('ACTION:')[1].strip()
                break
        
        if not tool_name:
            state.setdefault('tool_observations', []).append('Could not extract tool name')
            return state
        
        # Extract arguments
        arguments = {}
        for line in action_text.split('\n'):
            if line.strip().startswith('ARGUMENTS:'):
                args_str = line.split('ARGUMENTS:')[1].strip()
                arguments = json.loads(args_str)
                break
        
        tool_func = Tools.TOOLS_MAPPING_TO_FUNC.get(tool_name)
        
        if not tool_func:
            state.setdefault('tool_observations', []).append(f'Tool {tool_name} not found')
            return state
        
        print(f'\n>>> Executing tool: {tool_name} with args: {arguments}')
        result = tool_func(**arguments)
        
        # In ra các chunk tìm được từ retriever
        if tool_name == 'get_qa_retriever':
            print(f'\n{"="*60}')
            print(f'[RETRIEVED CHUNKS] Query: "{arguments.get("query", "")}"')
            print('='*60)
            if isinstance(result, list):
                for i, chunk in enumerate(result, 1):
                    print(f'\n--- Chunk {i} ---')
                    print(chunk)
                    print()
            else:
                # result là string (đã join sẵn)
                chunks = result.split('\n\n') if isinstance(result, str) else [str(result)]
                for i, chunk in enumerate(chunks, 1):
                    if chunk.strip():
                        print(f'\n--- Chunk {i} ---')
                        print(chunk.strip())
            print('='*60 + '\n')
        
        observation = f'TOOL: {tool_name}\nRESULT: {result}'
        state.setdefault('tool_observations', []).append(observation)
        
        logger.info(f'Tool {tool_name} executed successfully')
        
    except json.JSONDecodeError as e:
        state.setdefault('tool_observations', []).append(f'JSON parsing error: {str(e)}')
        logger.error(f'JSON error: {e}')
    except Exception as e:
        state.setdefault('tool_observations', []).append(f'Tool execution error: {str(e)}')
        logger.error(f'Tool execution error: {e}')
    
    return state
    
    
def generate_answer(state: AgentState) -> AgentState:
    observations_list = state.get('tool_observations', [])
    observations = '\n\n'.join(observations_list)

    # Guard: kiểm tra có dữ liệu thực sự không (có RESULT và không phải lỗi)
    has_real_data = any(
        obs for obs in observations_list
        if 'RESULT:' in obs and 'Error' not in obs and 'not found' not in obs.lower()
    )

    last_response = state.get('last_agent_response', '')
    ready_summary = ''
    if 'READY:' in last_response.upper():
        ready_summary = last_response.split('READY:', 1)[1].strip()

    history_text = _format_history(state.get('conversation_history', []))

    if not has_real_data and not ready_summary:
        no_data_instruction = (
            "Lưu ý: Không có dữ liệu nào được truy xuất thành công. "
            "Hãy trả lời dựa trên kiến thức chung nhưng nói rõ với người dùng rằng "
            "bạn không tìm thấy thông tin cụ thể từ nguồn dữ liệu. "
            "KHÔNG bịa đặt số liệu, tên tổ chức hay chi tiết cụ thể."
        )
    else:
        no_data_instruction = (
            "Chỉ sử dụng thông tin trong DỮ LIỆU ĐÃ THU THẬP bên dưới. "
            "KHÔNG thêm thông tin không có trong dữ liệu."
        )

    prompt = f"""
{ANSWER_INSTRUCTION}

{no_data_instruction}

LỊCH SỬ HỘI THOẠI (các lượt trước để tham khảo ngữ cảnh):
{history_text}

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:
{state.get('query')}

DỮ LIỆU ĐÃ THU THẬP:
{observations if observations else "Không có dữ liệu nào được thu thập."}

TÓM TẮT KẾT QUẢ:
{ready_summary}

Hãy viết câu trả lời cuối cùng:
"""
    answer = openai_model.invoke(prompt=prompt, temperature=0.7)
    state['final_answer'] = answer

    print(f'\n--- FINAL ANSWER (temp=0.7) ---')
    print(answer)
    print('---' * 20)

    return state


def should_continue(state: AgentState) -> str:
    response = state.get("last_agent_response", "").upper()
    num_steps = state.get("num_steps", 0)
    has_observations = bool(state.get("tool_observations"))

    # Nếu response rỗng (Groq lỗi / rate limit)
    if not response.strip():
        # Đã có tool results → generate_answer để tránh hallucinate
        if has_observations:
            print("Routing to GENERATE_ANSWER (empty response but has observations)")
            return "answer"
        # Chưa có gì, thử lại nếu còn bước
        if num_steps < 3:
            print("Routing to AGENT retry (empty response, no observations yet)")
            return "retry"
        print("Routing to GENERATE_ANSWER (empty response, max steps)")
        return "answer"

    if "READY:" in response:
        print("Routing to GENERATE_ANSWER (found READY)")
        return "answer"

    if "ACTION:" in response:
        print("Routing to TOOLS (found ACTION)")
        return "continue"

    if num_steps >= 3:
        print("Routing to GENERATE_ANSWER (max steps reached)")
        return "answer"

    print("Routing to AGENT (incomplete response, retrying)")
    return "retry"


def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node('agent', call_agent)
    workflow.add_node('tools', call_tool)
    workflow.add_node('generate_answer', generate_answer)
    
    workflow.set_entry_point('agent')
    
    workflow.add_conditional_edges(
        'agent',
        should_continue,
        {
            'continue': 'tools',
            'answer': 'generate_answer',
            'retry': 'agent',
        }
    )
    
    workflow.add_edge('tools', 'agent')
    workflow.add_edge('generate_answer', END)
    
    return workflow.compile()


def run_query(query: str, graph, conversation_history: list = None) -> str:
    state = {
        "query": query,
        "last_agent_response": "",
        "tool_observations": [],
        "num_steps": 0,
        "final_answer": "",
        "conversation_history": conversation_history or [],
    }
    
    result = graph.invoke(state)
    return result.get('final_answer') or result.get('last_agent_response', '')

    
def main():

    # graph = build_graph()
    # png_byte = graph.get_graph().draw_mermaid_png()
    # with open('system_graph.png', 'wb') as f:
    #     f.write(png_byte)
    # logger.info('Save graph Successful')

    print("Initializing Chatbot...")
    preload_embedding_model()
    graph = build_graph()
    print("Ready! Type 'quit', 'exit', or 'esc' to stop.")
    print("       Type 'new'  to start a new session.")
    print("       Type 'list' to switch session.\n")

    session = ChatHistory.select_or_create_session()

    conversation_history = ChatHistory.get_history_list(session["messages"])

    while True:
        query = input('User: ').strip()

        if query.lower() in ['quit', 'exit', 'esc']:
            print("Goodbye!")
            break

        if query.lower() == 'new':
            session = ChatHistory.create_session()
            conversation_history = []
            print(f"  ✓ Phiên mới: {session['session_id']}\n")
            continue

        if query.lower() == 'list':
            session = ChatHistory.select_or_create_session()
            conversation_history = ChatHistory.get_history_list(session["messages"])
            continue

        if not query:
            continue

        try:
            response = run_query(
                query=query,
                graph=graph,
                conversation_history=conversation_history,
            )
            print(f'\nBot: {response}')
            print('---' * 20 + '\n')

            # Cập nhật RAM
            conversation_history.append({"role": "user",      "content": query})
            conversation_history.append({"role": "assistant",  "content": response})

            # Lưu xuống file JSON
            session = ChatHistory.append_messages(session, query, response)

        except Exception as e:
            logger.error(f'Error processing query: {e}')
            print(f"Sorry, an error occurred: {e}\n")
        
if __name__ == '__main__':
    main()