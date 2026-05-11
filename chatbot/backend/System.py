import os
import torch
import google.generativeai as genai
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
    GEMINI_API = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'gemini-2.5-flash'
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


class Gemini:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=self.config.GEMINI_API)
        self.llm = genai.GenerativeModel(self.config.GEMINI_MODEL)
        
    def invoke(self, prompt: str, temperature: float = 0) -> str:
        max_retries = self.config.MAX_RETRIES
        last_error = None
        try:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.llm.generate_content(
                        contents=prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=self.config.MAX_OUTPUT_TOKENS,
                        )
                    )
                    return response.text.strip()
                except Exception as e:
                    last_error = e
                    logger.error(f'Gemini ERROR (attempt {attempt}/{max_retries}): {str(e)}')
            return ""
        except Exception as e:
            logger.error(f'Gemini ERROR: {str(e)}')
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
Bước 2: Quyết định:
   - Người dùng muốn dịch/chuyển đổi câu sang VSL → Gọi vsl_translate, truyền đúng câu cần dịch
   - Cần tra cứu kiến thức → Gọi get_qa_retriever (ƯU TIÊN)
   - Không có trong tài liệu → Gọi get_web_search
   - Đã đủ thông tin → Viết READY

ĐỊNH DẠNG ĐẦU RA (BẮT BUỘC):

Nếu đã đủ thông tin:
THOUGHT: Giải thích ngắn gọn vì sao có thể trả lời dựa trên dữ liệu đã có
READY: <tóm tắt ngắn các dữ liệu/kết quả sẽ dùng để trả lời>

Nếu cần sử dụng công cụ:
THOUGHT: Giải thích thông tin nào đang thiếu và công cụ nào sẽ dùng
ACTION: get_qa_retriever | get_web_search | vsl_translate
ARGUMENTS: <JSON hợp lệ>

ARGUMENTS THEO TỪNG CÔNG CỤ:
- get_qa_retriever  → {"query": "câu truy vấn bằng tiếng Việt"}
- get_web_search    → {"query": "câu truy vấn bằng tiếng Việt"}
- vsl_translate     → {"text": "câu tiếng Việt cần dịch sang VSL"}

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
gemini_model = Gemini(config)

Tools.TOOLS_MAPPING_TO_FUNC["get_qa_retriever"] = (
    lambda query, top_k=3: Tools.get_qa_retriever(query, top_k, llm_client=gemini_model)
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
    
    response = gemini_model.invoke(prompt=prompt)
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
    observations = '\n\n'.join(state.get('tool_observations', []))
    
    last_response = state.get('last_agent_response', '')
    ready_summary = ''
    if 'READY:' in last_response.upper():
        ready_summary = last_response.split('READY:', 1)[1].strip()

    history_text = _format_history(state.get('conversation_history', []))

    prompt = f"""
{ANSWER_INSTRUCTION}

LỊCH SỬ HỘI THOẠI (các lượt trước để tham khảo ngữ cảnh):
{history_text}

CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG:
{state.get('query')}

DỮ LIỆU ĐÃ THU THẬP:
{observations}

TÓM TẮT KẾT QUẢ:
{ready_summary}

Hãy viết câu trả lời cuối cùng:
"""
    answer = gemini_model.invoke(prompt=prompt, temperature=0.7)
    state['final_answer'] = answer

    print(f'\n--- FINAL ANSWER (temp=0.7) ---')
    print(answer)
    print('---' * 20)

    return state


def should_continue(state: AgentState) -> str:
    response = state.get("last_agent_response", "").upper()
    if not response.strip():
        print("Routing to GENERATE_ANSWER (empty response)")
        return "answer"
    
    if "READY:" in response:
        print("Routing to GENERATE_ANSWER (found READY)")
        return "answer"

    if "ACTION:" in response:
        print("Routing to TOOLS (found ACTION)")
        return "continue"
    
    if state.get("num_steps", 0) >= 3:
        print("Routing to END (max steps reached)")
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
