import os
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, List
import logging
import requests

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config():
    QDRANT_URL: str = os.getenv('QDRANT_URL')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY')
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME')
    EMBEDDING_MODEL: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    TOP_K: int = 5  
    WEB_SEARCH_NUM: int = 5
    USE_LLM_RERANK: bool = _env_bool("USE_LLM_RERANK", False)

    SERPAPI_KEY: str = os.getenv('SERPAPI_KEY')
    LLAMA_SERVER_URL: str = os.getenv('LLAMA_SERVER_URL')
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class QA_Retriever:
    def __init__(self, config: Config, llm_client=None):
        self.config = config
        self.llm = llm_client # Nhận instance Gemini từ System.py
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL, 
            api_key=config.QDRANT_API_KEY
        )
        self.encoder = SentenceTransformer(
            model_name_or_path=config.EMBEDDING_MODEL,
            device=config.DEVICE
        )

    def _critique_is_rel(self, query: str, context: str) -> float:
        """
        Thực hiện bước 'IsREL' trong Algorithm 1.
        Đánh giá xem đoạn văn bản có thực sự chứa thông tin giải quyết câu hỏi không.
        """
        if not self.llm or not self.config.USE_LLM_RERANK:
            return 1.0 # Nếu không có LLM, mặc định tin vào Vector Score
            
        prompt = f"""
        Nhiệm vụ: Đánh giá mức độ liên quan (IsREL).
        Câu hỏi: {query}
        Tài liệu: {context}
        
        Trả về một số thực từ 0.0 đến 1.0 (1.0 là cực kỳ liên quan, 0.0 là không liên quan).
        Chỉ trả về con số, không giải thích gì thêm.
        """
        try:
            res = self.llm.invoke(prompt)
            return float(res.strip())
        except:
            return 0.5

    def search(self, query: str, top_k: Optional[int] = None) -> Dict:
        try:
            collection = self.config.COLLECTION_NAME
            # 1. Vector Search (Retrieve)
            query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
            
            raw_results = self.qdrant_client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=(top_k or self.config.TOP_K) * 2 if self.config.USE_LLM_RERANK else (top_k or self.config.TOP_K),
            ).points

            # 2. Critique & Re-rank (Algorithm 1: Rank yt based on IsREL)
            scored_results = []
            for point in raw_results:
                text = point.payload.get('text', 'N/A')
                if self.config.USE_LLM_RERANK:
                    # Tính toán điểm kết hợp: Vector Score + LLM Critique
                    rel_score = self._critique_is_rel(query, text)
                    # Công thức Rank: Trọng số 40% Vector, 60% LLM Critique
                    final_score = (point.score * 0.4) + (rel_score * 0.6)
                else:
                    rel_score = 1.0
                    final_score = point.score

                scored_results.append({
                    'text': text,
                    'source': point.payload.get('source', 'N/A'),
                    'page': point.payload.get('page', 'N/A'),
                    'score': final_score,
                    'is_rel': rel_score
                })

            # Sắp xếp lại theo điểm Rank mới
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = scored_results[:(top_k or self.config.TOP_K)]

            context_parts = [
                f"[{i+1}] Score: {r['score']:.3f} | IsREL: {r['is_rel']} | Source: {r['source']}\n{r['text']}"
                for i, r in enumerate(top_results)
            ]

            return {
                'context': '\n\n'.join(context_parts),
                'source': collection,
                'results': top_results,
                'top_score': top_results[0]['score'] if top_results else 0.0
            }
        except Exception as e:
            logger.error(f'QA search ERROR: {str(e)}')
            return {
                'context': f'Error retrieving information: {str(e)}',
                'source': 'Error',
                'results': [],
                'top_score': 0.0
            }
            
            
class WebSearcher:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.api_key = config.SERPAPI_KEY
        self.url = "https://serpapi.com/search"
        
        if not self.api_key:
            logger.warning("SERPAPI_KEY not found in .env file")
    
    def search(self, query: str, add_medical_context: bool = True) -> Dict:

        if not self.api_key:
            return {
                "context": "SerpAPI key not configured. Add SERPAPI_KEY to .env file.",
                "source": "Web Search Error",
                "num_results": 0,
                "results": []
            }
        
        try:
            search_query = f"{query} medical health" if add_medical_context else query
            
            params = {
                "q": search_query,
                "api_key": self.api_key,
                "num": self.config.WEB_SEARCH_NUM,
                "gl": "vn",
                "hl": "en"
            }
            
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            results_details = []
            
            for idx, item in enumerate(data.get("organic_results", [])[:self.config.WEB_SEARCH_NUM], 1):
                result_text = (
                    f"[{idx}] Title: {item.get('title')}\n"
                    f"Source: {item.get('link')}\n"
                    f"Summary: {item.get('snippet')}"
                )
                results.append(result_text)
                
                results_details.append({
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'snippet': item.get('snippet'),
                    'position': idx
                })
            
            context = "\n\n".join(results) if results else "No web results found"
            
            return {
                "context": context,
                "source": "Web Search (SerpAPI)",
                "num_results": len(results),
                "results": results_details
            }
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "context": f"Web search unavailable: {str(e)}",
                "source": "Web Search Error",
                "num_results": 0,
                "results": []
            }

def vsl_restruct(text: str) -> Dict:

    config = get_config()

    if not config.LLAMA_SERVER_URL:
        logger.warning("LLAMA_SERVER_URL not configured")
        return {
            "context": "LLaMA server URL chưa được cấu hình. Thêm LLAMA_SERVER_URL vào file .env.",
            "source": "VSL Translation Error",
            "num_results": 0,
            "original": text
        }

    if not text or not text.strip():
        return {
            "context": "Văn bản đầu vào trống.",
            "source": "VSL Translation Error",
            "num_results": 0,
            "original": text
        }

    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Bạn là một chuyên gia ngôn ngữ, có thể tái cấu trúc câu từ ngôn ngữ nói Tiếng Việt thành cấu trúc câu Ngôn ngữ ký hiệu Việt Nam, hãy chuyển đổi câu sau:"
                },
                {
                    "role": "user",
                    "content": text.strip()
                }
            ],
            "temperature": 0,
            "max_tokens": 128
        }

        logger.info(f"Calling LLaMA server at {config.LLAMA_SERVER_URL} for VSL restruct")
        r = requests.post(
            config.LLAMA_SERVER_URL,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        r.raise_for_status()
        data = r.json()

        result = data["choices"][0]["message"]["content"].strip()
        logger.info(f"VSL restruct successful: '{text[:50]}...' -> '{result[:50]}...'")

        return {
            "context": result,
            "source": "VSL Translation (LLaMA)",
            "num_results": 1,
            "original": text
        }

    except requests.exceptions.ConnectionError:
        msg = f"Không thể kết nối LLaMA server tại {config.LLAMA_SERVER_URL}. Kiểm tra server đã chạy chưa."
        logger.error(msg)
        return {
            "context": msg,
            "source": "VSL Translation Error",
            "num_results": 0,
            "original": text
        }
    except requests.exceptions.Timeout:
        msg = "LLaMA server không phản hồi trong thời gian chờ (30s)."
        logger.error(msg)
        return {
            "context": msg,
            "source": "VSL Translation Error",
            "num_results": 0,
            "original": text
        }
    except (KeyError, IndexError) as e:
        msg = f"Lỗi phân tích phản hồi từ LLaMA server: {str(e)}"
        logger.error(msg)
        return {
            "context": msg,
            "source": "VSL Translation Error",
            "num_results": 0,
            "original": text
        }
    except Exception as e:
        logger.error(f"VSL restruct unexpected error: {e}")
        return {
            "context": f"Lỗi restruct VSL: {str(e)}",
            "source": "VSL Translation Error",
            "num_results": 0,
            "original": text
        }

_config_instance = None
_qa_retriever_instance = None
_web_searcher_instance = None

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def get_qa_retriever_instance() -> QA_Retriever:
    global _qa_retriever_instance
    if _qa_retriever_instance is None:
        _qa_retriever_instance = QA_Retriever(get_config())
    return _qa_retriever_instance

def get_web_searcher_instance() -> WebSearcher:
    global _web_searcher_instance
    if _web_searcher_instance is None:
        _web_searcher_instance = WebSearcher(get_config())
    return _web_searcher_instance

# @tool
# Trong file Tools.py

def get_qa_retriever(query: str, top_k: Optional[int] = None, llm_client=None) -> Dict:
    retriever_instance = get_qa_retriever_instance()
    retriever_instance.llm = llm_client 
    return retriever_instance.search(query, top_k=top_k)

# @tool
def get_web_search(query: str) -> Dict:
    return get_web_searcher_instance().search(query)
    

TOOLS_MAPPING_TO_FUNC = {
    "get_qa_retriever": get_qa_retriever,
    "get_web_search": get_web_search,
    "vsl_restruct": vsl_restruct
}

AGENT_TOOLS_LIST = {
    'TOOLS': [
        {
            'name': 'get_qa_retriever',
            'description': (
                'Tìm kiếm và trích xuất thông tin liên quan từ cơ sở tri thức PDF nội bộ. '
                'Sử dụng khi câu hỏi liên quan đến kiến thức đã được lưu trong tài liệu.'
            ),
            'args': 'query (str)'
        },
        {
            'name': 'get_web_search',
            'description': (
                'Tìm kiếm thông tin trên internet. '
                'Sử dụng khi câu hỏi cần thông tin mới, cập nhật, hoặc không có trong tài liệu nội bộ.'
            ),
            'args': 'query (str)'
        },
        {
            "name": "vsl_restruct",
            "description": "Tái cấu trúc câu tiếng Việt theo ngữ pháp VSL",
            "args": "text (str)"
        }
    ]
}
