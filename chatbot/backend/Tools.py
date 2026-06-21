import os
import time
import torch
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import httpx

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_CACHE_TTL = int(os.getenv('SEARCH_CACHE_TTL', '300'))
_search_cache: Dict[str, dict] = {}

def _get_cached(key: str) -> dict | None:
    entry = _search_cache.get(key)
    if entry and (time.time() - entry["time"]) < _CACHE_TTL:
        logger.info(f"Web search cache HIT for: {key[:60]}...")
        return entry["data"]
    return None

def _set_cache(key: str, data: dict):
    _search_cache[key] = {"data": data, "time": time.time()}

def _invalidate_cache():
    now = time.time()
    expired = [k for k, v in _search_cache.items() if (now - v["time"]) >= _CACHE_TTL]
    for k in expired:
        del _search_cache[k]
    if expired:
        logger.info(f"Cleared {len(expired)} expired cache entries")


class Config:
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME', 'vsl_knowledge_base')
    CHROMA_PATH: str = os.getenv('CHROMA_PATH', './chroma_db')
    EMBEDDING_MODEL: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    TOP_K: int = 3

    SERPAPI_KEY: str = os.getenv('SERPAPI_KEY')
    LLAMA_SERVER_URL: str = os.getenv('LLAMA_SERVER_URL')

    WEB_SEARCH_TIMEOUT: float = 10.0
    LLAMA_TIMEOUT: float = 30.0

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChromaRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.embedding_func = SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL,
            device=config.DEVICE
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(
                    name=self.config.COLLECTION_NAME,
                    embedding_function=self.embedding_func
                )
            except Exception as e:
                logger.warning(f"Collection '{self.config.COLLECTION_NAME}' not found: {e}")
                self._collection = None
        return self._collection

    def search(self, query: str, top_k: Optional[int] = None) -> Dict:
        k = top_k or self.config.TOP_K
        col = self.collection
        if not col:
            return {
                'context': 'Thư viện tri thức chưa được khởi tạo. Vui lòng chạy Create_vectorDB.py trước.',
                'source': 'ChromaDB Error',
                'results': [],
                'top_score': 0.0
            }

        try:
            results = col.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )

            if not results['documents'] or not results['documents'][0]:
                return {
                    'context': 'Không tìm thấy thông tin liên quan.',
                    'source': self.config.COLLECTION_NAME,
                    'results': [],
                    'top_score': 0.0
                }

            scored_results = []
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                score = 1.0 - dist if dist <= 1.0 else 0.0
                scored_results.append({
                    'text': doc,
                    'source': meta.get('source', 'N/A'),
                    'page': meta.get('page', 'N/A'),
                    'score': score,
                })

            scored_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = scored_results[:k]

            context_parts = [
                f"[{i+1}] Score: {r['score']:.3f} | Source: {r['source']}\n{r['text']}"
                for i, r in enumerate(top_results)
            ]

            return {
                'context': '\n\n'.join(context_parts),
                'source': self.config.COLLECTION_NAME,
                'results': top_results,
                'top_score': top_results[0]['score'] if top_results else 0.0
            }

        except Exception as e:
            logger.error(f'ChromaDB search error: {str(e)}')
            return {
                'context': f'Lỗi truy xuất thông tin: {str(e)}',
                'source': 'Error',
                'results': [],
                'top_score': 0.0
            }


class WebSearcher:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.api_key = config.SERPAPI_KEY
        self.url = "https://serpapi.com/search"
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key:
            logger.warning("SERPAPI_KEY not found in .env file")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.WEB_SEARCH_TIMEOUT)
        return self._client

    async def search(self, query: str, add_medical_context: bool = True) -> Dict:
        cache_key = f"{query}::med={add_medical_context}"
        cached = _get_cached(cache_key)
        if cached:
            return cached

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
                "num": 5,
                "gl": "vn",
                "hl": "en"
            }

            client = await self._get_client()
            response = await client.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            results_details = []

            for idx, item in enumerate(data.get("organic_results", [])[:5], 1):
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

            result = {
                "context": context,
                "source": "Web Search (SerpAPI)",
                "num_results": len(results),
                "results": results_details
            }

            _set_cache(cache_key, result)
            _invalidate_cache()
            logger.info(f"Web search cached for: {query[:60]}...")

            return result

        except httpx.TimeoutException:
            logger.error("Web search timeout")
            return {
                "context": "Web search timeout",
                "source": "Web Search Error",
                "num_results": 0,
                "results": []
            }
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {
                "context": f"Web search unavailable: {str(e)}",
                "source": "Web Search Error",
                "num_results": 0,
                "results": []
            }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


class VSL_Restructurer:
    def __init__(self, config: Config = Config()):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.LLAMA_TIMEOUT)
        return self._client

    async def restruct(self, text: str) -> Dict:
        if not self.config.LLAMA_SERVER_URL:
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

            logger.info(f"Calling LLaMA server at {self.config.LLAMA_SERVER_URL} for VSL restruct")
            client = await self._get_client()
            r = await client.post(
                self.config.LLAMA_SERVER_URL,
                json=payload,
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

        except httpx.ConnectError:
            msg = f"Không thể kết nối LLaMA server tại {self.config.LLAMA_SERVER_URL}. Kiểm tra server đã chạy chưa."
            logger.error(msg)
            return {
                "context": msg,
                "source": "VSL Translation Error",
                "num_results": 0,
                "original": text
            }
        except httpx.TimeoutException:
            msg = f"LLaMA server không phản hồi trong thời gian chờ ({self.config.LLAMA_TIMEOUT}s)."
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

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton instances
_config_instance: Optional[Config] = None
_chroma_retriever_instance: Optional[ChromaRetriever] = None
_web_searcher_instance: Optional[WebSearcher] = None
_vsl_restructurer_instance: Optional[VSL_Restructurer] = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def get_chroma_retriever() -> ChromaRetriever:
    global _chroma_retriever_instance
    if _chroma_retriever_instance is None:
        _chroma_retriever_instance = ChromaRetriever(get_config())
    return _chroma_retriever_instance


async def get_web_searcher_instance() -> WebSearcher:
    global _web_searcher_instance
    if _web_searcher_instance is None:
        _web_searcher_instance = WebSearcher(get_config())
    return _web_searcher_instance


async def get_vsl_restructurer_instance() -> VSL_Restructurer:
    global _vsl_restructurer_instance
    if _vsl_restructurer_instance is None:
        _vsl_restructurer_instance = VSL_Restructurer(get_config())
    return _vsl_restructurer_instance


def preload():
    logger.info("Preloading ChromaDB + embedding model...")
    try:
        retriever = get_chroma_retriever()
        col = retriever.collection
        if col is not None:
            _ = col.query(query_texts=["test"], n_results=1)
            count = col.count()
            logger.info(f"ChromaDB sẵn sàng ({count} documents trong collection)")
        else:
            logger.warning("ChromaDB collection chưa được tạo. Chạy Create_vectorDB.py trước.")
    except Exception as e:
        logger.error(f"Lỗi preload ChromaDB: {e}")
        logger.warning("ChromaDB sẽ load lazy khi có request đầu tiên.")


async def cleanup():
    global _web_searcher_instance, _vsl_restructurer_instance
    logger.info("Tools cleanup...")

    if _web_searcher_instance is not None:
        try:
            await _web_searcher_instance.close()
        except Exception as e:
            logger.warning(f"Lỗi cleanup web searcher: {e}")
        _web_searcher_instance = None

    if _vsl_restructurer_instance is not None:
        try:
            await _vsl_restructurer_instance.close()
        except Exception as e:
            logger.warning(f"Lỗi cleanup vsl restructurer: {e}")
        _vsl_restructurer_instance = None

    logger.info("Tools cleanup done.")


def get_qa_retriever(query: str, top_k: Optional[int] = None) -> Dict:
    retriever = get_chroma_retriever()
    return retriever.search(query, top_k=top_k)


async def get_web_search(query: str) -> Dict:
    searcher = await get_web_searcher_instance()
    return await searcher.search(query)


async def vsl_restruct(text: str) -> Dict:
    restructurer = await get_vsl_restructurer_instance()
    return await restructurer.restruct(text)


def get_web_search_sync(query: str) -> Dict:
    import asyncio
    return asyncio.run(get_web_search(query))


def vsl_restruct_sync(text: str) -> Dict:
    import asyncio
    return asyncio.run(vsl_restruct(text))


TOOLS_MAPPING_TO_FUNC = {
    "get_qa_retriever": get_qa_retriever,
    "get_web_search": get_web_search_sync,
    "vsl_restruct": vsl_restruct_sync,
}

TOOLS_MAPPING_TO_FUNC_ASYNC = {
    "get_qa_retriever": get_qa_retriever,
    "get_web_search": get_web_search,
    "vsl_restruct": vsl_restruct,
}

AGENT_TOOLS_LIST = {
    'TOOLS': [
        {
            'name': 'get_qa_retriever',
            'description': (
                'Tìm kiếm và trích xuất thông tin liên quan từ cơ sở tri thức PDF nội bộ (ChromaDB local). '
                'Sử dụng khi câu hỏi liên quan đến kiến thức đã được lưu trong tài liệu VSL.'
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
            "description": "Tái cấu trúc câu tiếng Việt theo ngữ pháp VSL (Ngôn ngữ ký hiệu Việt Nam)",
            "args": "text (str)"
        }
    ]
}
