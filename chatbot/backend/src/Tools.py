import os
import time
import torch
import signal
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import httpx

load_dotenv()
logger = logging.getLogger(__name__)

# ── Absolute path dựa trên vị trí file (không phụ thuộc CWD) ──
_SRC_DIR = Path(__file__).resolve().parent          # chatbot/backend/src/
_BACKEND_DIR = _SRC_DIR.parent                      # chatbot/backend/
_DEFAULT_CHROMA_PATH = str(_SRC_DIR / 'chroma_db')  # chatbot/backend/src/chroma_db/

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
    CHROMA_PATH: str = os.getenv('CHROMA_PATH', _DEFAULT_CHROMA_PATH)
    EMBEDDING_MODEL: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    TOP_K: int = int(os.getenv('CHROMA_TOP_K', '3'))            # giảm 5→3
    CHROMA_CACHE_TTL: int = int(os.getenv('CHROMA_CACHE_TTL', '300'))  # 5 phút

    SERPAPI_KEY: str = os.getenv('SERPAPI_KEY')
    LLAMA_SERVER_URL: str = os.getenv('LLAMA_SERVER_URL')

    WEB_SEARCH_TIMEOUT: float = 15.0
    LLAMA_TIMEOUT: float = 30.0

    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ChromaRetriever:
    def __init__(self, config: Config):
        self.config = config
        resolved_path = str(Path(config.CHROMA_PATH).resolve())
        logger.info(f"ChromaDB path: {resolved_path}")
        self.client = chromadb.PersistentClient(path=resolved_path)
        _chroma_clients.append(self.client)  # track for cleanup
        self.embedding_func = SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL,
            device=config.DEVICE
        )
        self._collection = None
        self._cache: dict[str, dict] = {}  # query → result cache

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

    @staticmethod
    def _is_noise(text: str) -> bool:
        """Kiểm tra text có phải noise (page number, header rỗng, ...) không."""
        t = text.strip()
        if not t:
            return True
        # Toàn số, dấu câu, khoảng trắng
        if all(c in '0123456789.,;:!?()[]-–— \t\n\r.' for c in t):
            return True
        # Quá ngắn (< 10 ký tự có nghĩa)
        meaningful = sum(1 for c in t if c.isalpha())
        if meaningful < 5:
            return True
        # Header phổ biến từ PDF
        noise_headers = ['các tác giả', 'mục lục', 'lời nói đầu', 'lời mở đầu',
                         'phụ lục', 'tài liệu tham khảo', 'danh mục', 'bảng']
        if t.lower().strip() in noise_headers:
            return True
        return False

    def search(self, query: str, top_k: Optional[int] = None, search_word: Optional[str] = None) -> Dict:
        k = top_k or self.config.TOP_K
        cache_key = f"{query.strip().lower()}_{k}_{search_word or ''}"

        # Check cache
        cached_entry = self._cache.get(cache_key)
        if cached_entry:
            now = time.time()
            if (now - cached_entry.get('time', 0)) < self.config.CHROMA_CACHE_TTL:
                logger.info(f"ChromaDB cache HIT for: {query[:60]}...")
                return cached_entry['data']

        col = self.collection
        if not col:
            return {
                'context': 'Thư viện tri thức chưa được khởi tạo. Vui lòng chạy Create_vectorDB.py trước.',
                'source': 'ChromaDB Error',
                'results': [],
                'top_score': 0.0
            }

        try:
            all_results = []
            seen_texts = set()
            # search_word được truyền từ WordExtractor (LLM) bên System.py
            # Nếu None, bỏ qua Phase 1 exact match, chỉ chạy vector search

            # ── Phase 1: Exact word match (từ JSON từ điển) ──
            if search_word:
                try:
                    # Thử exact match với search_word gốc
                    exact_results = col.query(
                        query_texts=[query],
                        n_results=max(k * 2, 10),
                        where={'word': search_word},
                        include=['documents', 'metadatas', 'distances']
                    )
                    has_exact = (exact_results['documents']
                                 and exact_results['documents'][0])
                    if has_exact:
                        for doc, meta, dist in zip(
                            exact_results['documents'][0],
                            exact_results['metadatas'][0],
                            exact_results['distances'][0]
                        ):
                            score = 1.0 - dist if dist <= 1.0 else 0.0
                            # Exact word match luôn đáng tin, dùng threshold thấp hơn
                            if score >= 0.15:
                                key = doc[:100]  # dedup
                                if key not in seen_texts:
                                    seen_texts.add(key)
                                    all_results.append({
                                        'text': doc,
                                        'source': meta.get('source', 'JSON'),
                                        'page': meta.get('page', 'N/A'),
                                        'score': score + 0.5,  # boost cho exact match
                                    })
                                    logger.info(f"Exact word match: '{search_word}' (score boosted: {score + 0.5:.3f})")

                    # Fallback: nếu exact match không có kết quả và search_word có nhiều từ,
                    # thử từ cuối cùng (thường là danh từ chính, VD: "núi" từ "ngọn núi")
                    if not has_exact and len(search_word.split()) > 1:
                        fallback_word = search_word.split()[-1]
                        if fallback_word != search_word:
                            logger.info(f"Exact match '{search_word}' không có, thử fallback: '{fallback_word}'")
                            fallback_results = col.query(
                                query_texts=[query],
                                n_results=max(k * 2, 10),
                                where={'word': fallback_word},
                                include=['documents', 'metadatas', 'distances']
                            )
                            if fallback_results['documents'] and fallback_results['documents'][0]:
                                for doc, meta, dist in zip(
                                    fallback_results['documents'][0],
                                    fallback_results['metadatas'][0],
                                    fallback_results['distances'][0]
                                ):
                                    score = 1.0 - dist if dist <= 1.0 else 0.0
                                    if score >= 0.15:
                                        key = doc[:100]
                                        if key not in seen_texts:
                                            seen_texts.add(key)
                                            all_results.append({
                                                'text': doc,
                                                'source': meta.get('source', 'JSON'),
                                                'page': meta.get('page', 'N/A'),
                                                'score': score + 0.4,  # boost thấp hơn 1 chút
                                            })
                                            logger.info(f"Fallback word match: '{fallback_word}' (score boosted: {score + 0.4:.3f})")
                except Exception as e:
                    logger.warning(f"Exact word match error: {e}")

            # ── Phase 2: Vector search (với noise filtering) ──
            vec_results = col.query(
                query_texts=[query],
                n_results=k * 3,  # lấy nhiều hơn để lọc noise
                include=['documents', 'metadatas', 'distances']
            )

            if vec_results['documents'] and vec_results['documents'][0]:
                for doc, meta, dist in zip(
                    vec_results['documents'][0],
                    vec_results['metadatas'][0],
                    vec_results['distances'][0]
                ):
                    score = 1.0 - dist if dist <= 1.0 else 0.0
                    # Lọc noise
                    if self._is_noise(doc):
                        continue
                    key = doc[:100]
                    if key not in seen_texts:
                        seen_texts.add(key)
                        all_results.append({
                            'text': doc,
                            'source': meta.get('source', 'N/A'),
                            'page': meta.get('page', 'N/A'),
                            'score': score,
                        })

            # Sắp xếp theo score, lấy top_k
            all_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = all_results[:k]

            if not top_results:
                return {
                    'context': 'Không tìm thấy thông tin liên quan.',
                    'source': self.config.COLLECTION_NAME,
                    'results': [],
                    'top_score': 0.0
                }

            context_parts = [
                f"[{i+1}] Score: {r['score']:.3f} | Source: {r['source']}\n{r['text']}"
                for i, r in enumerate(top_results)
            ]

            result = {
                'context': '\n\n'.join(context_parts),
                'source': self.config.COLLECTION_NAME,
                'results': top_results,
                'top_score': top_results[0]['score'] if top_results else 0.0
            }

            # Cache + TTL invalidation
            now = time.time()
            self._cache[cache_key] = {
                'data': result,
                'time': now,
            }
            self._invalidate_cache(now)
            logger.info(f"ChromaDB cached: {query[:60]}... ({len(top_results)} results)")

            return result

        except Exception as e:
            logger.error(f'ChromaDB search error: {str(e)}')
            return {
                'context': f'Lỗi truy xuất thông tin: {str(e)}',
                'source': 'Error',
                'results': [],
                'top_score': 0.0
            }

    def _invalidate_cache(self, now: float):
        ttl = self.config.CHROMA_CACHE_TTL
        expired = [k for k, v in self._cache.items()
                   if (now - v.get('time', 0)) >= ttl]
        for k in expired:
            del self._cache[k]


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

    async def search(self, query: str) -> Dict:
        cache_key = query.strip().lower()
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
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": 3,
                "gl": "vn",
                "hl": "en"
            }

            client = await self._get_client()
            response = await client.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            results_details = []
            links = []

            for idx, item in enumerate(data.get("organic_results", [])[:5], 1):
                title = item.get('title', '')
                link = item.get('link', '')
                snippet = item.get('snippet', '')

                result_text = (
                    f"[{idx}] {title}\n"
                    f"  Link: {link}\n"
                    f"  Nội dung: {snippet}"
                )
                results.append(result_text)
                links.append({'title': title, 'url': link})

                results_details.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet,
                    'position': idx
                })

            context = "\n\n".join(results) if results else "No web results found"
            links_text = "\n".join(f"- {l['title']}: {l['url']}" for l in links) if links else ""

            result = {
                "context": context,
                "source": "Web Search (SerpAPI)",
                "num_results": len(results),
                "results": results_details,
                "links": links,
                "links_text": links_text,
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
                    {"role": "user", "content": text.strip()}
                ],
                "temperature": 0,
                "max_tokens": 256
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

# ChromaDB client(s) cần cleanup — lưu riêng để đóng chính xác
_chroma_clients: list[chromadb.PersistentClient] = []


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
    global _web_searcher_instance, _vsl_restructurer_instance, \
           _chroma_retriever_instance, _chroma_clients
    logger.info("Tools cleanup...")

    # 1. Đóng HTTP clients (WebSearcher, VSL_Restructurer)
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

    # 2. Đóng ChromaDB clients
    for client in _chroma_clients:
        try:
            # ChromaDB's PersistentClient không có close() rõ ràng,
            # nhưng xoá reference để GC giải phóng file handles + mmap
            del client
        except Exception as e:
            logger.warning(f"Lỗi cleanup chroma client: {e}")
    _chroma_clients.clear()
    _chroma_retriever_instance = None

    # 3. Clear tất cả cache
    _search_cache.clear()
    logger.info("All search caches cleared.")

    logger.info("Tools cleanup done.")


def cleanup_sync():
    """Phiên bản synchronous cho các context không có event loop."""
    global _web_searcher_instance, _vsl_restructurer_instance, \
           _chroma_retriever_instance, _chroma_clients

    _search_cache.clear()

    for client in _chroma_clients:
        try:
            del client
        except Exception:
            pass
    _chroma_clients.clear()
    _chroma_retriever_instance = None
    _web_searcher_instance = None
    _vsl_restructurer_instance = None
    logger.info("Tools cleanup_sync done.")


def release_port(port: int, host: str = "0.0.0.0"):
    """Giải phóng port đang chiếm giữ bằng cách:
    1. Tìm PID đang listen trên port đó
    2. Gửi SIGTERM → chờ → SIGKILL nếu cần
    """
    import subprocess
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"Port {port} released via fuser.")
            return
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"fuser on port {port} failed: {e}")

    # Fallback: lsof
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split()
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to PID {pid} on port {port}")
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.warning(f"Kill PID {pid} on port {port} error: {e}")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"lsof on port {port} failed: {e}")
    logger.info(f"Port {port} release attempted.")


def get_qa_retriever(query: str, top_k: Optional[int] = None, search_word: Optional[str] = None) -> Dict:
    retriever = get_chroma_retriever()
    return retriever.search(query, top_k=top_k, search_word=search_word)


async def get_web_search(query: str) -> Dict:
    searcher = await get_web_searcher_instance()
    return await searcher.search(query)


async def vsl_restruct(text: str) -> Dict:
    restructurer = await get_vsl_restructurer_instance()
    return await restructurer.restruct(text)


TOOLS_MAPPING_TO_FUNC_ASYNC = {
    "get_qa_retriever": get_qa_retriever,
    "get_web_search": get_web_search,
    "vsl_restruct": vsl_restruct,
}