import json
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'vsl_knowledge_base')
JSON_COLLECTION_NAME = os.getenv('JSON_COLLECTION_NAME', 'vsl_json_entries')


def load_pdf_files(path_dir: Path) -> List:
    if not Path(path_dir).exists():
        logger.error(f'Thư mục {path_dir} không tồn tại')
        return []

    pdf_files = list(Path(path_dir).glob('*.pdf'))
    logger.info(f'Tìm thấy {len(pdf_files)} file PDF: {[f.name for f in pdf_files]}')

    loader = DirectoryLoader(
        path=str(path_dir),
        glob='*.pdf',
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    logger.info(f'Đã tải {len(documents)} trang từ PDF')

    return documents


class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def load_json_data(json_path: Path) -> List[Document]:
    if not json_path.exists():
        logger.error(f'File JSON {json_path} không tồn tại')
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error('JSON phải là một mảng (array) các entry')
        return []

    logger.info(f'Đã tải {len(data)} mẫu từ JSON')

    documents = []
    for i, entry in enumerate(data):
        tu = entry.get('Từ', '')
        mo_ta = entry.get('Biểu diễn / Mô tả hành động', '')
        loai = entry.get('Loại', '')
        khu_vuc = entry.get('Khu vực', '')

        content = (
            f"Từ: {tu}\n"
            f"Biểu diễn / Mô tả hành động: {mo_ta}\n"
            f"Loại: {loai}\n"
            f"Khu vực: {khu_vuc}"
        )

        metadata = {
            'source': str(json_path),
            'word': tu,
            'category': loai,
            'region': khu_vuc,
            'entry_index': i,
        }

        documents.append(Document(page_content=content, metadata=metadata))

    logger.info(f'Đã tạo {len(documents)} chunks từ JSON (mỗi mẫu là 1 chunk)')
    return documents


def semantic_chunking(
    documents: List,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> List:
    logger.info(
        f'Đang chia chunk semantic | model="{EMBEDDING_MODEL}" | '
        f'threshold={breakpoint_threshold_amount}'
    )

    model_kwargs = {'device': DEVICE}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
    )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    chunks = splitter.split_documents(documents)
    logger.info(f'Đã tạo {len(chunks)} chunks')
    return chunks


def create_chromadb(chunks: List, collection_name: str, chroma_path: str):
    os.makedirs(chroma_path, exist_ok=True)

    client = chromadb.PersistentClient(path=chroma_path)
    logger.info(f'ChromaDB persistent path: {chroma_path}')

    embedding_func = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=DEVICE
    )

    try:
        client.delete_collection(collection_name)
        logger.info(f'Đã xoá collection cũ: {collection_name}')
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
    )
    logger.info(f'Đã tạo collection: {collection_name}')

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        meta = dict(chunk.metadata)  # copy all original metadata
        meta.setdefault('source', '')
        meta.setdefault('page', -1)
        meta['chunk_index'] = i
        metadatas.append(meta)
        ids.append(f'chunk_{i}')

    batch_size = 64
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        logger.info(f'Đã thêm {end}/{len(documents)} chunks vào ChromaDB')

    logger.info(f'Hoàn tất! Đã lưu {len(documents)} chunks vào ChromaDB collection "{collection_name}"')

    logger.info('Đang kiểm tra truy vấn thử...')
    test_results = collection.query(
        query_texts=["Ngôn ngữ ký hiệu Việt Nam"],
        n_results=1,
    )
    if test_results['documents'] and test_results['documents'][0]:
        logger.info(f'✓ Truy vấn thử thành công. Kết quả đầu tiên: {test_results["documents"][0][0][:100]}...')
    else:
        logger.warning('Truy vấn thử không có kết quả.')


def main():
    parser = argparse.ArgumentParser(description='Tạo ChromaDB vector database từ PDF, JSON hoặc cả hai')
    parser.add_argument('--data_type', type=str, default='pdf',
                        choices=['pdf', 'json', 'both'],
                        help='Loại dữ liệu nguồn: pdf (mặc định), json, hoặc both (gộp cả 2)')
    parser.add_argument('--pdf_dir', type=str, default='./data_vsl',
                        help='Đường dẫn thư mục chứa file PDF (dùng khi --data_type=pdf hoặc both)')
    parser.add_argument('--json_file', type=str, default='./data_vsl/sign_language_data.json',
                        help='Đường dẫn file JSON (dùng khi --data_type=json hoặc both)')
    parser.add_argument('--collection', type=str, default=None,
                        help='Tên collection (mặc định: vsl_knowledge_base cho pdf/both, vsl_json_entries cho json)')
    parser.add_argument('--chroma_path', type=str, default=CHROMA_PATH,
                        help=f'Đường dẫn lưu ChromaDB (mặc định: {CHROMA_PATH})')
    parser.add_argument('--threshold_type', type=str, default='percentile',
                        choices=['percentile', 'standard_deviation', 'interquartile'],
                        help='Kiểu threshold cho semantic chunking (chỉ dùng với PDF)')
    parser.add_argument('--threshold_amount', type=float, default=95.0,
                        help='Ngưỡng cho semantic chunking (chỉ dùng với PDF)')

    args = parser.parse_args()

    # Chọn collection name mặc định theo data_type
    if args.collection is None:
        if args.data_type == 'json':
            args.collection = JSON_COLLECTION_NAME
        else:
            args.collection = COLLECTION_NAME

    if args.data_type == 'json':
        # ---- XỬ LÝ JSON ----
        json_path = Path(args.json_file)
        if not json_path.exists():
            logger.error(f'File JSON không tồn tại: {json_path}')
            return

        chunks = load_json_data(json_path)
        if not chunks:
            logger.error('Không có dữ liệu JSON để xử lý.')
            return

    elif args.data_type == 'both':
        # ---- GỘP CẢ PDF + JSON VÀO CHUNG 1 COLLECTION ----
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            logger.error(f'Thư mục PDF không tồn tại: {pdf_dir}')
            return
        pdf_documents = load_pdf_files(pdf_dir)
        if not pdf_documents:
            logger.error('Không tìm thấy PDF nào.')
            return
        pdf_chunks = semantic_chunking(
            documents=pdf_documents,
            breakpoint_threshold_type=args.threshold_type,
            breakpoint_threshold_amount=args.threshold_amount,
        )

        json_path = Path(args.json_file)
        if not json_path.exists():
            logger.error(f'File JSON không tồn tại: {json_path}')
            return
        json_chunks = load_json_data(json_path)
        if not json_chunks:
            logger.error('Không có dữ liệu JSON.')
            return

        # Gộp: PDF chunks + JSON chunks
        chunks = pdf_chunks + json_chunks
        logger.info(f'=== TỔNG KẾT GỘP: {len(pdf_chunks)} chunks PDF + {len(json_chunks)} chunks JSON = {len(chunks)} chunks ===')

    else:
        # ---- XỬ LÝ PDF ----
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.exists():
            logger.error(f'Thư mục PDF không tồn tại: {pdf_dir}')
            logger.info('Tạo thư mục và đặt file PDF vào đó, sau đó chạy lại.')
            os.makedirs(pdf_dir, exist_ok=True)
            return

        documents = load_pdf_files(pdf_dir)
        if not documents:
            logger.error('Không tìm thấy PDF nào. Hãy đặt file PDF vào thư mục.')
            return

        chunks = semantic_chunking(
            documents=documents,
            breakpoint_threshold_type=args.threshold_type,
            breakpoint_threshold_amount=args.threshold_amount,
        )

    create_chromadb(
        chunks=chunks,
        collection_name=args.collection,
        chroma_path=args.chroma_path,
    )

    logger.info('Hoàn tất quá trình tạo vector database!')


if __name__ == '__main__':
    main()