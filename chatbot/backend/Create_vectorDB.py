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
        metadatas.append({
            'source': chunk.metadata.get('source', ''),
            'page': chunk.metadata.get('page', -1),
            'chunk_index': i,
        })
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
    parser = argparse.ArgumentParser(description='Tạo ChromaDB vector database từ PDF')
    parser.add_argument('--pdf_dir', type=str, default='./data_vsl',
                        help='Đường dẫn thư mục chứa file PDF (mặc định: ./data)')
    parser.add_argument('--collection', type=str, default=COLLECTION_NAME,
                        help=f'Tên collection (mặc định: {COLLECTION_NAME})')
    parser.add_argument('--chroma_path', type=str, default=CHROMA_PATH,
                        help=f'Đường dẫn lưu ChromaDB (mặc định: {CHROMA_PATH})')
    parser.add_argument('--threshold_type', type=str, default='percentile',
                        choices=['percentile', 'standard_deviation', 'interquartile'],
                        help='Kiểu threshold cho semantic chunking')
    parser.add_argument('--threshold_amount', type=float, default=95.0,
                        help='Ngưỡng cho semantic chunking (mặc định: 95.0)')

    args = parser.parse_args()

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