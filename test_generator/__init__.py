"""production_pipeline — chunking + indexing giống hệt luồng production.

Flow 1 (DOCX -> chunk mới):
    from production_pipeline.production_chunker import chunk_docx
    chunks = chunk_docx("path/to/file.docx")

Flow 2 (kéo chunk đã index sẵn trong Postgres):
    from production_pipeline.pg_reader import fetch_chunks_for_documents
    chunks = fetch_chunks_for_documents(["doc-id-1", "doc-id-2"])

Ghi dữ liệu vào Postgres (nếu cần tự index, không chỉ đọc):
    from production_pipeline.pg_writer import index_document

Cả 2 flow trả về CÙNG format:
    [{"chunk_id": int, "text": str, "metadata": {...}}, ...]
-> đưa thẳng vào question_generator / candidate_retriever / label_generator /
   answer_generator / exporter hiện có, không cần sửa gì ở các file đó.
"""
from .production_chunker import chunk_docx, chunk_documents
from .pg_reader import fetch_chunks_for_document, fetch_chunks_for_documents
from .pg_writer import index_document

__all__ = [
    "chunk_docx",
    "chunk_documents",
    "fetch_chunks_for_document",
    "fetch_chunks_for_documents",
    "index_document",
]
