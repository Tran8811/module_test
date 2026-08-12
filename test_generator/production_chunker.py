# production_chunker.py
"""
Chunk giống hệt luồng indexing production. Có 2 hàm entry point:

  - chunk_docx(path)          : FLOW dành cho 1 file .docx thật (có bảng HTML
                                 thật, đọc bằng python-docx).
  - chunk_documents(documents): thay thế TRỰC TIẾP cho
                                 split_documents_hierarchical(documents) cũ —
                                 nhận list Document (page_content + metadata)
                                 như PyPDFLoader trả về, ví dụ input hiện tại
                                 của main.py (load_pdf() trên file .pdf).

Cả 2 hàm trả về cùng format:
    list[{"chunk_id": int, "text": str, "metadata": {...}}]
"""
import os

from .docx_parser import extract_docx_lines
from .toc_detector import detect_heading_levels
from .tree_builder import build_tree
from .tree_postprocess import insert_tables_and_split, process_tree
from .tree_to_chunks import get_ancestor_breadcrumb, get_list_leaf, tree_to_chunks
from .utils import assign_chunk_ids


def chunk_docx(path: str, source_name: str | None = None) -> list[dict]:
    """FLOW dành cho 1 file .docx (có bảng HTML thật)."""
    source_name = source_name or os.path.basename(path)

    # Bước 1: đọc DOCX theo đúng thứ tự (đoạn văn + bảng)
    lines, tables = extract_docx_lines(path)

    # Bước 2: LLM suy luận cấp tiêu đề cho từng dòng (thay regex)
    levels = detect_heading_levels(lines)

    # Bước 3: dựng cây theo level (thuật toán stack, giống build_tree_from_df)
    root = build_tree(lines, levels, source_name)

    # Bước 4: chèn bảng thật vào leaf, cắt leaf quá dài
    insert_tables_and_split(root, tables)

    # Bước 5: gộp/tối ưu cây (merge_with_parent/sibling/header + format)
    process_tree(root)

    # Bước 6: duyệt leaf, ghép breadcrumb, convert sang chunk dict
    return tree_to_chunks(root)


def _group_pages_by_source(documents) -> list[tuple[str, str]]:
    """PyPDFLoader trả 1 Document/trang, cùng 1 file có cùng metadata.source.
    Gộp lại thành TOÀN VĂN theo đúng thứ tự trang, để dựng cây trên cả file
    (giữ mạch tiêu đề xuyên trang) thay vì cắt rời từng trang như hàm cũ."""
    order: list[str] = []
    pages: dict[str, list[str]] = {}

    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("file_name") or meta.get("path") or "unknown"
        if source not in pages:
            pages[source] = []
            order.append(source)
        pages[source].append(doc.page_content or "")

    return [(source, "\n".join(pages[source])) for source in order]


def _text_to_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def chunk_documents(documents) -> list[dict]:
    """Thay thế trực tiếp cho split_documents_hierarchical(documents) cũ.

    Nhận list Document (đầu ra của load_pdf() hoặc bất kỳ loader nào có
    .page_content/.metadata). KHÔNG có bảng HTML thật (PyPDFLoader không
    giữ cấu trúc bảng) — đây là giới hạn giống hệt bản chunker regex cũ,
    không phải lỗi mới phát sinh.
    """
    all_chunks: list[dict] = []

    for source, full_text in _group_pages_by_source(documents):
        lines = _text_to_lines(full_text)
        if not lines:
            continue

        levels = detect_heading_levels(lines)
        root = build_tree(lines, levels, source_name=source)
        # không có bảng thật -> insert_tables_and_split chỉ đóng vai trò
        # "cắt leaf quá dài", tables truyền rỗng
        insert_tables_and_split(root, tables={})
        process_tree(root)

        for leaf in get_list_leaf(root):
            breadcrumb = get_ancestor_breadcrumb(leaf)
            full_text_chunk = f"[{breadcrumb}]\n{leaf.content}" if breadcrumb else leaf.content
            all_chunks.append(
                {
                    "chunk_id": None,
                    "text": full_text_chunk,
                    "metadata": {
                        "source": source,
                        "file_name": os.path.basename(source),
                        "breadcrumb": breadcrumb,
                        "node_id": leaf.id,
                    },
                }
            )

    # assign_chunk_ids GỌI MỘT LẦN DUY NHẤT ở cuối, sau khi đã gộp hết các
    # file -> chunk_id không bị trùng giữa các file (xem lưu ý trong README).
    return assign_chunk_ids(all_chunks)
