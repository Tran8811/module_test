# pg_writer.py
"""
Tương đương tree_to_pgvector() + create_bm25_index() bên production:
ghi toàn bộ cây (root_node từ production_chunker) vào bảng nodes/node_links.

    from production_chunker import chunk_docx  # chỉ để tham khảo, KHÔNG dùng
    # -> muốn ghi DB cần đi từ cây (TreeNode), không phải list chunk dict
    # đã "làm phẳng" -> gọi index_document() dùng thẳng root_node, xem
    # ví dụ cuối file.

Embedding: mặc định để trống (embed_fn=None) — bạn truyền vào 1 hàm
`embed_fn(text: str) -> list[float]` gọi tới embedding endpoint thật của
bạn. Nếu không truyền, node lá vẫn được ghi nhưng embedding=NULL (không
search vector được, chỉ full-text/BM25).
"""
import uuid
from typing import Callable, Optional

import psycopg2
import psycopg2.extras

from .pg_config import PG_CONN_PARAMS
from .tree_node import TreeNode
from .tree_to_chunks import get_ancestor_breadcrumb, get_list_leaf


def _insert_document(cur, document_id: str, bot_id: str, file_name: str):
    cur.execute(
        "INSERT INTO documents (id, bot_id, file_name, active) VALUES (%s, %s, %s, TRUE) "
        "ON CONFLICT (id) DO NOTHING",
        (document_id, bot_id, file_name),
    )


def _insert_parent_node(cur, node_id: str, document_id: str, content: str):
    cur.execute(
        "INSERT INTO nodes (id, document_id, content, embedding) VALUES (%s, %s, %s, NULL) "
        "ON CONFLICT (id) DO NOTHING",
        (node_id, document_id, content),
    )


def _insert_leaf_node(cur, node_id: str, document_id: str, content: str, embedding: Optional[list]):
    cur.execute(
        "INSERT INTO nodes (id, document_id, content, embedding) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (id) DO NOTHING",
        (node_id, document_id, content, embedding),
    )


def _insert_node_link(cur, parent_id: str, child_id: str, order_index: int):
    cur.execute(
        "INSERT INTO node_links (parent_id, child_id, order_index) VALUES (%s, %s, %s) "
        "ON CONFLICT (parent_id, child_id, order_index) DO NOTHING",
        (parent_id, child_id, order_index),
    )


def _write_tree(cur, node: TreeNode, document_id: str, embed_fn: Optional[Callable[[str], list]]):
    if node.child:
        _insert_parent_node(cur, node.id, document_id, node.content)
        for child in node.child:
            _write_tree(cur, child, document_id, embed_fn)
            _insert_node_link(cur, node.id, child.id, child.order_index)
    else:
        # Node lá: giống production, embed = breadcrumb tổ tiên + nội dung
        # chính nó, để vector "biết" ngữ cảnh phân cấp.
        embedding = None
        if embed_fn is not None:
            breadcrumb = get_ancestor_breadcrumb(node)
            text_to_embed = f"{breadcrumb}\n{node.content}" if breadcrumb else node.content
            embedding = embed_fn(text_to_embed)
        _insert_leaf_node(cur, node.id, document_id, node.content, embedding)


def _create_bm25_index(cur, document_id: str):
    cur.execute(
        "UPDATE nodes SET tsv = to_tsvector('simple', content) "
        "WHERE document_id = %s AND embedding IS NOT NULL AND tsv IS NULL",
        (document_id,),
    )


def index_document(
    root: TreeNode,
    bot_id: str,
    file_name: str,
    embed_fn: Optional[Callable[[str], list]] = None,
    document_id: Optional[str] = None,
) -> str:
    """Ghi cây `root` (trả về từ production_chunker trước khi flatten thành
    chunk dict) vào Postgres. Trả về document_id đã dùng."""
    document_id = document_id or str(uuid.uuid4())

    conn = psycopg2.connect(**PG_CONN_PARAMS)
    try:
        with conn:
            with conn.cursor() as cur:
                _insert_document(cur, document_id, bot_id, file_name)
                _write_tree(cur, root, document_id, embed_fn)
                _create_bm25_index(cur, document_id)
    finally:
        conn.close()

    return document_id


# ---------------------------------------------------------------------
# Ví dụ dùng kèm production_chunker (cần cây thô, không dùng chunk_docx()
# vì hàm đó đã "làm phẳng" cây thành list — sửa production_chunker.py 1
# dòng để trả thêm `root`, hoặc gọi lại từng bước thủ công như dưới đây).
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from .docx_parser import extract_docx_lines
    from .toc_detector import detect_heading_levels
    from .tree_builder import build_tree
    from .tree_postprocess import insert_tables_and_split, process_tree

    path = "sample.docx"
    lines, tables = extract_docx_lines(path)
    levels = detect_heading_levels(lines)
    root = build_tree(lines, levels, source_name=path)
    insert_tables_and_split(root, tables)
    process_tree(root)

    doc_id = index_document(root, bot_id="demo-bot", file_name=path, embed_fn=None)
    print(f"Đã index xong, document_id={doc_id}")
