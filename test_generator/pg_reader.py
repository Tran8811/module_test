# pg_reader.py
"""
FLOW 2: lấy chunk (leaf node) đã index sẵn trong Postgres/pgvector, kèm
breadcrumb tổ tiên (leo node_links), convert sang đúng format
{chunk_id, text, metadata} như production_chunker.tree_to_chunks() —
để gen-test pipeline dùng chung API bất kể nguồn chunk là DOCX mới hay
dữ liệu đã index sẵn.

    from pg_reader import fetch_chunks_for_documents
    chunks = fetch_chunks_for_documents(["doc-id-1", "doc-id-2"])
"""
import psycopg2
import psycopg2.extras

from .pg_config import PG_CONN_PARAMS
from .utils import assign_chunk_ids

_RECURSIVE_LEAF_QUERY = """
WITH RECURSIVE ancestors AS (
    SELECT
        n.id        AS leaf_id,
        n.id        AS node_id,
        n.content   AS content,
        0           AS depth
    FROM nodes n
    WHERE n.document_id = %(document_id)s
      AND n.embedding IS NOT NULL

    UNION ALL

    SELECT
        a.leaf_id,
        p.id        AS node_id,
        p.content   AS content,
        a.depth + 1
    FROM ancestors a
    JOIN node_links l ON l.child_id = a.node_id
    JOIN nodes p       ON p.id = l.parent_id
)
SELECT
    a.leaf_id,
    array_agg(a.content ORDER BY a.depth DESC) AS path_top_to_bottom,
    (
        SELECT order_index FROM node_links
        WHERE child_id = a.leaf_id
        LIMIT 1
    ) AS order_index
FROM ancestors a
GROUP BY a.leaf_id
ORDER BY order_index;
"""


def _fetch_document_leaves(cur, document_id: str, source_name: str) -> list[dict]:
    cur.execute(_RECURSIVE_LEAF_QUERY, {"document_id": document_id})
    rows = cur.fetchall()

    chunks = []
    for row in rows:
        path = row["path_top_to_bottom"]
        leaf_content = path[-1]
        breadcrumb = " ".join(t.strip() for t in path[:-1] if t and t.strip())

        full_text = f"[{breadcrumb}]\n{leaf_content}" if breadcrumb else leaf_content
        chunks.append(
            {
                "chunk_id": None,
                "text": full_text,
                "metadata": {
                    "source": source_name,
                    "breadcrumb": breadcrumb,
                    "node_id": row["leaf_id"],
                    "document_id": document_id,
                },
            }
        )
    return chunks


def fetch_chunks_for_documents(document_ids: list[str]) -> list[dict]:
    """Lấy chunk cho NHIỀU document cùng lúc, nối theo đúng thứ tự
    document_ids -> gọi assign_chunk_ids MỘT LẦN DUY NHẤT ở cuối.

    Quan trọng: candidate_retriever._expand_neighbors() dùng chunk_id LÀM
    INDEX trực tiếp vào list -> nếu gọi assign_chunk_ids riêng cho từng
    document rồi mới nối list lại, chunk_id sẽ bị trùng giữa các document.
    """
    conn = psycopg2.connect(**PG_CONN_PARAMS)
    all_chunks: list[dict] = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for document_id in document_ids:
                cur.execute("SELECT file_name FROM documents WHERE id = %s", (document_id,))
                row = cur.fetchone()
                source_name = row["file_name"] if row else document_id

                all_chunks.extend(_fetch_document_leaves(cur, document_id, source_name))
    finally:
        conn.close()

    return assign_chunk_ids(all_chunks)


def fetch_chunks_for_document(document_id: str) -> list[dict]:
    return fetch_chunks_for_documents([document_id])
