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
                cur.execute("SELECT name FROM documents WHERE id = %s", (document_id,))
                row = cur.fetchone()
                source_name = row["name"] if row else document_id

                all_chunks.extend(_fetch_document_leaves(cur, document_id, source_name))
    finally:
        conn.close()

    return assign_chunk_ids(all_chunks)


def fetch_chunks_for_document(document_id: str) -> list[dict]:
    return fetch_chunks_for_documents([document_id])


def list_documents(bot_id: str | None = None, limit: int = 50) -> list[dict]:
    """Liệt kê document có sẵn trong DB (id, name, bot_id) -- tiện tra
    document_id thật để truyền vào fetch_chunks_for_documents(), không cần
    tự viết SQL/mở pgAdmin.

        for d in list_documents():
            print(d["id"], d["name"])
    """
    conn = psycopg2.connect(**PG_CONN_PARAMS)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if bot_id:
                cur.execute(
                    "SELECT id, name, bot_id FROM documents WHERE bot_id = %s ORDER BY name LIMIT %s",
                    (bot_id, limit),
                )
            else:
                cur.execute("SELECT id, name, bot_id FROM documents ORDER BY name LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Vector search: tìm chunk GẦN NGHĨA NHẤT với 1 câu query, thay vì lấy
# HẾT mọi leaf của 1 document như fetch_chunks_for_documents() ở trên.
# Đây mới là kiểu "vector search" thật (giống production dùng để retrieval
# khi chatbot trả lời câu hỏi người dùng).
# ---------------------------------------------------------------------

_VECTOR_TOPK_QUERY = """
SELECT
    n.id            AS leaf_id,
    n.document_id   AS document_id,
    n.embedding <=> %(query_vector)s::vector AS distance
FROM nodes n
WHERE n.embedding IS NOT NULL
{bot_filter}
ORDER BY n.embedding <=> %(query_vector)s::vector
LIMIT %(top_k)s;
"""

_ANCESTOR_QUERY_FOR_LEAF = """
WITH RECURSIVE ancestors AS (
    SELECT n.id AS node_id, n.content AS content, 0 AS depth
    FROM nodes n
    WHERE n.id = %(leaf_id)s

    UNION ALL

    SELECT p.id, p.content, a.depth + 1
    FROM ancestors a
    JOIN node_links l ON l.child_id = a.node_id
    JOIN nodes p       ON p.id = l.parent_id
)
SELECT array_agg(content ORDER BY depth DESC) AS path_top_to_bottom
FROM ancestors;
"""


def _vector_literal(vec: list[float]) -> str:
    """pgvector nhận input dạng chuỗi '[0.1,0.2,...]', cast bằng ::vector
    trong SQL. Không cần cài thêm package pgvector-python, chỉ cần format
    đúng chuỗi này."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def search_chunks_by_vector(
    query_text: str,
    embed_fn,
    top_k: int = 5,
    bot_id: str | None = None,
) -> list[dict]:
    """Embed `query_text` bằng `embed_fn`, tìm top_k node lá gần nghĩa nhất
    trong TOÀN BỘ DB (hoặc lọc theo bot_id nếu bảng documents có cột đó),
    ghép breadcrumb cho từng kết quả, trả về list chunk dict.

    Khác với fetch_chunks_for_documents(): hàm đó lấy HẾT chunk của 1/nhiều
    document cụ thể (biết trước document_id). Hàm này mô phỏng đúng bước
    "retrieval" thật — không biết trước document nào, tìm bằng ngữ nghĩa.

        chunks = search_chunks_by_vector("quy trình xin nghỉ phép", embed_fn, top_k=5)
    """
    query_vector = embed_fn(query_text)
    vector_literal = _vector_literal(query_vector)

    bot_filter = ""
    params = {"query_vector": vector_literal, "top_k": top_k}
    if bot_id is not None:
        bot_filter = "AND n.document_id IN (SELECT id FROM documents WHERE bot_id = %(bot_id)s)"
        params["bot_id"] = bot_id

    conn = psycopg2.connect(**PG_CONN_PARAMS)
    chunks: list[dict] = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_VECTOR_TOPK_QUERY.format(bot_filter=bot_filter), params)
            hits = cur.fetchall()

            for hit in hits:
                cur.execute(_ANCESTOR_QUERY_FOR_LEAF, {"leaf_id": hit["leaf_id"]})
                row = cur.fetchone()
                path = row["path_top_to_bottom"] or []
                if not path:
                    continue

                leaf_content = path[-1]
                breadcrumb = " ".join(t.strip() for t in path[:-1] if t and t.strip())
                full_text = f"[{breadcrumb}]\n{leaf_content}" if breadcrumb else leaf_content

                chunks.append(
                    {
                        "chunk_id": None,
                        "text": full_text,
                        "metadata": {
                            "breadcrumb": breadcrumb,
                            "node_id": hit["leaf_id"],
                            "document_id": hit["document_id"],
                            "distance": float(hit["distance"]),  # càng nhỏ càng gần nghĩa (cosine distance)
                        },
                    }
                )
    finally:
        conn.close()

    return assign_chunk_ids(chunks)
