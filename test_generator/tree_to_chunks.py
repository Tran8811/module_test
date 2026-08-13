from .tree_node import TreeNode
from .utils import assign_chunk_ids


def get_list_leaf(node: TreeNode) -> list[TreeNode]:
    """DFS theo đúng thứ tự node.child (= thứ tự đọc gốc trong văn bản)."""
    if node.child:
        leaves: list[TreeNode] = []
        for c in node.child:
            leaves.extend(get_list_leaf(c))
        return leaves
    return [node] if node.content.strip() else []


def get_ancestor_breadcrumb(node: TreeNode) -> str:
    """Nối tiêu đề mọi node tổ tiên (không tính chính node này) thành
    breadcrumb"""
    titles = []
    cur = node.parent
    while cur is not None:
        if cur.content.strip():
            titles.append(cur.content.strip())
        cur = cur.parent
    titles.reverse()
    return " > ".join(titles)


def tree_to_chunks(root: TreeNode) -> list[dict]:
    leaves = get_list_leaf(root)
    chunks = []

    for leaf in leaves:
        breadcrumb = get_ancestor_breadcrumb(leaf)
        full_text = f"[{breadcrumb}]\n{leaf.content}" if breadcrumb else leaf.content

        chunks.append(
            {
                "chunk_id": None,
                "text": full_text,
                "metadata": {
                    "source": root.source,
                    "breadcrumb": breadcrumb,
                    "node_id": leaf.id,   # giữ lại để đối chiếu với node.id thật trong Postgres sau này
                },
            }
        )

    # chunk_id = index tuần tự trong list, ĐÚNG thứ tự đọc (DFS) — bắt buộc
    # để _expand_neighbors() trong candidate_retriever.py hoạt động đúng.
    return assign_chunk_ids(chunks)
