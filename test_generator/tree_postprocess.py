# tree_postprocess.py
"""
Tương đương traverse_and_replace() + process_tree() bên production:
  1. insert_tables_and_split(): thay placeholder <tab>id</tab> bằng HTML
     bảng thật ở các node lá; nếu node lá sau đó quá dài thì cắt nhỏ bằng
     bộ splitter token-aware sẵn có trong chunker.py.
  2. process_tree(): gộp cây để tránh leaf quá nhỏ / header trơ trọi:
       - merge_with_parent: gộp cha-con nếu tổng nhỏ và không có cháu
       - merge_sibling_nodes: gộp các leaf anh em liền kề tới ngưỡng
       - merge_with_header: gộp header ngắn/rỗng vào con đầu tiên
       - format_tree_structure: đảm bảo mọi node cha có ít nhất 1 con lá thật
"""
import re

from .chunker import DEFAULT_SEPARATORS, _normalize_table_row_boundaries, _tla_split_text, _token_count
from .tree_node import TreeNode

# Ngưỡng giống production (đơn vị: token xấp xỉ theo word-count, nhất quán
# với cách đếm token đang dùng trong toàn bộ project).
MAX_LEAF_TOKENS_BEFORE_SPLIT = 4000     # leaf > ngưỡng này thì phải cắt
MERGE_PARENT_MAX_TOKENS = 500           # gộp cha-con nếu tổng <= ngưỡng
MERGE_SIBLING_MAX_TOKENS = 1000         # gộp 2 leaf anh em nếu tổng < ngưỡng
MERGE_HEADER_MAX_TOKENS = 200           # gộp header rỗng/ngắn vào con đầu


# ---------------------------------------------------------------------
# Bước 1: chèn bảng + cắt leaf quá dài
# ---------------------------------------------------------------------

def _replace_table_placeholders(text: str, tables: dict[str, str]) -> str:
    def _sub(m: re.Match) -> str:
        return tables.get(m.group(1), m.group(0))

    return re.sub(r"<tab>(\w+)</tab>", _sub, text)


def insert_tables_and_split(node: TreeNode, tables: dict[str, str]) -> None:
    if node.child:
        for c in list(node.child):
            insert_tables_and_split(c, tables)
        return

    node.content = _replace_table_placeholders(node.content, tables)

    if _token_count(node.content) <= MAX_LEAF_TOKENS_BEFORE_SPLIT:
        return

    # Cắt bằng bộ splitter token-aware đã có sẵn trong project (tôn trọng
    # ranh giới bảng/dòng nhờ _normalize_table_row_boundaries).
    normalized = _normalize_table_row_boundaries(node.content)
    pieces = _tla_split_text(normalized, DEFAULT_SEPARATORS)

    if node.parent is None or len(pieces) <= 1:
        node.content = pieces[0] if pieces else node.content
        return

    parent = node.parent
    idx_in_parent = parent.child.index(node)
    new_nodes = []
    for i, piece in enumerate(pieces):
        n = TreeNode()
        n.parent = parent
        n.content = piece
        n.order_index = node.order_index + i
        n.source = node.source
        new_nodes.append(n)
    parent.child[idx_in_parent : idx_in_parent + 1] = new_nodes


# ---------------------------------------------------------------------
# Bước 2: tối ưu hoá cấu trúc cây
# ---------------------------------------------------------------------

def _merge_with_parent(node: TreeNode) -> bool:
    changed = False
    for c in list(node.child):
        changed = _merge_with_parent(c) or changed

    if node.parent is None or not node.child:
        return changed

    total = _token_count(node.content) + sum(_token_count(c.content) for c in node.child)
    no_grandchildren = all(not c.child for c in node.child)

    if total <= MERGE_PARENT_MAX_TOKENS and no_grandchildren:
        for c in node.child:
            node.content = (node.content + "\n" + c.content) if node.content else c.content
        node.child = []
        changed = True

    return changed


def _merge_sibling_nodes(node: TreeNode) -> bool:
    changed = False
    for c in node.child:
        changed = _merge_sibling_nodes(c) or changed

    new_children: list[TreeNode] = []
    buffer_leaf: TreeNode | None = None

    for c in node.child:
        is_leaf = not c.child
        if is_leaf and buffer_leaf is not None:
            merged_tokens = _token_count(buffer_leaf.content) + _token_count(c.content)
            if merged_tokens < MERGE_SIBLING_MAX_TOKENS:
                buffer_leaf.content = (
                    (buffer_leaf.content + "\n" + c.content) if buffer_leaf.content else c.content
                )
                changed = True
                continue
        new_children.append(c)
        buffer_leaf = c if is_leaf else None

    node.child = new_children
    return changed


def _merge_with_header(node: TreeNode) -> None:
    for c in list(node.child):
        _merge_with_header(c)

    if (
        node.parent is not None
        and _token_count(node.content) <= MERGE_HEADER_MAX_TOKENS
        and len(node.child) == 1
        and node.child[0].child  # con duy nhất lại có cháu -> đủ điều kiện gộp
    ):
        only_child = node.child[0]
        only_child.content = (node.content + "\n" + only_child.content) if node.content else only_child.content
        idx = node.parent.child.index(node)
        only_child.parent = node.parent
        node.parent.child[idx] = only_child


def _format_tree_structure(node: TreeNode) -> None:
    if node.child:
        if not any(not c.child for c in node.child):
            placeholder = TreeNode()
            placeholder.parent = node
            placeholder.source = node.source
            placeholder.content = ""
            node.child.insert(0, placeholder)
        for c in node.child:
            _format_tree_structure(c)


def process_tree(root: TreeNode) -> None:
    changed = True
    while changed:
        a = _merge_with_parent(root)
        b = _merge_sibling_nodes(root)
        changed = a or b

    _merge_with_header(root)
    _format_tree_structure(root)
