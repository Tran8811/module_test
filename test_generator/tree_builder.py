"""
Thuật toán (stack theo level):
- level >= 1 (tiêu đề): tạo node mới; cha là node gần nhất đang mở ở
  level-1 (nếu chưa có thì lùi dần); đóng mọi nhánh có level >= level hiện
  tại (bắt đầu 1 nhánh con mới).
- level == -1 (nội dung thường): gộp vào "leaf hiện tại" của node đang mở
  sâu nhất; nếu leaf hiện tại + dòng mới vượt LEAF_SPLIT_MAX_TOKENS thì
  tạo leaf mới.
"""
from .chunker import _token_count
from .tree_node import TreeNode

# Giống ngưỡng bên production (build_tree_from_df): 1 leaf tối đa ~500 token
# trước khi tách sang leaf mới.
LEAF_SPLIT_MAX_TOKENS = 500


def build_tree(lines: list[str], levels: dict[int, int], source_name: str) -> TreeNode:
    root = TreeNode()
    root.content = source_name
    root.source = source_name

    # current_nodes[level] = node đang mở ở level đó (nhánh hiện hành)
    current_nodes: dict[int, TreeNode] = {0: root}
    # leaf hiện tại đang gom nội dung thường, theo từng node cha (key = id node)
    current_leaf: dict[int, TreeNode | None] = {id(root): None}
    order_counter: dict[int, int] = {}

    def _next_order(parent: TreeNode) -> int:
        key = id(parent)
        idx = order_counter.get(key, 0)
        order_counter[key] = idx + 1
        return idx

    def _new_child(parent: TreeNode) -> TreeNode:
        node = TreeNode()
        node.parent = parent
        node.source = source_name
        node.order_index = _next_order(parent)
        parent.child.append(node)
        return node

    for i, line in enumerate(lines):
        level = levels.get(i, -1)

        if level == -1:
            # nội dung thường -> gộp vào leaf hiện tại của node đang mở sâu nhất
            deepest_level = max(current_nodes.keys())
            parent = current_nodes[deepest_level]
            leaf = current_leaf.get(id(parent))

            need_new_leaf = leaf is None or _token_count(
                (leaf.content + "\n" + line) if leaf.content else line
            ) > LEAF_SPLIT_MAX_TOKENS

            if need_new_leaf:
                leaf = _new_child(parent)
                current_leaf[id(parent)] = leaf

            leaf.content = (leaf.content + "\n" + line) if leaf.content else line
            continue

        # level >= 1: là tiêu đề -> tìm cha ở level-1, lùi dần nếu chưa mở
        target_level = level
        while target_level - 1 not in current_nodes and target_level > 1:
            target_level -= 1
        parent = current_nodes.get(target_level - 1, root)

        node = _new_child(parent)
        node.content = line

        # đóng mọi nhánh có level >= target_level (bắt đầu nhánh con mới)
        for k in [k for k in current_nodes if k >= target_level]:
            del current_nodes[k]
        current_nodes[target_level] = node
        current_leaf[id(node)] = None

    return root
