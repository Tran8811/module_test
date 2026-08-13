"""
Cấu trúc cây phân cấp tài liệu. Mọi module chunking mới (docx_parser, tree_builder,
tree_postprocess, tree_to_chunks) đều thao tác trên cấu trúc này.
"""
import uuid


class TreeNode:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.content = ""          # nội dung của CHÍNH node này (tiêu đề nếu là node cha,
                                    # đoạn văn/bảng nếu là node lá)
        self.child: list["TreeNode"] = []
        self.parent: "TreeNode | None" = None
        self.order_index = 0       # thứ tự xuất hiện trong văn bản gốc (dùng để sort khi cần)
        self.source = None         # tên/đường dẫn file nguồn

    def is_leaf(self) -> bool:
        return len(self.child) == 0

    def __repr__(self):
        preview = (self.content or "")[:40].replace("\n", " ")
        return f"<TreeNode leaf={self.is_leaf()} content='{preview}...'>"
