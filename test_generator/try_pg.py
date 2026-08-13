# try_pg.py -- test đọc/ghi Postgres thật, dùng embed_fn GIẢ (không cần
# embedding model thật) chỉ để xác nhận luồng ghi -> đọc hoạt động đúng.
#
# LƯU Ý: dùng embed_fn=None (như bản trước) sẽ khiến leaf được ghi với
# embedding=NULL -> fetch_chunks_for_documents() lọc "embedding IS NOT NULL"
# sẽ không thấy leaf đó -> trả về 0 chunk. Đây không phải lỗi, mà là đúng
# thiết kế: cột embedding dùng để phân biệt node lá (có) và node cha (NULL).

import hashlib
import random

from test_generator.tree_node import TreeNode
from test_generator.pg_writer import index_document
from test_generator.pg_reader import fetch_chunks_for_documents


def fake_embed(text: str) -> list[float]:
    """Vector giả 3072 chiều (đúng số chiều schema VECTOR(3072)) -- chỉ để
    test luồng, KHÔNG mang ý nghĩa ngữ nghĩa thật. Khi có embedding model
    thật, thay hàm này bằng lệnh gọi API embedding thật."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rnd = random.Random(seed)
    return [rnd.random() for _ in range(3072)]


root = TreeNode()
root.content = "Test doc"
leaf = TreeNode()
leaf.content = "Nội dung thử nghiệm"
leaf.parent = root
root.child.append(leaf)

doc_id = index_document(root, bot_id="test-bot", file_name="test.docx", embed_fn=fake_embed)
print("Đã ghi vào DB thật, document_id =", doc_id)

chunks = fetch_chunks_for_documents([doc_id])
print(f"Đọc lại được {len(chunks)} chunk:")
for c in chunks:
    print(" -", c["text"])