"""
"""

TOC_EXTRACTION_PROMPT = """Bạn đang phân tích cấu trúc phân cấp (mục lục) của một văn bản.

Cho danh sách các dòng văn bản dưới đây (mỗi dòng có id ở đầu), hãy xác định:
- Dòng nào là TIÊU ĐỀ (chương, mục, phần, điều khoản...) và cấp độ (level) của nó.
  level = 1 là tiêu đề cấp cao nhất, level tăng dần theo độ sâu (2, 3, 4...).
- Dòng KHÔNG phải tiêu đề (nội dung thường, đoạn văn, bảng) thì level = -1.

Các tiêu đề đã xác định ở đoạn trước (để giữ mạch phân cấp xuyên suốt toàn văn bản):
{previous_headings}

Danh sách dòng cần phân loại:
{lines}

Chỉ trả về JSON, KHÔNG giải thích, KHÔNG dùng markdown, KHÔNG bọc ```json.

Định dạng:
{{
  "items": [
    {{"id": 0, "level": -1}},
    {{"id": 1, "level": 1}},
    {{"id": 2, "level": 2}}
  ]
}}
"""
