from typing import Iterable
import re
import uuid
from .llm import _estimate_tokens
# Số token overlap giữa các chunk liền kề
CHUNK_OVERLAP_TOKENS = 50

# Giới hạn số token tối đa cho mỗi chunk
CHUNK_MAX_TOKENS = 500

# Các dấu phân cách theo thứ tự ưu tiên khi chia nhỏ văn bản (TLA split)
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_count(text: str) -> int:
    """Ước lượng số token đơn giản bằng cách đếm số từ (whitespace-split)."""
    return _estimate_tokens(text)


def _normalize_table_row_boundaries(text: str) -> str:
    """Chèn ký tự xuống dòng ngay trước mỗi vị trí được phát hiện là
    bắt đầu một dòng dữ liệu dạng "<STT>. <MSSV 8 chữ số> <Tên...>",
    để \\n\\n / \\n có thể chia đúng ranh giới giữa các dòng.
    """
    pattern = re.compile(r"(?<!^)(?<!\n)(\d+\.\s+\d{8}\s+)")
    return pattern.sub(r"\n\1", text)


def _tla_split_text(
    text: str,
    separators: list[str],
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> list[str]:
    """Chia văn bản đệ quy theo danh sách separators (Text-Level-Aware),
    đảm bảo mỗi phần không vượt quá max_tokens token.
    """
    if not text.strip():
        return []
    if _token_count(text) <= max_tokens or not separators:
        return [text]

    sep, rest_separators = separators[0], separators[1:]

    if sep == "":
        # Hết dấu phân cách: cắt cứng theo số từ.
        words = text.split()
        return [
            " ".join(words[i:i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ]

    parts = text.split(sep)
    results: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{sep}{part}" if buffer else part
        if _token_count(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                results.extend(_tla_split_text(buffer, rest_separators, max_tokens))
            buffer = part
    if buffer:
        results.extend(_tla_split_text(buffer, rest_separators, max_tokens))

    return [r for r in results if r.strip()]


def assign_chunk_ids(chunks: list[dict]) -> list[dict]:
    """Gán chunk_id duy nhất (UUID4 dạng chuỗi) cho từng chunk."""
    for chunk in chunks:
        chunk["chunk_id"] = str(uuid.uuid4())
    return chunks

def _extract_media_blocks(text: str) -> Iterable[tuple[str, bool]]:
    # Mẫu này khớp với các khối <table ...>...</table> hoặc <img .../>
    # và không phân biệt chữ hoa/chữ thường.
    table_re = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
    img_re = re.compile(r"(<img[^>]*>)", re.IGNORECASE)

    idx = 0
    # Đầu tiên tách các bảng, sau đó tách các thẻ ảnh bên trong phần không phải bảng.
    for m in table_re.finditer(text):
        if m.start() > idx:
            pre = text[idx : m.start()]
            # Tách riêng các thẻ ảnh khỏi phần văn bản trước bảng.
            last = 0
            for im in img_re.finditer(pre):
                if im.start() > last:
                    yield pre[last:im.start()], False
                yield im.group(1), True
                last = im.end()
            if last < len(pre):
                yield pre[last:], False
        yield m.group(1), True
        idx = m.end()

    if idx < len(text):
        tail = text[idx:]
        last = 0
        for im in img_re.finditer(tail):
            if im.start() > last:
                yield tail[last:im.start()], False
            yield im.group(1), True
            last = im.end()
        if last < len(tail):
            yield tail[last:], False


def _apply_overlap(chunks: list[dict]) -> None:
    # Nếu không có chunk hoặc số token overlap không lớn hơn 0 thì không cần xử lý.
    if not chunks or CHUNK_OVERLAP_TOKENS <= 0:
        return

    # Thêm phần cuối của chunk trước vào đầu chunk hiện tại
    # để tạo vùng chồng lấn giữa các chunk.
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]["text"]
        cur = chunks[i]["text"]
        prev_tokens = prev.split()
        overlap_n = min(CHUNK_OVERLAP_TOKENS, len(prev_tokens))
        if overlap_n > 0:
            tail = " ".join(prev_tokens[-overlap_n:])
            if not cur.startswith(tail):
                chunks[i]["text"] = tail + "\n" + cur


def split_documents(documents):
    chunks = []
    for doc in documents:
        # Lấy nội dung văn bản từ LangChain Document.
        content = doc.page_content or ""

        # Sao chép metadata của document để giữ lại các thông tin nguồn.
        base_meta = {**(getattr(doc, "metadata", {}) or {})}

        # Xác định đường dẫn/tên file nguồn từ các key metadata phổ biến.
        source = (
            base_meta.get("source")
            or base_meta.get("file_name")
            or base_meta.get("path")
        )

        if not source and hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            # Thử các key metadata thường được LangChain loader sử dụng.
            source = doc.metadata.get("source") or doc.metadata.get("file_name")

        if source:
            base_meta["source"] = source
        else:
            base_meta.setdefault("source", "unknown")

        # Giữ lại thông tin số trang nếu metadata có trường "page".
        if "page" in base_meta:
            base_meta["page"] = base_meta.get("page")

        for segment, is_media in _extract_media_blocks(content):
            if is_media:
                # Giữ nguyên toàn bộ khối media thành một chunk duy nhất.
                chunks.append({
                    "chunk_id": None,
                    "text": segment,
                    "metadata": {**base_meta}
                })
            else:
                # Chuẩn hóa ranh giới giữa các dòng của bảng trước khi chia chunk.
                segment = _normalize_table_row_boundaries(segment)

                # Chia văn bản theo cơ chế TLA có tính đến giới hạn token.
                texts = _tla_split_text(segment, DEFAULT_SEPARATORS)

                for text in texts:
                    chunks.append({
                        "chunk_id": None,
                        "text": text,
                        "metadata": {**base_meta}
                    })

    # Gán ID duy nhất cho từng chunk.
    chunks = assign_chunk_ids(chunks)

    for chunk in chunks:
        # Đảm bảo metadata của chunk có chứa chunk_id
        # và vẫn giữ lại source/file_name/page.
        chunk_meta = chunk.get("metadata", {}) or {}
        chunk_meta["chunk_id"] = chunk["chunk_id"]

        # Chuẩn hóa file_name từ đường dẫn source nếu có.
        if (
            "file_name" not in chunk_meta
            and chunk_meta.get("source")
            and "/" in chunk_meta.get("source")
        ):
            chunk_meta["file_name"] = chunk_meta["source"].split("/")[-1]

        chunk["metadata"] = chunk_meta

    # Thêm phần overlap giữa các chunk sau khi đã chia xong.
    _apply_overlap(chunks)

    return chunks


def split_text_items(text_items):
    """Chia danh sách các đoạn văn bản thành các chunk theo cơ chế TLA
    có tính đến giới hạn token.

    Nhận vào danh sách dict có dạng:
    {"text": str, "metadata": {...}}

    và trả về danh sách chunk có cấu trúc tương tự `split_documents`.
    """
    chunks = []

    for item in text_items:
        # Lấy nội dung text từ item.
        text = item.get("text", "") or ""

        # Lấy metadata, nếu không có thì sử dụng dictionary rỗng.
        metadata = item.get("metadata", {}) or {}

        # Sao chép metadata để tránh thay đổi dữ liệu gốc.
        base_meta = {**metadata}

        # Xác định nguồn từ các key metadata phổ biến.
        source = (
            base_meta.get("source")
            or base_meta.get("file_name")
            or base_meta.get("path")
        )

        if source:
            base_meta["source"] = source
        else:
            # Nếu không xác định được nguồn thì gán giá trị mặc định.
            base_meta.setdefault("source", "unknown")

        for segment, is_media in _extract_media_blocks(text):
            if is_media:
                # Giữ nguyên toàn bộ khối media thành một chunk duy nhất.
                chunks.append({
                    "chunk_id": None,
                    "text": segment,
                    "metadata": {**base_meta}
                })
            else:
                # Chuẩn hóa ranh giới giữa các dòng của bảng
                # trước khi thực hiện chia chunk.
                segment = _normalize_table_row_boundaries(segment)

                # Chia văn bản theo cơ chế TLA có tính đến giới hạn token.
                texts = _tla_split_text(segment, DEFAULT_SEPARATORS)

                for t in texts:
                    chunks.append({
                        "chunk_id": None,
                        "text": t,
                        "metadata": {**base_meta}
                    })

    # Gán ID duy nhất cho từng chunk.
    chunks = assign_chunk_ids(chunks)

    for chunk in chunks:
        # Đảm bảo metadata chứa chunk_id.
        chunk_meta = chunk.get("metadata", {}) or {}
        chunk_meta["chunk_id"] = chunk["chunk_id"]

        # Chuẩn hóa file_name từ đường dẫn source nếu có.
        if (
            "file_name" not in chunk_meta
            and chunk_meta.get("source")
            and "/" in chunk_meta.get("source")
        ):
            chunk_meta["file_name"] = chunk_meta["source"].split("/")[-1]

        chunk["metadata"] = chunk_meta

    # Thêm phần overlap giữa các chunk.
    _apply_overlap(chunks)

    return chunks