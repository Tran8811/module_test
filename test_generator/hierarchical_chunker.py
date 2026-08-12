import re
from typing import Dict, List, Optional, Tuple

from .chunker import (
    DEFAULT_SEPARATORS,
    _normalize_table_row_boundaries,
    _tla_split_text,
)
from .utils import assign_chunk_ids

# ==========================================================
# BỘ REGEX NHẬN DIỆN TIÊU ĐỀ TỔNG QUÁT (GENERIC HEADING PATTERNS)
# Được sắp xếp theo thứ tự cấp độ từ cao đến thấp (Cấp 1 -> Cấp N)
# ==========================================================

GENERIC_HEADING_PATTERNS = [
    # Cấp 1: Markdown H1 (#) hoặc Phần/Chương/Mục La Mã (I., II.) hoặc Dòng IN HOA độc lập
    re.compile(
        r"^\s*(?:#\s+|(?:PHẦN|CHƯƠNG|PART|SECTION)\s+[IVXLCDM\d]+[\.\:]?\s+.*|[IVXLCDM]+\.\s+[A-ZÀ-Ỵ].*|^[A-ZÀ-Ỵ0-9\s,\.\-]{8,}$)",
        re.MULTILINE
    ),
    # Cấp 2: Markdown H2 (##) hoặc Mục số cấp 1 (1., 2.) hoặc Chữ cái hoa (A., B.)
    re.compile(
        r"^\s*(?:##\s+|\d+\.\s+[A-ZÀ-Ỵa-zà-ỵ].*|[A-Z]\.\s+[A-ZÀ-Ỵa-zà-ỵ].*)",
        re.MULTILINE
    ),
    # Cấp 3: Markdown H3 (###) hoặc Mục số cấp 2 (1.1, 1.2)
    re.compile(
        r"^\s*(?:###\s+|\d+\.\d+\.?\s+.*)",
        re.MULTILINE
    ),
    # Cấp 4: Markdown H4 (####) hoặc Mục số cấp 3 (1.1.1) hoặc Đầu dòng dạng a), b)
    re.compile(
        r"^\s*(?:####\s+|\d+\.\d+\.\d+\.?\s+.*|[a-zđ]\)\s+.*)",
        re.MULTILINE
    ),
]


def _find_headings_at_level(text: str, pattern: re.Pattern) -> List[Tuple[int, int, str]]:
    """Tìm tất cả tiêu đề khớp với 1 regex cụ thể trong văn bản."""
    matches = []
    for m in pattern.finditer(text):
        raw = m.group()
        title = raw.strip()
        # Loại bỏ các ký tự Markdown # ở đầu tiêu đề nếu có
        title = re.sub(r"^#+\s*", "", title)
        if not title:
            continue
        leading_ws = len(raw) - len(raw.lstrip())
        trailing_ws = len(raw) - len(raw.rstrip())
        real_start = m.start() + leading_ws
        real_end = m.end() - trailing_ws
        matches.append((real_start, real_end, title))

    matches.sort(key=lambda x: x[0])

    # Khử trùng lặp các match nằm quá gần nhau
    deduped = []
    for start, end, title in matches:
        if deduped and start - deduped[-1][0] < 3:
            continue
        deduped.append((start, end, title))
    return deduped


def _split_text_by_pattern(text: str, pattern: re.Pattern) -> List[Dict]:
    """Chia nhỏ đoạn văn bản theo một mẫu Regex tiêu đề nhất định."""
    headings = _find_headings_at_level(text, pattern)

    if not headings:
        return [{"title": None, "body": text}]

    blocks = []
    # Đoạn văn bản trước tiêu đề đầu tiên (nếu có)
    if headings[0][0] > 0:
        pre = text[: headings[0][0]].strip()
        if pre:
            blocks.append({"title": None, "body": pre})

    # Cắt từng block theo vị trí các tiêu đề
    for i, (start, end, title) in enumerate(headings):
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        body = text[end:next_start].strip()
        blocks.append({"title": title, "body": body})

    return blocks


def _recursive_tree_split(
    text: str,
    patterns: List[re.Pattern],
    current_level_idx: int = 0,
    parent_titles: Optional[List[str]] = None,
) -> List[Dict]:
    """Chia văn bản đệ quy theo cây đa cấp dựa trên danh sách các Pattern tiêu đề."""
    if parent_titles is None:
        parent_titles = []

    # Nếu đã đi hết các cấp Pattern tiêu đề -> Trả về khối lá (Leaf Block)
    if current_level_idx >= len(patterns) or not text.strip():
        return [{"titles": parent_titles, "body": text}]

    current_pattern = patterns[current_level_idx]
    blocks = _split_text_by_pattern(text, current_pattern)

    # Nếu pattern hiện tại không tìm thấy tiêu đề nào, thử xuống cấp tiêu đề tiếp theo
    if len(blocks) == 1 and blocks[0]["title"] is None:
        return _recursive_tree_split(
            text, patterns, current_level_idx + 1, parent_titles
        )

    results = []
    for block in blocks:
        title = block["title"]
        body = block["body"]

        # Cập nhật danh sách tiêu đề cha (Path/Breadcrumb)
        new_titles = list(parent_titles)
        if title:
            new_titles.append(title)

        if not body.strip():
            continue

        # Đệ quy xuống các cấp tiêu đề tiếp theo cho phần body
        child_results = _recursive_tree_split(
            body, patterns, current_level_idx + 1, new_titles
        )
        results.extend(child_results)

    return results


def split_documents_hierarchical(
    documents,
    heading_patterns: Optional[List[re.Pattern]] = None
) -> List[Dict]:
    """Chunking phân cấp dạng Cây (N-level Dynamic Tree Chunking).

    Args:
        documents: Danh sách tài liệu đầu vào (mỗi doc có page_content, metadata).
        heading_patterns: Danh sách Regex phân cấp tiêu đề (nếu None sẽ dùng bộ mặc định).

    Returns:
        Danh sách các leaf chunks chứa text đã đính kèm breadcrumb và metadata phân cấp.
    """
    patterns = heading_patterns or GENERIC_HEADING_PATTERNS
    all_chunks = []

    for doc in documents:
        content = doc.page_content or ""
        base_meta = {**(getattr(doc, "metadata", {}) or {})}
        source = base_meta.get("source") or base_meta.get("file_name") or "unknown"
        base_meta["source"] = source

        # 1. Phân rã văn bản thành các khối lá trên Cây tiêu đề
        tree_leaf_blocks = _recursive_tree_split(content, patterns)

        # 2. Xử lý từng khối lá (Chia theo Token/Size tối đa)
        for block in tree_leaf_blocks:
            titles = block["titles"]
            body_text = block["body"]

            if not body_text.strip():
                continue

            # Chuẩn hóa bảng & Chia nhỏ văn bản cấp lá (Token-aware splitting)
            normalized_body = _normalize_table_row_boundaries(body_text)
            leaf_texts = _tla_split_text(normalized_body, DEFAULT_SEPARATORS)

            # Xây dựng đường dẫn Breadcrumb (VD: "Chương 1 > Mục 1.1 > Khái niệm")
            breadcrumb = " > ".join(titles) if titles else ""

            for text in leaf_texts:
                if not text.strip():
                    continue

                # Tạo metadata linh hoạt lưu danh sách tiêu đề các cấp
                meta = {
                    **base_meta,
                    "headings": titles,
                    "breadcrumb": breadcrumb,
                }
                
                # Lưu thêm các cấp riêng lẻ nếu cần tương thích ngược
                for i, t in enumerate(titles):
                    meta[f"level_{i+1}"] = t

                full_text = f"[{breadcrumb}]\n{text}" if breadcrumb else text

                all_chunks.append({
                    "chunk_id": None,
                    "text": full_text,
                    "metadata": meta,
                })

    return assign_chunk_ids(all_chunks)