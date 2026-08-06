import re
from typing import Dict, List

from .chunker import (
    DEFAULT_SEPARATORS,
    _normalize_table_row_boundaries,
    _tla_split_text,
)
from .utils import assign_chunk_ids

# ==========================================================
# Regex nhận diện heading theo 2 cấp.
# Cấp 1 (danh mục): tiêu đề lớn - vd "PHẦN I", "CHƯƠNG 2", dòng IN HOA dài,
#                    hoặc "BẢNG ĐIỂM HỌC PHẦN: ...".
# Cấp 2 (đề mục)  : tiêu đề phụ trong 1 danh mục - vd "Lớp: ...",
#                    "Môn: ...", "1.1 ...", "a) ...".
#
# CHỈNH LẠI CHO KHỚP VĂN BẢN THẬT trước khi dùng.
# ==========================================================

DANH_MUC_PATTERNS = [
    re.compile(r"^\s*(PHẦN|CHƯƠNG)\s+[IVXLCDM\d]+[\.\:]?\s*.*$", re.MULTILINE),
    re.compile(r"^\s*[IVXLCDM]+\.\s+[A-ZÀ-Ỵ].*$", re.MULTILINE),          # I. TIÊU ĐỀ
    re.compile(r"^\s*BẢNG\s+ĐIỂM.*$", re.MULTILINE | re.IGNORECASE),      # BẢNG ĐIỂM HỌC PHẦN ...
    re.compile(r"^\s*[A-ZÀ-Ỵ0-9\s,\.\-]{10,}$", re.MULTILINE),            # dòng IN HOA dài (tiêu đề)
]

DE_MUC_PATTERNS = [
    re.compile(r"^\s*(Lớp|Môn|Học phần|Kỳ thi|Bảng)\s*[:\-]\s*.+$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\d+\.\d+\.?\s+.+$", re.MULTILINE),                    # 1.1  Tiêu đề phụ
    re.compile(r"^\s*[a-zđ]\)\s+.+$", re.MULTILINE),                       # a) Tiêu đề phụ
]


def _find_headings(text: str, patterns) -> List[tuple]:
    matches = []
    for pattern in patterns:
        for m in pattern.finditer(text):
            raw = m.group()
            title = raw.strip()
            if not title:
                continue
            leading_ws = len(raw) - len(raw.lstrip())
            trailing_ws = len(raw) - len(raw.rstrip())
            real_start = m.start() + leading_ws
            real_end = m.end() - trailing_ws
            matches.append((real_start, real_end, title))

    matches.sort(key=lambda x: x[0])

    deduped = []
    for start, end, title in matches:
        # 2 pattern khác nhau khớp cùng 1 dòng (offset rất gần) -> chỉ giữ 1
        if deduped and start - deduped[-1][0] < 3:
            continue
        deduped.append((start, end, title))
    return deduped


def _split_by_headings(text: str, patterns) -> List[Dict]:
    """Chia text thành block theo heading cấp hiện tại.

    Trả list dict {"title": str | None, "body": str}. Nếu không tìm thấy
    heading nào, trả về đúng 1 block với title=None (toàn bộ text là body).
    """
    headings = _find_headings(text, patterns)

    if not headings:
        return [{"title": None, "body": text}]

    blocks = []

    if headings[0][0] > 0:
        pre = text[: headings[0][0]].strip()
        if pre:
            blocks.append({"title": None, "body": pre})

    for i, (start, end, title) in enumerate(headings):
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        body = text[end:next_start].strip()  # dùng thẳng end của match, không suy từ len(title)
        blocks.append({"title": title, "body": body})

    return blocks


def split_documents_hierarchical(documents) -> List[Dict]:
    """Chunking phân cấp 3 tầng: DANH MỤC -> ĐỀ MỤC -> leaf chunk (token-aware).

    Mỗi leaf chunk trả về có:
      - chunk["text"]              : "[breadcrumb]\\n<nội dung>" (đã chèn ngữ cảnh)
      - chunk["metadata"]["danh_muc"], ["de_muc"], ["breadcrumb"] : để lọc/debug
    """
    all_chunks = []

    for doc in documents:
        content = doc.page_content or ""
        base_meta = {**(getattr(doc, "metadata", {}) or {})}
        source = base_meta.get("source") or base_meta.get("file_name") or "unknown"
        base_meta["source"] = source

        danh_muc_blocks = _split_by_headings(content, DANH_MUC_PATTERNS)

        for dm_block in danh_muc_blocks:
            danh_muc_title = dm_block["title"]
            de_muc_blocks = _split_by_headings(dm_block["body"], DE_MUC_PATTERNS)

            for de_block in de_muc_blocks:
                de_muc_title = de_block["title"]

                leaf_text = _normalize_table_row_boundaries(de_block["body"])
                leaf_texts = _tla_split_text(leaf_text, DEFAULT_SEPARATORS)

                breadcrumb_parts = [t for t in [danh_muc_title, de_muc_title] if t]
                breadcrumb = " > ".join(breadcrumb_parts)

                for text in leaf_texts:
                    if not text.strip():
                        continue

                    meta = {
                        **base_meta,
                        "danh_muc": danh_muc_title,
                        "de_muc": de_muc_title,
                        "breadcrumb": breadcrumb,
                    }

                    full_text = f"[{breadcrumb}]\n{text}" if breadcrumb else text

                    all_chunks.append({
                        "chunk_id": None,
                        "text": full_text,
                        "metadata": meta,
                    })

    return assign_chunk_ids(all_chunks)
