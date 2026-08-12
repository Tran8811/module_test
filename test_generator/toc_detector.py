# toc_detector.py
"""
Thay cho GENERIC_HEADING_PATTERNS (regex) trong hierarchical_chunker.py cũ:
dùng LLM đọc và suy luận cấp tiêu đề, giống hệt cơ chế docx2tree() /
PROMPT_EXTRACT_TOC của hệ thống indexing production.

Vì sao đổi từ regex sang LLM:
- Regex chỉ bắt được các mẫu định dạng đã biết trước (số La Mã, markdown #,
  IN HOA...). Văn bản thực tế rất đa dạng cách trình bày tiêu đề.
- LLM đọc ngữ nghĩa nên nhận diện được cả tiêu đề không theo mẫu cố định,
  đúng với cách production đang làm -> chunk boundary khớp với dữ liệu
  thật trong Postgres.
"""
import json

from .chunker import _token_count
from .llm import chat
from .prompts_production import TOC_EXTRACTION_PROMPT

# Giống ngưỡng "context_length_detect_structre" bên production: gom nhiều
# dòng thành 1 batch gửi LLM, tránh vượt context window.
TOC_BATCH_MAX_TOKENS = 4000
# Số tiêu đề gần nhất truyền làm ngữ cảnh cho batch tiếp theo, để LLM giữ
# đúng mạch phân cấp xuyên suốt nhiều batch (giống production dùng toc[-5:]).
TOC_CONTEXT_WINDOW = 5


def _clean_json(text: str) -> str:
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text.strip()
    return text[start : end + 1]


def _batch_lines_by_tokens(lines: list[str], max_tokens: int) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_tokens = 0

    for idx, line in enumerate(lines):
        t = _token_count(line)
        if current and current_tokens + t > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((idx, line))
        current_tokens += t

    if current:
        batches.append(current)
    return batches


def detect_heading_levels(lines: list[str]) -> dict[int, int]:
    """Trả về map {line_index: level}. level = -1 nghĩa là nội dung thường.

    Nếu 1 batch gọi LLM lỗi/parse fail, toàn bộ dòng trong batch đó được
    coi là nội dung thường (level=-1) thay vì làm sập cả pipeline — tương
    tự cơ chế retry/fallback bên production (parse_llm_output).
    """
    levels: dict[int, int] = {}
    previous_headings: list[str] = []

    for batch in _batch_lines_by_tokens(lines, TOC_BATCH_MAX_TOKENS):
        formatted = "\n".join(f"[{i}] {text}" for i, text in batch)
        prompt = TOC_EXTRACTION_PROMPT.format(
            previous_headings="\n".join(previous_headings[-TOC_CONTEXT_WINDOW:]) or "(không có)",
            lines=formatted,
        )

        try:
            response = chat(prompt)
            data = json.loads(_clean_json(response))
            for item in data.get("items", []):
                idx = int(item["id"])
                level = int(item["level"])
                levels[idx] = level
                if level != -1:
                    previous_headings.append(f"[{idx}] {lines[idx]} (level {level})")
        except Exception as exc:
            print(f"[toc-detect] batch lỗi, fallback toàn bộ batch = nội dung thường. err={exc}")
            for i, _ in batch:
                levels[i] = -1

    return levels
