"""
Thay cho GENERIC_HEADING_PATTERNS (regex) trong hierarchical_chunker.py cũ:
dùng LLM đọc và suy luận cấp tiêu đề, giống hệt cơ chế docx2tree() /
PROMPT_EXTRACT_TOC của hệ thống indexing production.
"""
import json

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    repair_json = None

from .chunker import _token_count
from .llm import chat
from .prompts_production import TOC_EXTRACTION_PROMPT

TOC_BATCH_MAX_TOKENS = 4000
TOC_CONTEXT_WINDOW = 5

# Ước lượng token cho mỗi item JSON dạng {"id": 123, "level": 2},
# Log thực tế cho thấy 15 token/item vẫn còn hơi sát -> tăng lên và thêm
# hệ số an toàn nhân thêm để tránh bị cắt cụt sát nút như đã gặp phải.
TOKENS_PER_JSON_ITEM = 22
JSON_WRAPPER_OVERHEAD_TOKENS = 100
OUTPUT_TOKEN_SAFETY_FACTOR = 1.25


def _clean_json(text: str) -> str:
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text.strip()
    return text[start : end + 1]


def _attempt_close_truncated_json(text: str) -> str | None:
    """Vá tạm cho trường hợp response bị cắt cụt giữa chừng do chạm
    max_tokens: text vẫn còn dạng list các object hợp lệ nhưng thiếu phần
    đóng "]}" ở cuối (hoặc object cuối cùng bị dở dang).

    Cắt bỏ phần object cuối cùng nếu nó chưa hoàn chỉnh, rồi tự đóng lại
    mảng/JSON object. Không đảm bảo lấy được 100% dữ liệu của batch, nhưng
    cứu được phần lớn thay vì mất trắng toàn bộ batch.
    """
    text = text.rstrip()
    if not text:
        return None

    last_complete = text.rfind("},")
    if last_complete == -1:
        last_complete = text.rfind("}")
        if last_complete == -1:
            return None
        candidate = text[: last_complete + 1]
    else:
        candidate = text[: last_complete + 1]

    if not candidate.rstrip().endswith(("}", "},")):
        return None

    candidate = candidate.rstrip().rstrip(",")
    return candidate + "]}"


def _parse_llm_json(raw: str) -> dict | None:
    cleaned = _clean_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if repair_json is not None:
        try:
            return json.loads(repair_json(cleaned))
        except Exception:
            pass

    # Lưới an toàn cuối: thử tự đóng JSON bị cắt cụt do hết max_tokens,
    # giữ lại phần dữ liệu hợp lệ đã sinh ra được thay vì mất trắng.
    patched = _attempt_close_truncated_json(cleaned)
    if patched is not None:
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            if repair_json is not None:
                try:
                    return json.loads(repair_json(patched))
                except Exception:
                    return None

    return None


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
    levels: dict[int, int] = {}
    previous_headings: list[str] = []

    for batch in _batch_lines_by_tokens(lines, TOC_BATCH_MAX_TOKENS):
        formatted = "\n".join(f"[{i}] {text}" for i, text in batch)
        prompt = TOC_EXTRACTION_PROMPT.format(
            previous_headings="\n".join(previous_headings[-TOC_CONTEXT_WINDOW:]) or "(không có)",
            lines=formatted,
        )

        needed_output_tokens = int(
            (len(batch) * TOKENS_PER_JSON_ITEM + JSON_WRAPPER_OVERHEAD_TOKENS)
            * OUTPUT_TOKEN_SAFETY_FACTOR
        )

        try:
            response = chat(prompt, max_tokens=needed_output_tokens)
        except Exception as exc:
            print(f"[toc-detect] lỗi gọi LLM cho batch ({len(batch)} dòng): {exc}")
            for i, _ in batch:
                levels[i] = -1
            continue

        data = _parse_llm_json(response)

        if data is None:
            print(
                f"[toc-detect] batch lỗi ({len(batch)} dòng, "
                f"max_tokens={needed_output_tokens}), fallback toàn bộ batch "
                f"= nội dung thường. response_len={len(response)}, "
                f"300 ký tự cuối: ...{response[-300:]}"
            )
            for i, _ in batch:
                levels[i] = -1
            continue

        parsed_ids = set()
        try:
            for item in data.get("items", []):
                idx = int(item["id"])
                level = int(item["level"])
                levels[idx] = level
                parsed_ids.add(idx)
                if level != -1:
                    previous_headings.append(f"[{idx}] {lines[idx]} (level {level})")
        except (KeyError, ValueError, IndexError) as exc:
            print(
                f"[toc-detect] batch parse được JSON nhưng dữ liệu items "
                f"không hợp lệ, fallback toàn bộ batch = nội dung thường. err={exc}"
            )
            for i, _ in batch:
                levels[i] = -1
            continue

        # Nếu JSON bị vá (do cắt cụt) thì có thể thiếu vài dòng cuối batch
        # so với parsed_ids -> fallback level=-1 cho riêng những dòng đó
        # thay vì để trống hoàn toàn (dòng không có trong dict levels).
        missing = [i for i, _ in batch if i not in parsed_ids]
        if missing:
            print(f"[toc-detect] batch bị vá do cắt cụt, thiếu {len(missing)}/{len(batch)} dòng -> gán level=-1")
            for i in missing:
                levels[i] = -1

    return levels