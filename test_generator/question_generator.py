import json
import re

from .llm import chat
from .prompts import QUESTION_PROMPT


def clean_json(text: str) -> str:
    text = text.replace("```json", "")
    text = text.replace("```", "")
    return text.strip()


def parse_question_response(text: str) -> dict:
    clean_text = clean_json(text)
    if not clean_text:
        return {"questions": []}

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r'"questions"\s*:\s*(\[.*\])', text, re.DOTALL)
        if match:
            try:
                return {"questions": json.loads(match.group(1))}
            except json.JSONDecodeError:
                pass
        return {"questions": []}


def validate_question_item(item):
    if not isinstance(item, dict):
        return False
    if "question" not in item:
        return False
    if item.get("type") not in {"one-hop", "multi-hop", "table"}:
        return False
    if item.get("difficulty") not in {"easy", "medium", "hard"}:
        return False
    return True


def _normalize_source_chunk_ids(item, main_chunk_id, related_ids):
    """Chuẩn hoá field source_chunk_ids trả về từ LLM.

    - Nếu model trả về hợp lệ (list không rỗng, id nằm trong tập chunk đã
      đưa vào prompt: chunk chính + related_chunks) -> giữ nguyên, dedupe.
    - Nếu model quên trả field này, trả rỗng, hoặc trả id "bịa" không nằm
      trong tập chunk đã cung cấp -> fallback:
        + one-hop / table: gán về [main_chunk_id]
        + multi-hop: gán về [main_chunk_id] + related_ids (an toàn hơn để
          trống, tránh mất ground truth khiến retrieval đúng nhưng vẫn bị
          coi là "không tìm thấy câu trả lời" ở bước đánh giá sau này).
    """
    valid_ids = {main_chunk_id, *related_ids}

    raw = item.get("source_chunk_ids")
    if isinstance(raw, list) and raw:
        cleaned = []
        seen = set()
        for cid in raw:
            cid = str(cid)
            if cid in {str(v) for v in valid_ids} and cid not in seen:
                cleaned.append(cid)
                seen.add(cid)
        if cleaned:
            return cleaned

    # Fallback khi thiếu / rỗng / toàn id không khớp tập chunk đã cho.
    if item.get("type") == "multi-hop":
        return [str(main_chunk_id)] + [str(r) for r in related_ids]
    return [str(main_chunk_id)]


def _format_related_chunks(chunks):
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source", "unknown")
        parts.append(
            f"ID: {chunk['chunk_id']} (source: {source})\nContent:\n{chunk['text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------
# Chấm điểm độ liên quan giữa 2 đoạn text bằng word-overlap đơn giản
# (không cần import candidate_retriever để tránh phụ thuộc vòng, nhưng
# cùng ý tưởng: đếm số từ chung, ưu tiên từ dài/hiếm hơn).
# ---------------------------------------------------------------------
def _overlap_score(text_a: str, text_b: str) -> int:
    tokens_a = set(re.findall(r"\w{4,}", text_a.lower()))  # bỏ từ quá ngắn (ít thông tin)
    tokens_b = set(re.findall(r"\w{4,}", text_b.lower()))
    return len(tokens_a & tokens_b)


def _select_related_chunks(chunk, all_chunks, limit=2):
    source = chunk.get("metadata", {}).get("source")
    same_source = []
    other_source = []

    for other in all_chunks:
        if other["chunk_id"] == chunk["chunk_id"]:
            continue
        if source and other.get("metadata", {}).get("source") == source:
            same_source.append(other)
        else:
            other_source.append(other)

    related = []
    if same_source:
        related.append(same_source[0])

    if other_source and len(related) < limit:
        # QUAN TRỌNG: không lấy other_source[0] (chunk đầu tiên bất kỳ,
        # ngẫu nhiên, có thể hoàn toàn lạc đề) -- chọn chunk khác nguồn
        # NỘI DUNG GẦN NHẤT với main chunk, để câu hỏi multi-hop sinh ra
        # thực sự kết hợp 2 nội dung có liên quan, tránh model lẫn lộn
        # chi tiết giữa 2 tài liệu không ăn nhập.
        scored = [(_overlap_score(chunk["text"], o["text"]), o) for o in other_source]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_chunk = scored[0]
        if best_score > 0:
            related.append(best_chunk)
        # Nếu điểm overlap cao nhất vẫn = 0 (không chunk nào khác nguồn có
        # từ chung nào với main chunk) -> KHÔNG thêm related chunk khác
        # nguồn nữa, thà thiếu còn hơn thêm nhiễu hoàn toàn không liên quan.

    for other in same_source[1:limit]:
        if len(related) < limit:
            related.append(other)

    return related


def generate_questions(chunk, all_chunks=None):
    related_chunks = []
    if all_chunks is not None:
        related_chunks = _select_related_chunks(chunk, all_chunks, limit=2)

    related_ids = [c["chunk_id"] for c in related_chunks]

    prompt = QUESTION_PROMPT.replace(
        "{chunk_id}",
        str(chunk["chunk_id"])
    ).replace(
        "{chunk}",
        chunk["text"]
    ).replace(
        "{related_chunks}",
        _format_related_chunks(related_chunks)
    )

    response = chat(prompt)
    data = parse_question_response(response)

    questions = []
    for item in data.get("questions", []):
        if not validate_question_item(item):
            continue

        item["source_chunk_ids"] = _normalize_source_chunk_ids(
            item, chunk["chunk_id"], related_ids
        )
        questions.append(item)

    return questions