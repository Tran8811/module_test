import json
import re
import math

from .config import CANDIDATE_PREFILTER_LIMIT, CANDIDATE_CHUNK_SNIPPET_TOKENS
from .llm import chat, _estimate_payload_tokens, MODEL_CONTEXT_WINDOW, CONTEXT_WINDOW_SAFETY_MARGIN, MAX_TOKENS
from .prompts import CANDIDATE_PROMPT

# JSON schema ép output của LLM (response_format kiểu OpenAI structured
# output, vLLM/SGLang hỗ trợ native) -- thay cho việc dặn format JSON bằng
# lời trong CANDIDATE_PROMPT.
CANDIDATE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidate_chunks",
        "schema": {
            "type": "object",
            "properties": {
                "candidate_chunks": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            },
            "required": ["candidate_chunks"],
        },
    },
}


def clean_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    return text[start:end + 1]


def build_chunks(chunks):
    """Hiển thị cho LLM một ID CỤC BỘ (vị trí trong danh sách `chunks`
    truyền vào, bắt đầu từ 0) thay vì chunk_id thật (UUID string).

    Lý do: CANDIDATE_PROMPT yêu cầu LLM trả về "integer chunk IDs" --
    nếu đưa thẳng UUID cho nó xem, LLM không có cách nào round-trip lại
    đúng UUID đó ở output (dễ bịa số, không khớp chunk nào cả), khiến
    downstream không map lại được về đúng chunk dù nội dung đã chọn đúng.
    ID cục bộ chỉ có ý nghĩa trong phạm vi 1 lần gọi build_chunks/prompt
    này; cần map ngược lại chunk_id thật ngay sau khi parse response.
    """
    result = []

    for local_id, chunk in enumerate(chunks):
        result.append(
            f"""
ID: {local_id}

Content:
{chunk["text"]}

------------------------
"""
        )

    return "\n".join(result)

_VN_UPPER = "A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÐĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
_VN_LOWER = "a-zàáảãạăắằẳẵặâấầẩẫậðđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"

_PROPER_NOUN_RE = re.compile(
    rf"(?:[{_VN_UPPER}][{_VN_LOWER}]*(?:\s+[{_VN_UPPER}][{_VN_LOWER}]*){{1,}})"
)

_idf_cache = {}


def _extract_phrases(question):
    """Trích xuất những tên riêng, vd: Nguyễn Văn A để tìm """
    return [p.strip() for p in _PROPER_NOUN_RE.findall(question) if len(p.strip()) > 3]


def _build_idf_index(chunks):
    cache_key = id(chunks)
    cached = _idf_cache.get(cache_key)
    if cached is not None:
        return cached

    n_docs = len(chunks)
    doc_freq = {}
    for chunk in chunks:
        tokens = set(re.findall(r"\w+", chunk["text"].lower()))
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    idf = {
        token: math.log((n_docs + 1) / (df + 1)) + 1.0
        for token, df in doc_freq.items()
    }
    _idf_cache[cache_key] = idf
    return idf


def _score_chunk(question, chunk_text, idf, phrases):
    question_tokens = set(re.findall(r"\w+", question.lower()))
    if not question_tokens:
        return 0

    text_lower = chunk_text.lower()

    score = 0.0
    for token in question_tokens:
        count = text_lower.count(token)
        if count:
            score += count * idf.get(token, 1.0)

    for phrase in phrases:
        if phrase.lower() in text_lower:
            score += 50.0

    return score


def _prefilter_chunks(question, chunks, limit):
    if len(chunks) <= limit:
        return chunks

    idf = _build_idf_index(chunks)
    phrases = _extract_phrases(question)

    scored = [
        (_score_chunk(question, chunk["text"], idf, phrases), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    filtered = [chunk for score, chunk in scored[:limit] if score > 0]
    if not filtered:
        filtered = [chunk for _, chunk in scored[:limit]]

    return filtered


def _expand_neighbors(candidate_chunks, all_chunks, max_neighbors=3):
    """Mở rộng candidate_chunks bằng các chunk liền kề (cùng source)
    trong `all_chunks`.

    Trước đây hàm này cộng/trừ trực tiếp lên chunk["chunk_id"] để suy ra
    "chunk liền kề" (neighbor_id = cid + delta). Cách này CHỈ đúng nếu
    chunk_id là số nguyên liên tục = vị trí trong all_chunks. Nhưng
    chunk_id thực tế là UUID string (gán trong chunker.py) -> cid + delta
    sẽ raise TypeError ngay khi hàm này được gọi (qtype == "table" hoặc
    is_multi_hop).

    Sửa: dùng chunk_id chỉ để ĐỊNH DANH (dedupe, tra cứu), còn việc suy ra
    "liền kề" thì tra theo VỊ TRÍ THẬT của chunk trong all_chunks, thông
    qua id_to_pos.
    """
    if not candidate_chunks:
        return candidate_chunks

    id_to_pos = {c["chunk_id"]: pos for pos, c in enumerate(all_chunks)}

    selected_ids = {chunk["chunk_id"] for chunk in candidate_chunks}
    expanded_chunks = list(candidate_chunks)

    for chunk in candidate_chunks:
        source = chunk.get("metadata", {}).get("source")
        pos = id_to_pos.get(chunk["chunk_id"])
        if pos is None:
            # Chunk này không nằm trong all_chunks (không rõ vị trí gốc)
            # -> không thể suy ra chunk liền kề, bỏ qua.
            continue

        for delta in (1, -1, 2, -2, 3, -3):
            neighbor_pos = pos + delta
            if neighbor_pos < 0 or neighbor_pos >= len(all_chunks):
                continue

            neighbor = all_chunks[neighbor_pos]
            if neighbor["chunk_id"] in selected_ids:
                continue
            if neighbor.get("metadata", {}).get("source") != source:
                continue

            selected_ids.add(neighbor["chunk_id"])
            expanded_chunks.append(neighbor)
            if len(expanded_chunks) >= len(candidate_chunks) + max_neighbors:
                break

        if len(expanded_chunks) >= len(candidate_chunks) + max_neighbors:
            break

    return expanded_chunks


def retrieve_candidates(question, chunks, top_k=5):
    if isinstance(question, dict):
        question_text = question.get("question", "")
        qtype = question.get("type")
        difficulty = question.get("difficulty")
    else:
        question_text = str(question)
        qtype = None
        difficulty = None

    is_multi_hop = qtype == "multi-hop" or difficulty == "hard"
    prefilter_limit = CANDIDATE_PREFILTER_LIMIT * (2 if is_multi_hop else 1)

    
    HARD_MAX_CANDIDATES = 120
    prefilter_limit = min(prefilter_limit, HARD_MAX_CANDIDATES)

    candidate_chunks = _prefilter_chunks(
        question_text,
        chunks,
        prefilter_limit
    )

    if qtype == "table" or is_multi_hop:
        expanded = _expand_neighbors(candidate_chunks, chunks, max_neighbors=6)
        if len(expanded) > len(candidate_chunks):
            print(f"Expanded candidate chunks from {len(candidate_chunks)} to {len(expanded)} for query type={qtype}")
            candidate_chunks = expanded[:HARD_MAX_CANDIDATES]

    def _build_prompt(chunks_list, snippet_tokens=None):
        if snippet_tokens and snippet_tokens > 0:
            def _snippet(c):
                parts = c["text"].split()
                if len(parts) <= snippet_tokens:
                    return c["text"]
                return " ".join(parts[:snippet_tokens]) + " ..."

            chunks_for_prompt = [
                {**c, "text": _snippet(c)} for c in chunks_list
            ]
        else:
            chunks_for_prompt = chunks_list

        return CANDIDATE_PROMPT.replace("{question}", question_text).replace(
            "{chunks}", build_chunks(chunks_for_prompt)
        ).replace("{top_k}", str(top_k))

    snippet_tokens = CANDIDATE_CHUNK_SNIPPET_TOKENS * (2 if is_multi_hop else 1)
    prompt = _build_prompt(candidate_chunks, snippet_tokens=snippet_tokens)

    allowed = MODEL_CONTEXT_WINDOW - CONTEXT_WINDOW_SAFETY_MARGIN - MAX_TOKENS
    est = _estimate_payload_tokens(prompt)

    min_candidates = min(5, len(candidate_chunks))

    while est > allowed and len(candidate_chunks) > min_candidates:
        trim_n = max(1, len(candidate_chunks) // 5)
        candidate_chunks = candidate_chunks[:-trim_n]
        prompt = _build_prompt(candidate_chunks, snippet_tokens=snippet_tokens)
        est = _estimate_payload_tokens(prompt)

    while est > allowed and snippet_tokens > 20:
        snippet_tokens = max(20, snippet_tokens // 2)
        prompt = _build_prompt(candidate_chunks, snippet_tokens=snippet_tokens)
        est = _estimate_payload_tokens(prompt)

    if est > allowed:
        candidate_chunks = candidate_chunks[:min_candidates]
        snippet_tokens = 20
        prompt = _build_prompt(candidate_chunks, snippet_tokens=snippet_tokens)

    used_chunks = candidate_chunks

    try:
        response = chat(prompt, response_format=CANDIDATE_RESPONSE_FORMAT)
    except Exception:
        used_chunks = candidate_chunks[:min_candidates]
        prompt = _build_prompt(used_chunks, snippet_tokens=20)
        response = chat(prompt, response_format=CANDIDATE_RESPONSE_FORMAT)

    data = json.loads(clean_json(response))

    # LLM trả về ID CỤC BỘ (vị trí 0..len(used_chunks)-1 trong danh sách
    # đã đưa cho nó qua build_chunks() ở prompt cuối cùng, khớp với
    # `used_chunks` -- không phải chunk_id thật). Phải map ngược lại
    # chunk_id thật (UUID) ở đây; nếu trả thẳng số nguyên LLM sinh ra,
    # downstream sẽ không khớp được với chunk_id thật -> "truy xuất được
    # tài liệu nhưng không tìm thấy câu trả lời".
    local_ids = data.get("candidate_chunks", [])

    selected_chunk_ids = []
    seen = set()
    for lid in local_ids:
        if not isinstance(lid, int):
            continue
        if lid < 0 or lid >= len(used_chunks):
            # LLM bịa ra ID không nằm trong danh sách đã đưa -> bỏ qua,
            # không cố đoán/ép về chunk nào cả.
            continue
        real_id = used_chunks[lid]["chunk_id"]
        if real_id not in seen:
            selected_chunk_ids.append(real_id)
            seen.add(real_id)

    return selected_chunk_ids