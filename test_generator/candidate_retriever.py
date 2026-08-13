import json
import re
import math

from .config import CANDIDATE_PREFILTER_LIMIT, CANDIDATE_CHUNK_SNIPPET_TOKENS
from .llm import chat, _estimate_payload_tokens, MODEL_CONTEXT_WINDOW, CONTEXT_WINDOW_SAFETY_MARGIN, MAX_TOKENS
from .prompts import CANDIDATE_PROMPT


def clean_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    return text[start:end + 1]


def build_chunks(chunks):

    result = []

    for chunk in chunks:
        result.append(
            f"""
ID: {chunk["chunk_id"]}

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
    if not candidate_chunks:
        return candidate_chunks

    selected_ids = {chunk["chunk_id"] for chunk in candidate_chunks}
    expanded_chunks = []

    for chunk in candidate_chunks:
        expanded_chunks.append(chunk)
        source = chunk.get("metadata", {}).get("source")
        cid = chunk["chunk_id"]

        for delta in (1, -1, 2, -2, 3, -3):
            neighbor_id = cid + delta
            if neighbor_id in selected_ids:
                continue
            if neighbor_id < 0 or neighbor_id >= len(all_chunks):
                continue

            neighbor = all_chunks[neighbor_id]
            if neighbor.get("metadata", {}).get("source") != source:
                continue

            selected_ids.add(neighbor_id)
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

    try:
        response = chat(prompt)
    except Exception:
        fallback_chunks = candidate_chunks[:min_candidates]
        prompt = _build_prompt(fallback_chunks, snippet_tokens=20)
        response = chat(prompt)

    data = json.loads(clean_json(response))

    return data["candidate_chunks"]