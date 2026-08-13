import json
import re

from .llm import chat
from .prompts import ANSWER_PROMPT


def clean_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return text[start:].strip()


def parse_answer_response(text):
    """Robustly extract an `answer` from LLM output.

    Tries: balanced-JSON extraction, JSON parse, key/value regexes,
    header-style lines, keyword heuristics, then a fallback to raw text.
    """
    cleaned = clean_json(text)
    try:
        return json.loads(cleaned)
    except Exception:
        # Try common key:value patterns for `answer`
        m = re.search(r'["\']?answer["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
        if m:
            return {"answer": m.group(1).strip()}

        # Look for lines like "Answer: ..." or "Kết luận: ..."
        m2 = re.search(r'^(?:Answer|Kết luận|Kết quả)[:\s-]*([^\n]+)', text, re.IGNORECASE | re.MULTILINE)
        if m2:
            return {"answer": m2.group(1).strip()}

        # If the model explicitly says it cannot answer, map that to a safe message
        if re.search(r'not enough information|không đủ|insufficient', text, re.IGNORECASE):
            return {"answer": "Not enough information to answer the question."}

        # As a last resort, return the full trimmed text as the answer if non-empty
        if text and text.strip():
            return {"answer": text.strip()}

        return {"answer": "Not enough information to answer the question."}


def _index_by_id(chunks):
    """Build a chunk_id -> chunk lookup.

    IMPORTANT: `chunks` is not guaranteed to be a list indexed by
    chunk_id (subsets, filtering, or non-contiguous ids all break that
    assumption). Always look chunks up by their `chunk_id` field, never
    by list position.
    """
    return {c["chunk_id"]: c for c in chunks}


def build_chunks(labels, chunks):

    by_id = _index_by_id(chunks)

    result = []

    for item in labels:

        cid = item["chunk_id"]

        chunk = by_id.get(cid)
        if chunk is None:
            # labeled id doesn't correspond to any known chunk; skip it
            # rather than silently pulling in the wrong chunk by position.
            continue

        result.append(
            f"""
ID: {cid}

Content:
{chunk["text"]}

------------------------
"""
        )

    return "\n".join(result)


def generate_answer(question, labels, chunks):
    # If there are no labeled supporting chunks, bail out with a safe message.
    if not labels:
        return "Not enough information to answer the question."

    prompt = ANSWER_PROMPT.replace("{question}", question).replace(
        "{chunks}", build_chunks(labels, chunks)
    )

    response = chat(prompt)

    data = parse_answer_response(response)

    return data.get("answer", "Not enough information to answer the question.")