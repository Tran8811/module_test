import json
import re

from .llm import chat
from .prompts import LABEL_PROMPT


def clean_json(text):

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return text.strip()

    return text[start:end + 1]


def parse_label_response(text):
    clean_text = clean_json(text)

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        results = []
        pattern = re.compile(
            r"chunk_id\s*[\":=]*\s*([0-9]+).*?relevance\s*[\":=]*\s*([0-9]+)",
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(text):
            results.append(
                {
                    "chunk_id": int(match.group(1)),
                    "relevance": int(match.group(2)),
                }
            )
        return {"results": results}


def _index_by_id(chunks):
    """Build a chunk_id -> chunk lookup.

    IMPORTANT: `chunks` is not guaranteed to be a list indexed by
    chunk_id (subsets, filtering, or non-contiguous ids all break that
    assumption). Always look chunks up by their `chunk_id` field, never
    by list position.
    """
    return {c["chunk_id"]: c for c in chunks}


def build_chunks(candidate_ids, chunks):

    by_id = _index_by_id(chunks)

    result = []

    for cid in candidate_ids:

        chunk = by_id.get(cid)
        if chunk is None:
            # candidate id doesn't correspond to any known chunk; skip it
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

def generate_labels(question, candidate_ids, chunks):

    prompt = LABEL_PROMPT.replace(
        "{question}",
        question
    ).replace(
        "{chunks}",
        build_chunks(candidate_ids, chunks)
    )

    response = chat(prompt)

    data = parse_label_response(response)

    results = data.get("results", [])

    cleaned = []

    for item in results:

        cid = item.get("chunk_id")

        # Nếu Gemma trả "Chunk 2"
        if isinstance(cid, str):

            m = re.search(r"\d+", cid)

            if m:
                cid = int(m.group())
            else:
                continue

        relevance = int(item.get("relevance", 0))

        if relevance > 0:
            cleaned.append({
                "chunk_id": cid,
                "relevance": relevance
            })

    cleaned.sort(
        key=lambda x: x["relevance"],
        reverse=True
    )

    return cleaned