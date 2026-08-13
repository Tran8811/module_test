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

def _expand_table_neighbors(labels, candidate_ids, chunks, window=2, min_relevance=2):
    """For table-type questions, a table is frequently split across
    consecutive chunks during chunking (header/definitions in one chunk,
    data rows or formula notes in the next). The LABEL_PROMPT scores each
    candidate chunk in isolation, so a continuation chunk that doesn't
    obviously mention the question's keywords can get relevance=0 and be
    dropped -- even though it's the piece that actually answers "how is
    it calculated". This re-adds same-source neighbor chunks (within
    `window` chunk_ids) of any highly-relevant labeled chunk, as long as
    they were already surfaced by retrieval (candidate_ids), so the
    answer step gets the full contiguous table context instead of just
    the one fragment the model happened to score highly.
    """
    by_id = _index_by_id(chunks)
    candidate_set = set(candidate_ids)
    labeled_ids = {item["chunk_id"] for item in labels}

    extra = []
    for item in labels:
        if item["relevance"] < min_relevance:
            continue

        anchor = by_id.get(item["chunk_id"])
        if anchor is None:
            continue
        source = anchor.get("metadata", {}).get("source")

        for delta in range(-window, window + 1):
            if delta == 0:
                continue
            neighbor_id = item["chunk_id"] + delta
            if neighbor_id in labeled_ids or neighbor_id not in candidate_set:
                continue

            neighbor = by_id.get(neighbor_id)
            if neighbor is None:
                continue
            if neighbor.get("metadata", {}).get("source") != source:
                continue

            # Nominal relevance so it's kept (relevance > 0) and sorted
            # below chunks the model was actually confident about.
            extra.append({"chunk_id": neighbor_id, "relevance": 1})
            labeled_ids.add(neighbor_id)

    return labels + extra


def generate_labels(question, candidate_ids, chunks):

    if isinstance(question, dict):
        question_text = question.get("question", "")
        qtype = question.get("type")
    else:
        question_text = str(question)
        qtype = None

    prompt = LABEL_PROMPT.replace(
        "{question}",
        question_text
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

    if qtype == "table":
        cleaned = _expand_table_neighbors(cleaned, candidate_ids, chunks)

    cleaned.sort(
        key=lambda x: x["relevance"],
        reverse=True
    )

    return cleaned