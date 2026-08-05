import json
import re

from test_generator.llm import chat
from test_generator.prompts import LABEL_PROMPT


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


def build_chunks(candidate_ids, chunks):

    result = []

    for cid in candidate_ids:

        result.append(
            f"""
ID: {cid}

Content:
{chunks[cid]["text"]}

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