import json
import re

from llm import chat
from prompts import LABEL_PROMPT


def clean_json(text):

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    return text[start:end + 1]


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

    prompt = LABEL_PROMPT.format(
        question=question,
        chunks=build_chunks(candidate_ids, chunks)
    )

    response = chat(prompt)

    data = json.loads(clean_json(response))

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

        cleaned.append({
            "chunk_id": cid,
            "relevance": int(item.get("relevance", 0))
        })

    cleaned = [
        r for r in cleaned
        if r["relevance"] > 0
    ]

    cleaned.sort(
        key=lambda x: x["relevance"],
        reverse=True
    )

    return cleaned