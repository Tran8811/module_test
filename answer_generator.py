import json
import re

from llm import chat
from prompts import ANSWER_PROMPT


def clean_json(text):

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    start = text.find("{")
    end = text.rfind("}")

    return text[start:end + 1]


def build_chunks(labels, chunks):

    result = []

    for item in labels:

        cid = item["chunk_id"]

        result.append(
            f"""
ID: {cid}

Content:
{chunks[cid]["text"]}

------------------------
"""
        )

    return "\n".join(result)


def generate_answer(question, labels, chunks):

    prompt = ANSWER_PROMPT.format(
        question=question,
        chunks=build_chunks(labels, chunks)
    )

    response = chat(prompt)

    data = json.loads(clean_json(response))

    return data["answer"]