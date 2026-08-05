import json
import re

from test_generator.llm import chat
from test_generator.prompts import CANDIDATE_PROMPT


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

def retrieve_candidates(question, chunks, top_k=5):

    prompt = CANDIDATE_PROMPT.replace(
        "{question}",
        question
    ).replace(
        "{chunks}",
        build_chunks(chunks)
    ).replace(
        "{top_k}",
        str(top_k)
    )

    response = chat(prompt)

    data = json.loads(clean_json(response))

    return data["candidate_chunks"]