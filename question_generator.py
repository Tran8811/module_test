import json

from llm import chat
from prompts import QUESTION_PROMPT


def clean_json(text: str) -> str:
    text = text.replace("```json", "")
    text = text.replace("```", "")
    return text.strip()


def generate_questions(chunk):

    prompt = QUESTION_PROMPT.format(
        chunk=chunk["text"]
    )

    response = chat(prompt)

    response = clean_json(response)

    data = json.loads(response)

    return data["questions"]