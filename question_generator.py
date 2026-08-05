import json
import re

from llm import chat
from prompts import QUESTION_PROMPT


def clean_json(text: str) -> str:
    text = text.replace("```json", "")
    text = text.replace("```", "")
    return text.strip()


def parse_question_response(text: str) -> dict:
    clean_text = clean_json(text)
    if not clean_text:
        return {"questions": []}

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        match = re.search(r'"questions"\s*:\s*(\[.*\])', text, re.DOTALL)
        if match:
            try:
                return {"questions": json.loads(match.group(1))}
            except json.JSONDecodeError:
                pass
        return {"questions": []}


def validate_question_item(item):
    if not isinstance(item, dict):
        return False
    if "question" not in item:
        return False
    if item.get("type") not in {"one-hop", "multi-hop", "table"}:
        return False
    if item.get("difficulty") not in {"easy", "medium", "hard"}:
        return False
    return True


def generate_questions(chunk):

    prompt = QUESTION_PROMPT.replace(
        "{chunk}",
        chunk["text"]
    )

    response = chat(prompt)
    data = parse_question_response(response)

    questions = []
    for item in data.get("questions", []):
        if validate_question_item(item):
            questions.append(item)

    return questions