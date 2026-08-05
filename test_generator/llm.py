import requests

from config import *

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy"
}


def chat(prompt: str):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    response = requests.post(
        LLM_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]