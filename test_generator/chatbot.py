import requests

url = "http://10.0.128.19:8010/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy"
}

payload = {
    "model": "llm-model",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    "temperature": 0.2,
    "max_tokens": 1000
}

response = requests.post(
    url,
    headers=headers,
    json=payload
)

print(response.json()["choices"][0]["message"]["content"])