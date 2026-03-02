import os
import httpx
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

payload = {
    "model": "mistral-large-latest",
    "messages": [
        {"role": "system", "content": "You are a clinical assistant. Return JSON object: {\"summary\": \"test\"}"},
        {"role": "user", "content": "Test"}
    ],
    "response_format": {"type": "json_object"}
}

resp = httpx.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json=payload,
    timeout=30
)
print("Status:", resp.status_code)
print("Response:", resp.text)
