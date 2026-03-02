import asyncio
import os
import sys

# Ensure backend acts as the root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.llm_gateway import get_openai_client

async def main():
    settings = get_settings()
    print("Mistral Key:", bool(settings.mistral_api_key))
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a clinical assistant. Return JSON object with summary key."},
                {"role": "user", "content": "Test patient"}
            ],
            response_format={"type": "json_object"}
        )
        print("Success:", resp.choices[0].message.content)
    except Exception as e:
        print("Exception caught:", type(e).__name__)
        print(e)
        
if __name__ == "__main__":
    asyncio.run(main())
