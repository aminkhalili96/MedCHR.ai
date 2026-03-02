import asyncio
from backend.app.config import Settings
from backend.app.llm_gateway import get_openai_client
import sys

async def main():
    settings = Settings()
    client = get_openai_client(settings=settings)
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a clinical assistant. Return JSON: {\"summary\": \"test\"}"},
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
