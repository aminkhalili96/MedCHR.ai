import asyncio
from backend.app.config import Settings
from backend.app.llm_gateway import get_openai_client

async def main():
    settings = Settings()
    print(f"Using model: {settings.openai_model}")
    print(f"Mistral Key length: {len(settings.mistral_api_key) if settings.mistral_api_key else 0}")
    
    try:
        client = get_openai_client(settings)
        print("Client initialized")
        
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": "Say 'hello world' if you are working."}],
            max_tokens=10
        )
        print("Success:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", type(e).__name__, str(e))

if __name__ == "__main__":
    asyncio.run(main())
