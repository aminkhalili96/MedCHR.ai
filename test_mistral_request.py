import requests

try:
    print("Testing report generation...")
    # Get a CSRF token first
    session = requests.Session()
    session.get('http://127.0.0.1:8000/ui/login')
    
    # We can skip the UI auth by using the backend api route directly if it exists, 
    # but the simplest way is to test the mistral call directly
    import asyncio
    from app.config import get_settings
    from app.llm_gateway import get_openai_client
    
    async def run_mistral():
        settings = get_settings()
        client = get_openai_client(timeout_seconds=60)
        try:
            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are a clinical assistant. Return JSON object."},
                    {"role": "user", "content": "Test"}
                ],
                response_format={"type": "json_object"}
            )
            print("Success:", resp.choices[0].message.content)
        except Exception as e:
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            
    asyncio.run(run_mistral())
except Exception as e:
    print("Outside Exception:", e)
