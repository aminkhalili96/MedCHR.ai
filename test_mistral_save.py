import sys
import os

try:
    from httpx import Client
    import json
    
    # We load manually to skip the web app container
    with open(".env", "r") as f:
        env = dict(line.strip().split("=", 1) for line in f if "=" in line and not line.startswith("#"))
        
    api_key = env.get("MISTRAL_API_KEY", "").strip('"').strip("'")
    
    with Client() as client:
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "system", "content": "You are a clinical assistant. Return JSON object: {\"summary\": \"test\"}"},
                {"role": "user", "content": "Test"}
            ],
            "response_format": {"type": "json_object"}
        }
        res = client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=15
        )
        with open("/tmp/mistral_error.log", "w") as out:
            out.write(f"Status: {res.status_code}\n")
            out.write(f"Body: {res.text}\n")
            
except Exception as e:
    with open("/tmp/mistral_error.log", "w") as out:
        out.write(str(e))
