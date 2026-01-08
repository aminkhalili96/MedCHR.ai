from typing import List

from openai import OpenAI

from .config import get_settings


def embed_texts(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    return [item.embedding for item in resp.data]
