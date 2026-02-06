from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .llm_gateway import create_embedding, redact_if_enabled


def embed_texts(texts: List[str]) -> List[List[float]]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    safe_texts = [redact_if_enabled(text) for text in texts]
    resp = _create_embeddings(settings.openai_embedding_model, safe_texts)
    return [item.embedding for item in resp.data]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _create_embeddings(model: str, inputs: List[str]):
    return create_embedding(
        model=model,
        inputs=inputs,
    )
