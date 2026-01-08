from typing import Dict, Any, List, Tuple

from openai import OpenAI

from .config import get_settings


def generate_chr_draft(
    structured: Dict[str, Any],
    notes: str | None = None,
    context_chunks: List[Tuple[str, float]] | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    system = (
        "You are a clinical report assistant. Draft a clinician-facing summary "
        "based only on provided structured data, notes, and retrieved context. "
        "Be specific and clinically reasoned without making treatment decisions. "
        "Use the section headers below and cite sources using [#] where # is the context index.\n\n"
        "Sections:\n"
        "## Summary\n"
        "## Key Findings\n"
        "## Interpretation\n"
        "## Data Gaps\n"
        "## Follow-up Questions"
    )
    context_text = ""
    citations = []
    if context_chunks:
        formatted = []
        for idx, (chunk, score) in enumerate(context_chunks, start=1):
            formatted.append(f"[{idx}] {chunk}")
            citations.append({"index": idx, "score": score, "text": chunk})
        context_text = "\n\n".join(formatted)

    user = {
        "structured": structured,
        "notes": notes or "",
        "context": context_text,
    }

    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""
    return {
        "summary": content.strip(),
        "citations": citations,
    }


def query_chr(
    query: str,
    context_chunks: List[Tuple[str, float]] | None = None,
    patient_name: str | None = None,
) -> Dict[str, Any]:
    """Answer a specific clinical question using RAG context from patient documents."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    system = (
        "You are a clinical assistant helping a clinician answer specific questions about a patient's health data. "
        "Answer ONLY based on the provided context from the patient's medical documents. "
        "If the information is not available in the context, say so clearly. "
        "Be specific, cite sources using [#] notation, and provide clinically relevant insights. "
        "Format your response with clear sections if appropriate."
    )
    
    context_text = ""
    citations = []
    if context_chunks:
        formatted = []
        for idx, (chunk, score) in enumerate(context_chunks, start=1):
            formatted.append(f"[{idx}] {chunk}")
            citations.append({"index": idx, "score": score, "text": chunk})
        context_text = "\n\n".join(formatted)

    user_prompt = f"""
Patient: {patient_name or 'Unknown'}

Question: {query}

Relevant Context from Patient Documents:
{context_text if context_text else 'No relevant context found.'}

Please answer the question based on the above context.
"""

    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""
    return {
        "answer": content.strip(),
        "citations": citations,
        "query": query,
    }
