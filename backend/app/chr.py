from typing import Dict, Any, List

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .llm_gateway import create_chat_completion
from .phi import redact_payload, redact_text
from .auditor import audit_submission


def generate_chr_draft(
    structured: Dict[str, Any],
    notes: str | None = None,
    context_chunks: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")

    system = (
        "You are a clinical report assistant. Draft a clinician-facing summary "
        "based only on provided structured data, notes, and retrieved context. "
        "Be specific and clinically reasoned without making treatment decisions. "
        "Cite sources using [#] where # is the context index.\n\n"
        "Return a JSON object with keys: summary, key_findings, interpretation, data_gaps, "
        "follow_up_questions. Values should be strings or lists of strings."
    )
    context_text, citations = _format_context(context_chunks)

    user = {
        "structured": redact_payload(structured) if settings.phi_redaction_enabled else structured,
        "notes": redact_text(notes or "") if settings.phi_redaction_enabled else notes or "",
        "context": redact_text(context_text) if settings.phi_redaction_enabled else context_text,
    }

    resp = _create_chat_completion(
        settings.openai_model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or ""
    sections = _parse_sections(content)
    summary = _sections_to_markdown(sections) if sections else content.strip()
    
    # Phase 6: Run AI Auditor
    audit_result = audit_submission(summary, "system", context_chunks or [])
    
    return {
        "summary": summary,
        "sections": sections,
        "citations": citations,
        "audit_report": audit_result,
    }


def query_chr(
    query: str,
    context_chunks: List[Dict[str, Any]] | None = None,
    patient_name: str | None = None,
) -> Dict[str, Any]:
    """Answer a specific clinical question using RAG context from patient documents."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")

    system = (
        "You are a clinical assistant helping a clinician answer specific questions about a patient's health data. "
        "Answer ONLY based on the provided context from the patient's medical documents. "
        "If the information is not available in the context, say so clearly. "
        "Be specific, cite sources using [#] notation, and provide clinically relevant insights. "
        "Format your response with clear sections if appropriate."
    )
    
    context_text, citations = _format_context(context_chunks)

    safe_question = redact_text(query) if settings.phi_redaction_enabled else query
    safe_context = redact_text(context_text) if settings.phi_redaction_enabled else context_text
    user_prompt = f"""
Patient: {patient_name or 'Unknown'}

Question: {safe_question}

Relevant Context from Patient Documents:
{safe_context if safe_context else 'No relevant context found.'}

Please answer the question based on the above context.
"""

    resp = _create_chat_completion(
        settings.openai_model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = resp.choices[0].message.content or ""
    return {
        "answer": content.strip(),
        "citations": citations,
        "query": query,
    }


def _format_context(context_chunks: List[Dict[str, Any]] | None) -> tuple[str, List[Dict[str, Any]]]:
    if not context_chunks:
        return "", []
    formatted = []
    citations = []
    for idx, chunk in enumerate(context_chunks, start=1):
        meta = f"{chunk.get('filename', 'unknown')} (chunk {chunk.get('chunk_index', 'n/a')})"
        formatted.append(f"[{idx}] ({meta}) {chunk['chunk_text']}")
        citations.append(
            {
                "index": idx,
                "score": chunk.get("distance"),
                "text": chunk.get("chunk_text"),
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "content_type": chunk.get("content_type"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_start": chunk.get("chunk_start"),
                "chunk_end": chunk.get("chunk_end"),
                "extraction_id": chunk.get("extraction_id"),
            }
        )
    return "\n\n".join(formatted), citations


def _parse_sections(content: str) -> Dict[str, Any] | None:
    import json
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}")
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _sections_to_markdown(sections: Dict[str, Any]) -> str:
    order = [
        ("summary", "Summary"),
        ("key_findings", "Key Findings"),
        ("interpretation", "Interpretation"),
        ("data_gaps", "Data Gaps"),
        ("follow_up_questions", "Follow-up Questions"),
    ]
    parts: list[str] = []
    for key, title in order:
        value = sections.get(key)
        if value is None or value == "":
            continue
        parts.append(f"## {title}")
        if isinstance(value, list):
            parts.extend([f"- {item}" for item in value if item])
        else:
            parts.append(str(value))
    return "\n".join(parts).strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _create_chat_completion(
    model: str,
    messages: List[dict[str, str]],
    response_format: Dict[str, Any] | None = None,
):
    params: Dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
    if response_format:
        params["response_format"] = response_format
    try:
        return create_chat_completion(**params)
    except Exception:
        if response_format:
            params.pop("response_format", None)
            return create_chat_completion(**params)
        raise
