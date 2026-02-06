from __future__ import annotations

from time import monotonic
from typing import Any

from openai import OpenAI
from psycopg.types.json import Json

from .config import get_settings
from .db import get_actor_context, get_conn, get_tenant_context
from .phi import ensure_phi_processor, redact_text


def get_openai_client(*, timeout_seconds: int | None = None) -> OpenAI:
    """
    Create an OpenAI client with HIPAA/PHI policy enforcement.

    - In HIPAA mode, requires OpenAI to be listed in `PHI_PROCESSORS`.
    - Uses the configured request timeout.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured.")
    if settings.hipaa_mode:
        ensure_phi_processor("openai")
    timeout = timeout_seconds if timeout_seconds is not None else settings.openai_timeout_seconds
    return OpenAI(api_key=settings.openai_api_key, timeout=timeout)


def redact_if_enabled(text: str) -> str:
    settings = get_settings()
    if not settings.phi_redaction_enabled:
        return text
    return redact_text(text)


def redact_messages_if_enabled(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Redact PHI from chat messages while preserving non-text payloads (e.g., vision image_url parts).
    """
    settings = get_settings()
    if not settings.phi_redaction_enabled:
        return messages

    redacted_messages: list[dict[str, Any]] = []
    for message in messages:
        new_message = dict(message)
        content = new_message.get("content")
        if isinstance(content, str):
            new_message["content"] = redact_text(content)
        elif isinstance(content, list):
            new_parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    new_part = dict(part)
                    new_part["text"] = redact_text(part["text"])
                    new_parts.append(new_part)
                else:
                    new_parts.append(part)
            new_message["content"] = new_parts
        redacted_messages.append(new_message)

    return redacted_messages


def _as_processor_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {item.strip().lower() for item in raw.split(",") if item.strip()}
    if isinstance(raw, list):
        return {str(item).strip().lower() for item in raw if str(item).strip()}
    return set()


def _load_tenant_phi_policy(tenant_id: str | None) -> dict[str, Any] | None:
    if not tenant_id:
        return None
    try:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT allow_ai_processing, allowed_processors, require_redaction
                FROM tenant_phi_policies
                WHERE tenant_id = %s
                """,
                (tenant_id,),
            ).fetchone()
        return row if row else None
    except Exception:
        return None


def _enforce_phi_policy(*, processor: str, operation: str, tenant_id: str | None) -> None:
    settings = get_settings()
    if settings.hipaa_mode and not tenant_id:
        raise RuntimeError(f"Tenant context is required for PHI egress operation '{operation}' in HIPAA mode.")

    ensure_phi_processor(processor)

    policy = _load_tenant_phi_policy(tenant_id)
    if not policy:
        return

    if not bool(policy.get("allow_ai_processing", True)):
        raise RuntimeError(f"Tenant policy denies AI processing for operation '{operation}'.")

    allowed = _as_processor_set(policy.get("allowed_processors"))
    if allowed and processor.lower() not in allowed:
        raise RuntimeError(f"Processor '{processor}' is not allowed by tenant PHI policy.")

    require_redaction = bool(policy.get("require_redaction", False))
    if require_redaction and not settings.phi_redaction_enabled:
        raise RuntimeError(
            "Tenant PHI policy requires redaction, but PHI_REDACTION_ENABLED is disabled."
        )


def _log_phi_egress(
    *,
    tenant_id: str | None,
    actor_id: str | None,
    processor: str,
    operation: str,
    model: str | None,
    request_id: str | None,
    allowed: bool,
    redaction_applied: bool,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO phi_egress_events (
                    tenant_id,
                    actor_id,
                    processor,
                    operation,
                    model,
                    request_id,
                    allowed,
                    redaction_applied,
                    reason,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    tenant_id,
                    actor_id,
                    processor,
                    operation,
                    model,
                    request_id,
                    allowed,
                    redaction_applied,
                    reason,
                    Json(metadata or {}),
                ),
            )
            conn.commit()
    except Exception:
        return


def create_chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    operation: str = "chat.completions.create",
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
):
    settings = get_settings()
    target_model = model or settings.openai_model
    tenant_id = get_tenant_context()
    actor_id = get_actor_context()
    try:
        _enforce_phi_policy(processor="openai", operation=operation, tenant_id=tenant_id)
    except Exception as exc:
        _log_phi_egress(
            tenant_id=tenant_id,
            actor_id=actor_id,
            processor="openai",
            operation=operation,
            model=target_model,
            request_id=request_id,
            allowed=False,
            redaction_applied=False,
            reason=str(exc),
            metadata=metadata or {},
        )
        raise

    safe_messages = redact_messages_if_enabled(messages)
    redaction_applied = safe_messages != messages
    start = monotonic()
    response = get_openai_client().chat.completions.create(
        model=target_model,
        messages=safe_messages,
        **kwargs,
    )
    duration_ms = int((monotonic() - start) * 1000)
    usage = getattr(response, "usage", None)
    usage_meta = {
        "duration_ms": duration_ms,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    merged_meta = dict(metadata or {})
    merged_meta.update({k: v for k, v in usage_meta.items() if v is not None})
    _log_phi_egress(
        tenant_id=tenant_id,
        actor_id=actor_id,
        processor="openai",
        operation=operation,
        model=getattr(response, "model", target_model),
        request_id=request_id,
        allowed=True,
        redaction_applied=redaction_applied,
        metadata=merged_meta,
    )
    return response


def create_embedding(
    *,
    inputs: list[str] | str,
    model: str | None = None,
    operation: str = "embeddings.create",
    request_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
):
    settings = get_settings()
    target_model = model or settings.openai_embedding_model
    tenant_id = get_tenant_context()
    actor_id = get_actor_context()
    try:
        _enforce_phi_policy(processor="openai", operation=operation, tenant_id=tenant_id)
    except Exception as exc:
        _log_phi_egress(
            tenant_id=tenant_id,
            actor_id=actor_id,
            processor="openai",
            operation=operation,
            model=target_model,
            request_id=request_id,
            allowed=False,
            redaction_applied=False,
            reason=str(exc),
            metadata=metadata or {},
        )
        raise

    if isinstance(inputs, list):
        safe_inputs = [redact_if_enabled(text) for text in inputs]
    else:
        safe_inputs = redact_if_enabled(inputs)
    redaction_applied = safe_inputs != inputs
    start = monotonic()
    response = get_openai_client().embeddings.create(
        model=target_model,
        input=safe_inputs,
        **kwargs,
    )
    duration_ms = int((monotonic() - start) * 1000)
    usage = getattr(response, "usage", None)
    merged_meta = dict(metadata or {})
    merged_meta["duration_ms"] = duration_ms
    if usage:
        tokens = getattr(usage, "total_tokens", None)
        if tokens is not None:
            merged_meta["total_tokens"] = tokens
    _log_phi_egress(
        tenant_id=tenant_id,
        actor_id=actor_id,
        processor="openai",
        operation=operation,
        model=target_model,
        request_id=request_id,
        allowed=True,
        redaction_applied=redaction_applied,
        metadata=merged_meta,
    )
    return response
