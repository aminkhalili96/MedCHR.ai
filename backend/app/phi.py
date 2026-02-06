import re
from typing import Any

from .config import get_settings

_REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"(?i)\b(mrn|medical record number)[:\s]*[a-z0-9\-]+\b"), "[REDACTED_MRN]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(\+?\d{1,2}\s*)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"(?i)\b(dob|date of birth)[:\s\-]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[REDACTED_DOB]"),
]


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def ensure_phi_processor(processor: str) -> None:
    settings = get_settings()
    if not settings.hipaa_mode:
        return
    allowed = _split_csv(settings.phi_processors)
    if processor.lower() not in allowed:
        raise RuntimeError(
            f"PHI processor '{processor}' not approved. Add it to PHI_PROCESSORS or disable HIPAA mode."
        )


def redact_text(text: str) -> str:
    settings = get_settings()
    if not settings.phi_redaction_enabled or not text:
        return text
    redacted = text
    for pattern, replacement in _REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_payload(payload: Any) -> Any:
    settings = get_settings()
    if not settings.phi_redaction_enabled:
        return payload
    if isinstance(payload, dict):
        return {key: redact_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, str):
        return redact_text(payload)
    return payload
