from __future__ import annotations

import json
import logging

from .config import get_settings
from .llm_gateway import create_chat_completion, redact_if_enabled

logger = logging.getLogger(__name__)


class AuditorAgent:
    """
    An AI agent responsible for verifying claims in generated text against source documents.
    """

    def __init__(self):
        settings = get_settings()
        self.model = settings.openai_model

    def verify_draft(self, draft_text: str, source_chunks: list[dict]) -> dict:
        """
        Verify that the draft text is strictly supported by the provided source chunks.

        Returns:
            dict with:
            - is_verified (bool)
            - issues (list[str])
        """
        sources_text = "\n\n".join(
            [
                f"Source [{c.get('document_id', 'unknown')}]: {c.get('chunk_text', '')}"
                for c in (source_chunks or [])
            ]
        )

        system_prompt = (
            "You are a Clinical Auditor AI. Your goal is to verify that a "
            "Client Health Report (CHR) draft is strictly supported by the provided source text.\n"
            "Rules:\n"
            "1. Every medical claim (diagnosis, medication, value) must be present in the Source.\n"
            "2. If a claim is unsupported or contradicts the source, flag it as a Hallucination.\n"
            "3. If the draft is fully supported, return status 'VERIFIED'.\n"
            "4. Return JSON only: {'status': 'VERIFIED' | 'FAILED', 'issues': ['explanation 1', ...]}"
        )

        user_prompt = f"SOURCES:\n{redact_if_enabled(sources_text)}\n\nDRAFT:\n{redact_if_enabled(draft_text)}"

        try:
            response = create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            result = json.loads(content) if content else {}
            return {
                "is_verified": result.get("status") == "VERIFIED",
                "issues": result.get("issues", []) or [],
            }
        except Exception as exc:
            logger.warning("Auditor failed", exc_info=exc)
            return {
                "is_verified": False,
                "issues": [f"Auditor failed to execute: {str(exc)}"],
            }


def audit_submission(draft_text: str, patient_id: str, context_chunks: list[dict]) -> dict:
    """
    Run the auditor. Keep failures non-fatal (flag for human review).
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return {"is_verified": False, "issues": ["Auditor unavailable: OPENAI_API_KEY not configured."]}

    auditor = AuditorAgent()
    return auditor.verify_draft(draft_text, context_chunks or [])
