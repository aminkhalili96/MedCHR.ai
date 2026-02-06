from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.app.config import get_settings
from backend.app.llm_gateway import create_chat_completion, create_embedding, redact_messages_if_enabled


def test_redact_messages_preserves_image_url(monkeypatch):
    # Enable PHI redaction for this test only.
    monkeypatch.setenv("PHI_REDACTION_ENABLED", "true")
    get_settings.cache_clear()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "SSN 123-45-6789"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "high"}},
            ],
        }
    ]

    redacted = redact_messages_if_enabled(messages)

    assert "123-45-6789" not in redacted[0]["content"][0]["text"]
    assert "[REDACTED_SSN]" in redacted[0]["content"][0]["text"]
    assert redacted[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"

    # Reset cached settings to avoid leaking env changes into other tests.
    get_settings.cache_clear()


def test_create_chat_completion_requires_tenant_context_in_hipaa_mode(monkeypatch):
    monkeypatch.setenv("HIPAA_MODE", "true")
    monkeypatch.setenv("PHI_PROCESSORS", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    with patch("backend.app.llm_gateway.get_tenant_context", return_value=None), patch(
        "backend.app.llm_gateway.get_actor_context", return_value="user-1"
    ), patch("backend.app.llm_gateway._log_phi_egress"), patch(
        "backend.app.llm_gateway.get_openai_client"
    ) as mock_client:
        with pytest.raises(RuntimeError, match="Tenant context is required"):
            create_chat_completion(messages=[{"role": "user", "content": "hello"}])
        mock_client.assert_not_called()

    get_settings.cache_clear()


def test_create_embedding_applies_redaction_and_calls_openai(monkeypatch):
    monkeypatch.setenv("HIPAA_MODE", "true")
    monkeypatch.setenv("PHI_PROCESSORS", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHI_REDACTION_ENABLED", "true")
    get_settings.cache_clear()

    fake_response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])], usage=SimpleNamespace(total_tokens=3))
    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=MagicMock(return_value=fake_response)))

    with patch("backend.app.llm_gateway.get_tenant_context", return_value="tenant-1"), patch(
        "backend.app.llm_gateway.get_actor_context", return_value="actor-1"
    ), patch(
        "backend.app.llm_gateway._load_tenant_phi_policy",
        return_value={"allow_ai_processing": True, "allowed_processors": ["openai"], "require_redaction": True},
    ), patch("backend.app.llm_gateway.get_openai_client", return_value=fake_client), patch(
        "backend.app.llm_gateway._log_phi_egress"
    ):
        create_embedding(inputs=["SSN 123-45-6789"])

    input_payload = fake_client.embeddings.create.call_args.kwargs["input"][0]
    assert "[REDACTED_SSN]" in input_payload
    assert "123-45-6789" not in input_payload

    get_settings.cache_clear()
