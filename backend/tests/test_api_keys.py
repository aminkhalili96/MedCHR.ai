from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.app.config import get_settings
from backend.app.security import (
    ApiKeyMatch,
    _get_db_key_match,
    hash_api_key,
    require_api_key,
    require_read_scope,
    require_write_scope,
    validate_production_settings,
)


class DummyRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self.state = SimpleNamespace()


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_hash_api_key_is_deterministic(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "unit-test-secret")
    first = hash_api_key("abc123")
    second = hash_api_key("abc123")
    third = hash_api_key("different")

    assert first.startswith("hmac_sha256:")
    assert first == second
    assert first != third


def test_require_api_key_accepts_scoped_env_key(monkeypatch):
    tenant_id = "00000000-0000-0000-0000-000000000111"
    monkeypatch.setenv("API_KEYS", f"{tenant_id}:test-key")
    request = DummyRequest(headers={"x-api-key": "test-key"})

    value = require_api_key(request)

    assert value == "test-key"
    assert request.state.tenant_id == tenant_id
    assert request.state.actor.startswith("api_key_env:")


def test_get_db_key_match_updates_usage(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "unit-test-secret")
    key = "live-key"
    expected_hash = hash_api_key(key)

    with patch("backend.app.db.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = {
            "id": "db-key-id",
            "tenant_id": "00000000-0000-0000-0000-000000000222",
            "name": "ingestion-service",
            "scopes": ["read", "write"],
            "rate_limit": None,
        }

        match = _get_db_key_match(key)

        assert match is not None
        assert match.tenant_id == "00000000-0000-0000-0000-000000000222"
        assert match.actor == "api_key_db:ingestion-service"
        assert match.scopes == {"read", "write"}
        lookup_params = mock_conn.execute.call_args_list[0][0][1]
        assert lookup_params == (expected_hash,)
        assert mock_conn.commit.called


def test_require_api_key_accepts_db_key_when_configured(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    request = DummyRequest(headers={"x-api-key": "db-key"})

    with patch("backend.app.security._has_active_db_api_keys", return_value=True), patch(
        "backend.app.security._get_db_key_match",
        return_value=ApiKeyMatch(
            tenant_id="00000000-0000-0000-0000-000000000333",
            actor="api_key_db:test",
            scopes={"read", "write"},
        ),
    ):
        value = require_api_key(request)

    assert value == "db-key"
    assert request.state.tenant_id == "00000000-0000-0000-0000-000000000333"
    assert request.state.actor == "api_key_db:test"
    assert request.state.api_key_scopes == {"read", "write"}


def test_write_scope_requires_write_permission():
    request = DummyRequest()
    request.state.api_key_scopes = {"read"}

    with pytest.raises(HTTPException):
        require_write_scope(request)


def test_read_scope_allows_write_keys():
    request = DummyRequest()
    request.state.api_key_scopes = {"write"}

    require_read_scope(request)


def test_validate_production_settings_allows_db_keys(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("APP_SECRET_KEY", "prod-secret")
    monkeypatch.delenv("API_KEYS", raising=False)

    with patch("backend.app.security._has_active_db_api_keys", return_value=True):
        validate_production_settings()


def test_validate_production_settings_requires_any_key_source(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("APP_SECRET_KEY", "prod-secret")
    monkeypatch.delenv("API_KEYS", raising=False)

    with patch("backend.app.security._has_active_db_api_keys", return_value=False):
        with pytest.raises(RuntimeError, match="No API keys configured"):
            validate_production_settings()


def test_get_db_key_match_rate_limit_logs_and_blocks(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "unit-test-secret")
    key = "rate-limited-key"

    with patch("backend.app.db.get_conn") as mock_get_conn, patch(
        "backend.app.audit_events.append_audit_event"
    ) as mock_append_audit:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = [
            MagicMock(
                fetchone=MagicMock(
                    return_value={
                        "id": "00000000-0000-0000-0000-000000000999",
                        "tenant_id": "00000000-0000-0000-0000-000000000222",
                        "name": "ingestion-service",
                        "scopes": ["read"],
                        "rate_limit": 1,
                    }
                )
            ),
            MagicMock(fetchone=MagicMock(return_value={"request_count": 2})),
            MagicMock(),
        ]

        with pytest.raises(HTTPException) as exc:
            _get_db_key_match(key)

    assert exc.value.status_code == 429
    assert mock_append_audit.called
