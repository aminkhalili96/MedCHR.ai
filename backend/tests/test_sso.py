import pytest

from backend.app.config import get_settings
from backend.app.sso import _enforce_sso_policy


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_sso_policy_requires_verified_email(monkeypatch):
    monkeypatch.setenv("SSO_REQUIRE_VERIFIED_EMAIL", "true")
    with pytest.raises(ValueError, match="not verified"):
        _enforce_sso_policy(
            provider="oidc",
            user_info={"email": "user@example.com", "email_verified": False},
        )


def test_sso_policy_enforces_allowed_domains(monkeypatch):
    monkeypatch.setenv("SSO_ALLOWED_DOMAINS", "hospital.org")
    with pytest.raises(ValueError, match="domain"):
        _enforce_sso_policy(
            provider="oidc",
            user_info={"email": "user@example.com", "email_verified": True},
        )
