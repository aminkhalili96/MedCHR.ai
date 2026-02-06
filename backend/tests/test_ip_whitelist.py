from types import SimpleNamespace

import pytest

from backend.app.config import get_settings
from backend.app.ip_whitelist import extract_client_ip


class DummyRequest:
    def __init__(self, headers: dict[str, str], remote_ip: str):
        self.headers = headers
        self.client = SimpleNamespace(host=remote_ip)


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_extract_client_ip_ignores_forwarded_headers_when_proxy_not_trusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1")
    request = DummyRequest(
        headers={"x-forwarded-for": "203.0.113.9"},
        remote_ip="198.51.100.10",
    )
    assert extract_client_ip(request) == "198.51.100.10"


def test_extract_client_ip_uses_forwarded_header_for_trusted_proxy(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "10.0.0.1,10.0.0.0/24")
    request = DummyRequest(
        headers={"x-forwarded-for": "203.0.113.9, 10.0.0.1"},
        remote_ip="10.0.0.1",
    )
    assert extract_client_ip(request) == "203.0.113.9"
