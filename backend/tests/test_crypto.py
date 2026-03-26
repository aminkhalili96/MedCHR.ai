import pytest

from backend.app.config import get_settings
from backend.app.crypto import decrypt_value, encrypt_value, DecryptionError


def test_encrypt_decrypt_roundtrip(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    ciphertext = encrypt_value("secret-value")
    plaintext = decrypt_value(ciphertext)

    assert plaintext == "secret-value"
    assert ciphertext != "secret-value"

    get_settings.cache_clear()


def test_decrypt_wrong_key_raises_strict(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "key-alpha")
    get_settings.cache_clear()
    ciphertext = encrypt_value("totp-secret")

    monkeypatch.setenv("APP_SECRET_KEY", "key-beta")
    get_settings.cache_clear()

    with pytest.raises(DecryptionError):
        decrypt_value(ciphertext, strict=True)

    get_settings.cache_clear()


def test_decrypt_wrong_key_returns_empty_non_strict(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "key-alpha")
    get_settings.cache_clear()
    ciphertext = encrypt_value("totp-secret")

    monkeypatch.setenv("APP_SECRET_KEY", "key-beta")
    get_settings.cache_clear()

    result = decrypt_value(ciphertext, strict=False)
    assert result == ""

    get_settings.cache_clear()


def test_decrypt_garbage_raises(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "test-key")
    get_settings.cache_clear()

    with pytest.raises(DecryptionError):
        decrypt_value("not-a-valid-fernet-token")

    get_settings.cache_clear()
