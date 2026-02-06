from backend.app.config import get_settings
from backend.app.crypto import decrypt_value, encrypt_value


def test_encrypt_decrypt_roundtrip(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", "unit-test-secret")
    get_settings.cache_clear()

    ciphertext = encrypt_value("secret-value")
    plaintext = decrypt_value(ciphertext)

    assert plaintext == "secret-value"
    assert ciphertext != "secret-value"

    get_settings.cache_clear()
