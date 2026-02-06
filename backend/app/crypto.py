from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from .config import get_settings


def _fernet_key() -> bytes:
    settings = get_settings()
    if settings.mfa_secret_key:
        candidate = settings.mfa_secret_key.strip()
        if candidate:
            return candidate.encode("utf-8")
    digest = hashlib.sha256(settings.app_secret_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet() -> Fernet:
    return Fernet(_fernet_key())


def encrypt_value(value: str) -> str:
    token = _fernet().encrypt(value.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_value(value: str) -> str:
    try:
        raw = _fernet().decrypt(value.encode("utf-8"))
    except InvalidToken:
        return ""
    return raw.decode("utf-8")
