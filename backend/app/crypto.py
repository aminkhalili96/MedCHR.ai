from __future__ import annotations

import base64
import hashlib
import logging

from cryptography.fernet import Fernet, InvalidToken

from .config import get_settings

_logger = logging.getLogger(__name__)


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


class DecryptionError(Exception):
    """Raised when decryption fails — indicates key mismatch or data corruption."""


def decrypt_value(value: str, *, strict: bool = True) -> str:
    try:
        raw = _fernet().decrypt(value.encode("utf-8"))
    except InvalidToken:
        _logger.critical(
            "Decryption failed — encryption key may have changed or data is corrupt. "
            "MFA secrets cannot be read. Investigate immediately."
        )
        if strict:
            raise DecryptionError(
                "Failed to decrypt value. The encryption key may have changed."
            )
        return ""
    return raw.decode("utf-8")
