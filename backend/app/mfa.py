from __future__ import annotations

from datetime import datetime, timezone

from .db import get_conn
from .crypto import decrypt_value, encrypt_value


def create_mfa_setup_token(user_id: str, tenant_id: str, secret: str, *, ttl_minutes: int = 10) -> str:
    """
    Create a short-lived MFA setup token and store the TOTP secret server-side.

    Returns:
        token id (UUID as string)
    """
    if not secret:
        raise ValueError("secret is required")

    encrypted_secret = encrypt_value(secret)
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM mfa_setup_tokens WHERE user_id = %s AND used_at IS NULL",
            (user_id,),
        )
        row = conn.execute(
            """
            INSERT INTO mfa_setup_tokens (user_id, tenant_id, secret, expires_at)
            VALUES (%s, %s, %s, NOW() + make_interval(mins => %s))
            RETURNING id
            """,
            (user_id, tenant_id, encrypted_secret, int(ttl_minutes)),
        ).fetchone()
        conn.commit()
    return str(row["id"])


def get_mfa_setup_secret(token_id: str, user_id: str) -> str | None:
    """
    Fetch the MFA setup secret for the given user/token if it's still valid.
    """
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT secret, expires_at, used_at
            FROM mfa_setup_tokens
            WHERE id = %s AND user_id = %s
            """,
            (token_id, user_id),
        ).fetchone()
    if not row:
        return None
    if row.get("used_at") is not None:
        return None

    expires_at = row.get("expires_at")
    if expires_at is not None:
        now = datetime.now(timezone.utc)
        try:
            if expires_at < now:
                return None
        except TypeError:
            # If tz-naive, compare naive.
            if expires_at < now.replace(tzinfo=None):
                return None

    stored_secret = row.get("secret")
    if not stored_secret:
        return None
    decrypted = decrypt_value(str(stored_secret))
    if decrypted:
        return decrypted
    # Backward compatibility for any plaintext legacy records.
    return str(stored_secret)


def consume_mfa_setup_token(token_id: str, user_id: str) -> None:
    """
    Mark an MFA setup token as used to prevent replay.
    """
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE mfa_setup_tokens
            SET used_at = NOW()
            WHERE id = %s AND user_id = %s AND used_at IS NULL
            """,
            (token_id, user_id),
        )
        conn.commit()


def get_mfa_lockout_remaining(user_id: str, flow: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT lockout_until
            FROM mfa_lockouts
            WHERE user_id = %s AND flow = %s
            """,
            (user_id, flow),
        ).fetchone()
    if not row:
        return 0
    lockout_until = row.get("lockout_until")
    if not lockout_until:
        return 0
    now = datetime.now(timezone.utc)
    try:
        delta = int((lockout_until - now).total_seconds())
    except TypeError:
        delta = int((lockout_until - now.replace(tzinfo=None)).total_seconds())
    return max(0, delta)


def consume_mfa_lockout_expiry(user_id: str, flow: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """
            UPDATE mfa_lockouts
            SET failed_attempts = 0,
                lockout_until = NULL,
                updated_at = NOW()
            WHERE user_id = %s
              AND flow = %s
              AND lockout_until IS NOT NULL
              AND lockout_until <= NOW()
            RETURNING id
            """,
            (user_id, flow),
        ).fetchone()
        conn.commit()
    return bool(row)


def record_mfa_failure(user_id: str, flow: str, *, max_attempts: int = 5, lockout_seconds: int = 300) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO mfa_lockouts (user_id, flow, failed_attempts, last_failed_at, updated_at)
            VALUES (%s, %s, 1, NOW(), NOW())
            ON CONFLICT (user_id, flow)
            DO UPDATE
            SET failed_attempts = CASE
                    WHEN mfa_lockouts.lockout_until IS NOT NULL AND mfa_lockouts.lockout_until > NOW()
                      THEN mfa_lockouts.failed_attempts
                    ELSE mfa_lockouts.failed_attempts + 1
                END,
                last_failed_at = NOW(),
                updated_at = NOW()
            RETURNING failed_attempts, lockout_until
            """,
            (user_id, flow),
        ).fetchone()
        attempts = int(row.get("failed_attempts", 0)) if row else 0
        lockout_until = row.get("lockout_until") if row else None
        if lockout_until:
            conn.commit()
            return True
        if attempts >= max_attempts:
            conn.execute(
                """
                UPDATE mfa_lockouts
                SET lockout_until = NOW() + make_interval(secs => %s),
                    failed_attempts = 0,
                    updated_at = NOW()
                WHERE user_id = %s AND flow = %s
                """,
                (int(lockout_seconds), user_id, flow),
            )
            conn.commit()
            return True
        conn.commit()
        return False


def clear_mfa_failures(user_id: str, flow: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE mfa_lockouts
            SET failed_attempts = 0,
                lockout_until = NULL,
                updated_at = NOW()
            WHERE user_id = %s AND flow = %s
            """,
            (user_id, flow),
        )
        conn.commit()
