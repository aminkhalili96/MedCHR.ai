import hashlib
import hmac
import secrets
from dataclasses import dataclass
from uuid import UUID

import bleach
import markdown as md
from fastapi import HTTPException, Request, status
from passlib.context import CryptContext

from .config import get_settings
from .db import get_conn, set_actor_context, set_tenant_context
from .ip_whitelist import check_tenant_ip_access, extract_client_ip

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_ALLOWED_TAGS = [
    "a",
    "abbr",
    "b",
    "blockquote",
    "br",
    "code",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "i",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "ul",
    "span",
]
_ALLOWED_ATTRS = {
    "a": ["href", "title", "rel", "target", "class"],
    "span": ["class"],
    "th": ["colspan", "rowspan"],
    "td": ["colspan", "rowspan"],
}
_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class ApiKeyConfig:
    tenant_id: str | None
    key: str


@dataclass(frozen=True)
class ApiKeyMatch:
    tenant_id: str | None
    actor: str
    scopes: set[str]


def _parse_api_keys(value: str | None) -> list[ApiKeyConfig]:
    """
    Supports:
    - legacy:  API_KEYS="key1,key2"
    - scoped:  API_KEYS="tenant_uuid:key1,tenant_uuid:key2"
    """
    items = _split_csv(value)
    parsed: list[ApiKeyConfig] = []
    for item in items:
        if ":" in item:
            tenant_id, key = item.split(":", 1)
            tenant_id = tenant_id.strip()
            key = key.strip()
            if tenant_id and key:
                parsed.append(ApiKeyConfig(tenant_id=tenant_id, key=key))
        else:
            parsed.append(ApiKeyConfig(tenant_id=None, key=item))
    return parsed


def _normalize_scopes(raw) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {item.strip().lower() for item in raw.split(",") if item.strip()}
    if isinstance(raw, list):
        return {str(item).strip().lower() for item in raw if str(item).strip()}
    return set()


def _scope_implies(scopes: set[str], required: str) -> bool:
    if "*" in scopes or "admin" in scopes:
        return True
    if required == "read":
        return "read" in scopes or "write" in scopes
    if required == "write":
        return "write" in scopes
    return required in scopes


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def _constant_time_compare(value: str, other: str) -> bool:
    return hmac.compare_digest(value.encode("utf-8"), other.encode("utf-8"))


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for storage/lookup in the `api_keys` table.

    Uses HMAC-SHA256 keyed by APP_SECRET_KEY to avoid storing plaintext keys.
    """
    settings = get_settings()
    digest = hmac.new(
        settings.app_secret_key.encode("utf-8"),
        api_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac_sha256:{digest}"


def _get_env_key_match(provided: str, api_keys: list[ApiKeyConfig]) -> ApiKeyMatch | None:
    match = next((k for k in api_keys if _constant_time_compare(provided, k.key)), None)
    if not match:
        return None
    return ApiKeyMatch(
        tenant_id=match.tenant_id,
        actor=f"api_key_env:{provided[:6]}",
        scopes={"read", "write", "admin"},
    )


def _has_active_db_api_keys() -> bool:
    from .db import get_conn

    with get_conn() as conn:
        row = conn.execute(
            "SELECT EXISTS(SELECT 1 FROM api_keys WHERE revoked_at IS NULL) AS has_keys"
        ).fetchone()
    return bool(row and row.get("has_keys"))


def _get_db_key_match(provided: str) -> ApiKeyMatch | None:
    from .db import get_conn

    lookup_hash = hash_api_key(provided)
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, tenant_id, name, scopes, rate_limit
            FROM api_keys
            WHERE key_hash = %s
              AND revoked_at IS NULL
            LIMIT 1
            """,
            (lookup_hash,),
        ).fetchone()
        if not row:
            return None
        actor_label = row.get("name") or str(row["id"])[:8]
        scopes = _normalize_scopes(row.get("scopes")) or {"read"}
        rate_limit = row.get("rate_limit")
        if rate_limit:
            window_row = conn.execute(
                """
                INSERT INTO api_key_rate_windows (api_key_id, window_start, request_count)
                VALUES (%s, date_trunc('minute', NOW()), 1)
                ON CONFLICT (api_key_id, window_start)
                DO UPDATE SET request_count = api_key_rate_windows.request_count + 1
                RETURNING request_count
                """,
                (row["id"],),
            ).fetchone()
            count = int(window_row.get("request_count", 0)) if window_row else 0
            if count > int(rate_limit):
                try:
                    from .audit_events import append_audit_event

                    append_audit_event(
                        conn,
                        action="auth.api_key_rate_limited",
                        resource_type="system",
                        resource_id=str(row["id"]),
                        outcome="DENIED",
                        tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
                        actor=f"api_key_db:{actor_label}",
                        details={
                            "api_key_id": str(row["id"]),
                            "rate_limit": int(rate_limit),
                            "window_count": count,
                        },
                    )
                except Exception:
                    pass
                conn.execute(
                    """
                    UPDATE api_keys
                    SET last_used_at = NOW(),
                        request_count = request_count + 1
                    WHERE id = %s
                    """,
                    (row["id"],),
                )
                conn.commit()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="API key rate limit exceeded",
                )
        conn.execute(
            """
            UPDATE api_keys
            SET last_used_at = NOW(),
                request_count = request_count + 1
            WHERE id = %s
            """,
            (row["id"],),
        )
        conn.commit()

    return ApiKeyMatch(
        tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
        actor=f"api_key_db:{actor_label}",
        scopes=scopes,
    )


def require_api_key(request: Request) -> str | None:
    settings = get_settings()
    api_keys = _parse_api_keys(settings.api_keys)

    db_keys_configured = False
    if not api_keys:
        try:
            db_keys_configured = _has_active_db_api_keys()
        except Exception:
            db_keys_configured = False
    if not api_keys and not db_keys_configured and settings.app_env != "prod" and not settings.hipaa_mode:
        request.state.api_key_scopes = {"read", "write", "admin"}
        return None

    provided = request.headers.get("x-api-key")
    if not provided:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            provided = auth_header.split(None, 1)[1].strip()

    if not provided:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    match = _get_env_key_match(provided, api_keys)
    if not match:
        try:
            match = _get_db_key_match(provided)
        except HTTPException:
            raise
        except Exception:
            match = None
    if not match:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    request.state.actor = match.actor
    request.state.tenant_id = match.tenant_id
    request.state.api_key_scopes = set(match.scopes)
    set_actor_context(match.actor)
    set_tenant_context(match.tenant_id)
    if match.tenant_id:
        client_ip = extract_client_ip(request)
        try:
            with get_conn() as conn:
                access = check_tenant_ip_access(match.tenant_id, client_ip, conn)
            if not access.get("allowed", False):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied from this IP address",
                )
        except HTTPException:
            raise
        except Exception:
            if settings.app_env == "prod" or settings.hipaa_mode:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="IP allowlist validation failed",
                )

    if (settings.app_env == "prod" or settings.hipaa_mode) and not match.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is not tenant-scoped",
        )
    return provided


def _require_scope(request: Request, required_scope: str) -> None:
    scopes = getattr(request.state, "api_key_scopes", set())
    if not scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing API key scope: {required_scope}",
        )
    if not _scope_implies(set(scopes), required_scope):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing API key scope: {required_scope}",
        )


def require_read_scope(request: Request) -> None:
    _require_scope(request, "read")


def require_write_scope(request: Request) -> None:
    _require_scope(request, "write")


def get_csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token


def validate_csrf_token(request: Request, token: str | None) -> None:
    settings = get_settings()
    if not settings.csrf_enabled:
        return
    expected = request.session.get("csrf_token")
    if not expected or not token or not _constant_time_compare(token, expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


def render_markdown(text: str) -> str:
    if not text:
        return ""
    html = md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html",
    )
    cleaned = bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,
    )
    return bleach.linkify(
        cleaned,
        callbacks=[bleach.callbacks.nofollow, bleach.callbacks.target_blank],
        skip_tags=["pre", "code"],
    )


def validate_production_settings() -> None:
    settings = get_settings()
    if settings.app_env != "prod" and not settings.hipaa_mode:
        return

    if not settings.app_secret_key or settings.app_secret_key == "dev-secret":
        raise RuntimeError("APP_SECRET_KEY must be set to a strong value for production/HIPAA mode.")

    parsed = _parse_api_keys(settings.api_keys)
    if parsed and any(k.tenant_id is None for k in parsed):
        raise RuntimeError("API_KEYS must be tenant-scoped for production/HIPAA mode (tenant_uuid:key).")

    if parsed:
        return

    try:
        if _has_active_db_api_keys():
            return
    except Exception as exc:
        raise RuntimeError("Failed to verify DB-backed API keys for production/HIPAA mode.") from exc

    raise RuntimeError(
        "No API keys configured for production/HIPAA mode. Set tenant-scoped API_KEYS or provision active rows in api_keys."
    )


def get_app_password_hash() -> str | None:
    settings = get_settings()
    if settings.app_password_hash:
        return settings.app_password_hash
    if settings.app_password:
        return hash_password(settings.app_password)
    return None


def allowed_hosts() -> list[str]:
    settings = get_settings()
    return _split_csv(settings.allowed_hosts)


def cors_origins() -> list[str]:
    settings = get_settings()
    return _split_csv(settings.cors_origins)
