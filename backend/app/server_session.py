from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from itsdangerous import BadSignature, URLSafeSerializer
from psycopg.types.json import Json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from .db import get_conn


def _normalize_uuid(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(value))
    except (TypeError, ValueError):
        return None


def renew_session(request: Request) -> None:
    """
    Mark the current session for ID rotation on response commit.
    """
    request.session["_renew"] = True


class ServerSideSessionMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        secret_key: str,
        session_cookie: str = "session",
        max_age: int = 14 * 24 * 60 * 60,
        same_site: str = "lax",
        https_only: bool = False,
        path: str = "/",
    ) -> None:
        super().__init__(app)
        self.session_cookie = session_cookie
        self.max_age = max_age
        self.same_site = same_site
        self.https_only = https_only
        self.path = path
        self.signer = URLSafeSerializer(secret_key, salt="medchr-ui-session")

    def _sign_session_id(self, session_id: str) -> str:
        return self.signer.dumps(session_id)

    def _unsign_session_id(self, value: str | None) -> str | None:
        if not value:
            return None
        try:
            raw = self.signer.loads(value)
        except BadSignature:
            return None
        if not isinstance(raw, str):
            return None
        return _normalize_uuid(raw)

    def _load_session_data(self, session_id: str, now: datetime) -> dict:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT data, expires_at
                FROM ui_sessions
                WHERE id = %s
                  AND revoked_at IS NULL
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return {}
        expires_at = row.get("expires_at")
        if expires_at and expires_at < now:
            self._revoke_session(session_id)
            return {}
        data = row.get("data") or {}
        return data if isinstance(data, dict) else {}

    def _persist_session(self, session_id: str, session_data: dict, expires_at: datetime) -> None:
        user_id = _normalize_uuid(session_data.get("user_id"))
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ui_sessions (id, user_id, data, expires_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET user_id = EXCLUDED.user_id,
                    data = EXCLUDED.data,
                    expires_at = EXCLUDED.expires_at,
                    updated_at = NOW(),
                    revoked_at = NULL
                """,
                (session_id, user_id, Json(session_data), expires_at),
            )
            conn.commit()

    def _revoke_session(self, session_id: str) -> None:
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE ui_sessions
                SET revoked_at = NOW(), updated_at = NOW()
                WHERE id = %s
                  AND revoked_at IS NULL
                """,
                (session_id,),
            )
            conn.commit()

    async def dispatch(self, request: Request, call_next):
        now = datetime.now(timezone.utc)
        cookie_value = request.cookies.get(self.session_cookie)
        existing_session_id = self._unsign_session_id(cookie_value)
        session_data: dict = {}
        if existing_session_id:
            try:
                session_data = self._load_session_data(existing_session_id, now)
                if not session_data:
                    existing_session_id = None
            except Exception:
                existing_session_id = None
                session_data = {}

        request.scope["session"] = session_data
        response = await call_next(request)

        session = request.scope.get("session") or {}
        should_renew = bool(session.pop("_renew", False))

        if session:
            session_id = existing_session_id if (existing_session_id and not should_renew) else str(uuid4())
            expires_at = now + timedelta(seconds=self.max_age)
            self._persist_session(session_id, session, expires_at)
            response.set_cookie(
                key=self.session_cookie,
                value=self._sign_session_id(session_id),
                max_age=self.max_age,
                path=self.path,
                secure=self.https_only,
                httponly=True,
                samesite=self.same_site,
            )
        else:
            if existing_session_id:
                self._revoke_session(existing_session_id)
            response.delete_cookie(self.session_cookie, path=self.path)

        return response

