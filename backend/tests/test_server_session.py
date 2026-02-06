from types import SimpleNamespace

from backend.app.server_session import ServerSideSessionMiddleware, _normalize_uuid, renew_session


def test_normalize_uuid_accepts_valid_uuid():
    value = _normalize_uuid("00000000-0000-0000-0000-000000000001")
    assert value == "00000000-0000-0000-0000-000000000001"


def test_normalize_uuid_rejects_invalid_uuid():
    assert _normalize_uuid("not-a-uuid") is None


def test_renew_session_sets_internal_flag():
    request = SimpleNamespace(session={})
    renew_session(request)
    assert request.session.get("_renew") is True


def test_sign_and_unsign_session_id_roundtrip():
    middleware = ServerSideSessionMiddleware(
        app=lambda scope, receive, send: None,
        secret_key="test-secret",
        session_cookie="medchr_session",
        max_age=3600,
        same_site="lax",
        https_only=False,
    )
    session_id = "00000000-0000-0000-0000-000000000001"
    signed = middleware._sign_session_id(session_id)
    assert middleware._unsign_session_id(signed) == session_id

