import asyncio

from starlette.requests import Request
from starlette.responses import Response

from backend.app.main import security_headers_middleware


def _directive_value(csp: str, directive: str) -> str:
    for part in csp.split(";"):
        token = part.strip()
        if token.startswith(f"{directive} "):
            return token
    return ""


def test_csp_uses_nonce_for_scripts_without_unsafe_inline():
    async def _run():
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "GET",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
            "scheme": "http",
        }
        request = Request(scope)
        async def call_next(_: Request) -> Response:
            return Response("ok")
        return await security_headers_middleware(request, call_next)

    response = asyncio.run(_run())
    csp = response.headers.get("Content-Security-Policy", "")
    assert csp

    script_src = _directive_value(csp, "script-src")
    assert script_src
    assert "'nonce-" in script_src
    assert "'unsafe-inline'" not in script_src
