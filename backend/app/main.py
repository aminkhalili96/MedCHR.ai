from pathlib import Path
from uuid import uuid4
import logging
import re
import secrets
from urllib.parse import quote

import math

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from psycopg.types.json import Json
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .config import get_settings
from .db import get_conn, close_pool, clear_tenant_context, set_tenant_context, set_actor_context
from .schemas import (
    PatientCreate,
    Patient,
    Document,
)
from .storage import (
    download_bytes,
    ensure_bucket,
    storage_health,
    create_signed_download_url,
)
from .rag import build_query, retrieve_top_chunks
from .chr import query_chr
from .jobs import enqueue_job, list_jobs
from .observability import metrics, record_request
from . import clinical
from . import gap_features
from .api_routes import router as api_router
from .helpers import (
    _row_to_patient,
    _row_to_document,
    _log_action,
    _upload_document,
    _extract_document,
    _embed_document,
    _draft_chr,
    _aggregate_structured,
)
from .security import (
    get_csrf_token,
    validate_csrf_token,
    render_markdown,
    validate_production_settings,
    allowed_hosts,
    cors_origins,
)
from .auth import authenticate_user, get_current_user, User, get_password_hash
from .audit_events import append_audit_event
from .authz import (
    require_permission,
    mark_step_up_verified,
    is_step_up_verified,
)
from .phi import ensure_phi_processor
from .server_session import ServerSideSessionMiddleware, renew_session
from .sso import configure_sso, handle_sso_callback, initiate_sso_login, provision_sso_user
from .ip_whitelist import get_tenant_whitelist, update_tenant_whitelist
from .alerts import check_critical_values, check_drug_interactions, check_allergy_contraindications, run_safety_checks
from .report_templates import list_templates

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR.parent.parent / "frontend" / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

settings = get_settings()
docs_enabled = settings.api_docs_enabled and settings.app_env != "prod" and not settings.hipaa_mode
app = FastAPI(
    title="MedCHR API",
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ---------------------------------------------------------------------------
# Custom error pages (self-contained HTML, no dependency on base.html)
# ---------------------------------------------------------------------------

def _error_page_html(status_code: int, title: str, message: str) -> str:
    """Return a self-contained HTML error page with MedCHR branding."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - MedCHR.ai</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #fdf6f1;
    color: #3a2e2a;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .container {{
    text-align: center;
    padding: 2rem;
    max-width: 480px;
  }}
  .logo {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #d97756;
    margin-bottom: 2rem;
    letter-spacing: -0.02em;
  }}
  .logo span {{
    color: #3a2e2a;
  }}
  .code {{
    font-size: 5rem;
    font-weight: 700;
    color: #d97756;
    line-height: 1;
    margin-bottom: 0.5rem;
  }}
  h1 {{
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }}
  p {{
    color: #6b5b54;
    line-height: 1.6;
    margin-bottom: 2rem;
  }}
  a.btn {{
    display: inline-block;
    background: #d97756;
    color: #fff;
    text-decoration: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: background 0.2s;
  }}
  a.btn:hover {{
    background: #c4623f;
  }}
</style>
</head>
<body>
  <div class="container">
    <div class="logo">Med<span>CHR</span>.ai</div>
    <div class="code">{status_code}</div>
    <h1>{title}</h1>
    <p>{message}</p>
    <a class="btn" href="/ui">Back to Dashboard</a>
  </div>
</body>
</html>"""


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Render branded HTML error pages for browser requests; JSON for API clients."""
    accept = request.headers.get("accept", "")
    is_browser = "text/html" in accept

    if not is_browser:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    if exc.status_code == 404:
        html = _error_page_html(
            404,
            "Page Not Found",
            "The page you are looking for does not exist or has been moved.",
        )
    elif exc.status_code == 500:
        html = _error_page_html(
            500,
            "Internal Server Error",
            "Something went wrong on our end. Please try again later or contact support if the problem persists.",
        )
    else:
        html = _error_page_html(
            exc.status_code,
            str(exc.detail) if exc.detail else "Error",
            "An unexpected error occurred.",
        )

    return HTMLResponse(content=html, status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions -- shows a branded 500 page."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        html = _error_page_html(
            500,
            "Internal Server Error",
            "Something went wrong on our end. Please try again later or contact support if the problem persists.",
        )
        return HTMLResponse(content=html, status_code=500)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


session_kwargs = {
    "secret_key": settings.app_secret_key,
    "session_cookie": settings.session_cookie_name,
    "max_age": settings.session_max_age_minutes * 60,
    "same_site": "strict" if settings.app_env == "prod" or settings.hipaa_mode else "lax",
    "https_only": settings.app_env == "prod" or settings.hipaa_mode,
}


hosts = allowed_hosts()
if hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

origins = cors_origins()
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

# Mount sub-routers under /api/v1 prefix for versioned API access,
# while keeping original paths for backward compatibility.
app.include_router(clinical.router)
app.include_router(clinical.router, prefix="/api/v1", include_in_schema=False)
app.include_router(gap_features.router)
app.include_router(gap_features.router, prefix="/api/v1", include_in_schema=False)
app.include_router(api_router)
app.include_router(api_router, prefix="/api/v1", include_in_schema=False)

app.mount("/static", StaticFiles(directory=str(BASE_DIR.parent.parent / "frontend" / "static")), name="static")


@app.middleware("http")
async def db_context_middleware(request: Request, call_next):
    clear_tenant_context()
    try:
        return await call_next(request)
    finally:
        clear_tenant_context()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    return response




@app.middleware("http")
async def session_timeout_middleware(request: Request, call_next):
    """
    Session management middleware. Auto-authenticates in non-prod environments.
    """
    import time
    
    # Session timeout policy applies to UI surface only.
    if not request.url.path.startswith("/ui"):
        return await call_next(request)
    
    session = request.session
    current_time = int(time.time())
    
    # Auto-login: inject admin user_id if no session exists (dev only)
    if settings.app_env == "dev" and not session.get("user_id"):
        try:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
                ).fetchone()
                if row:
                    session["user_id"] = str(row["id"])
        except Exception:
            pass
    
    # Update last activity timestamp
    session["_last_activity"] = current_time
    
    response = await call_next(request)
    return response


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    csp_nonce = secrets.token_urlsafe(16)
    request.state.csp_nonce = csp_nonce
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = settings.referrer_policy
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cache-Control"] = "no-store"

    # Swagger/ReDoc pages use inline scripts and CDN CSS that the strict
    # nonce-based CSP blocks.  Use a relaxed policy for the docs paths only.
    _docs_paths = ("/docs", "/redoc", "/openapi.json")
    if request.url.path in _docs_paths:
        csp = (
            "default-src 'self'; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
    else:
        csp = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            f"script-src 'self' 'nonce-{csp_nonce}' https://cdn.jsdelivr.net; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )

    response.headers["Content-Security-Policy"] = csp
    if settings.app_env == "prod" or settings.hipaa_mode:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    import time

    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path
    if path.startswith("/ui/patients/"):
        path = "/ui/patients/:id"
    elif path.startswith("/patients/"):
        path = "/patients/:id"
    elif path.startswith("/documents/"):
        path = "/documents/:id"
    record_request(request.method, path, response.status_code, duration)
    return response


@app.on_event("startup")
def startup() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    validate_production_settings()
    configure_sso()
    if settings.hipaa_mode:
        ensure_phi_processor("supabase")
        if settings.openai_api_key:
            ensure_phi_processor("openai")
    # Ensure the storage bucket exists for uploads
    try:
        ensure_bucket(settings.storage_bucket)
    except Exception as exc:
        if settings.app_env == "dev":
            logging.getLogger(__name__).warning(
                "Supabase storage unavailable — uploads will not work: %s", exc
            )
        else:
            raise


@app.on_event("shutdown")
def shutdown() -> None:
    close_pool()


@limiter.exempt
@app.get("/health")
def health():
    return {"status": "ok"}


@limiter.exempt
@app.get("/ready")
def ready():
    db_ok = False
    storage_ok = False
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    try:
        storage_ok = storage_health(settings.storage_bucket)
    except Exception:
        storage_ok = False
    status_code = status.HTTP_200_OK if db_ok and storage_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse({"status": "ok" if db_ok and storage_ok else "degraded", "db": db_ok, "storage": storage_ok}, status_code=status_code)


@limiter.exempt
@app.get("/metrics", include_in_schema=False)
def metrics_endpoint(request: Request):
    if settings.app_env == "prod" or settings.hipaa_mode:
        user = _require_ui_user(request)
        if not user or user.role != "admin":
            raise HTTPException(status_code=404, detail="Not found")
    body = metrics.export_prometheus()
    return Response(content=body, media_type="text/plain; version=0.0.4")


# -------------------- UI --------------------

def _require_ui_user(request: Request) -> User | None:
    try:
        return get_current_user(request)
    except Exception:
        pass
    # Auto-authenticate as first admin user in dev mode only
    if settings.app_env != "dev":
        return None
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT id, email, role, tenant_id, created_at, mfa_enabled, mfa_secret FROM users LIMIT 1"
            ).fetchone()
            if row:
                request.session["user_id"] = str(row["id"])
                user = User(
                    id=row["id"],
                    email=row["email"],
                    role=row["role"],
                    tenant_id=row["tenant_id"],
                    created_at=row["created_at"],
                    mfa_enabled=row.get("mfa_enabled", False),
                    mfa_secret=row.get("mfa_secret"),
                )
                set_tenant_context(str(row["tenant_id"]))
                set_actor_context(str(row["id"]))
                return user
    except Exception:
        pass
    return None


def _sso_enabled() -> bool:
    return bool(settings.oidc_enabled or settings.azure_ad_enabled or settings.google_workspace_enabled)


def _default_sso_provider() -> str:
    if settings.oidc_enabled:
        return "oidc"
    if settings.azure_ad_enabled:
        return "azure"
    if settings.google_workspace_enabled:
        return "google"
    return "oidc"


def _require_step_up_for_sensitive_action(request: Request, user: User, next_path: str):
    if settings.app_env != "prod" and not settings.hipaa_mode:
        return None
    if not user.mfa_enabled:
        return RedirectResponse("/ui/profile/mfa", status_code=303)
    if is_step_up_verified(request, settings.step_up_window_minutes):
        return None
    safe_next = next_path if next_path.startswith("/ui/") else "/ui"
    return RedirectResponse(f"/ui/step-up?next={quote(safe_next, safe='')}", status_code=303)


def _render_template(request: Request, template: str, context: dict, status_code: int | None = None):
    payload = dict(context)
    payload["request"] = request
    payload["csrf_token"] = get_csrf_token(request)
    payload["csp_nonce"] = getattr(request.state, "csp_nonce", "")
    # Ensure sidebar always has user and dev_mode context
    if "dev_mode" not in payload:
        payload["dev_mode"] = request.session.get("dev_mode", False)
    if "user" not in payload:
        payload["user"] = None
    if status_code is None:
        return templates.TemplateResponse(template, payload)
    return templates.TemplateResponse(template, payload, status_code=status_code)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui", status_code=302)



@limiter.limit("10/minute")
@app.post("/ui/mfa-verify", response_class=HTMLResponse, include_in_schema=False)
def mfa_verify(
    request: Request,
    code: str = Form(...),
    csrf_token: str = Form(...),
):
    validate_csrf_token(request, csrf_token)
    user_id = request.session.get("partial_auth_user_id")
    if not user_id:
        return RedirectResponse("/ui/login", status_code=303)
        
    from .auth import verify_totp, get_user_by_id
    from .mfa import (
        clear_mfa_failures,
        consume_mfa_lockout_expiry,
        get_mfa_lockout_remaining,
        record_mfa_failure,
    )
    with get_conn() as conn:
        user = get_user_by_id(user_id, conn)
        
    if not user or not user.get("mfa_secret"):
         return RedirectResponse("/ui/login", status_code=303)
    if consume_mfa_lockout_expiry(str(user["id"]), "login"):
        with get_conn() as conn:
            _log_action(
                conn,
                None,
                "auth.mfa_lockout_cleared",
                user["email"],
                {"ip": request.client.host, "flow": "login"},
                tenant_id=str(user["tenant_id"]),
            )
            conn.commit()
    remaining = get_mfa_lockout_remaining(str(user["id"]), "login")
    if remaining > 0:
        return _render_template(
            request,
            "mfa_challenge.html",
            {"error": f"Too many attempts. Try again in {remaining} seconds.", "email": user["email"]},
            status_code=429,
        )

    if verify_totp(user["mfa_secret"], code):
        clear_mfa_failures(str(user["id"]), "login")
        request.session.clear()
        renew_session(request)
        request.session["user_id"] = str(user["id"])
        mark_step_up_verified(request)
        with get_conn() as conn:
             _log_action(conn, None, "auth.login_mfa_success", user["email"], {"ip": request.client.host}, tenant_id=str(user["tenant_id"]))
             conn.commit()
        return RedirectResponse("/ui", status_code=303)
    locked = record_mfa_failure(str(user["id"]), "login")
    if locked:
        with get_conn() as conn:
            _log_action(
                conn,
                None,
                "auth.mfa_lockout",
                user["email"],
                {"ip": request.client.host, "flow": "login"},
                tenant_id=str(user["tenant_id"]),
            )
            conn.commit()
        remaining = get_mfa_lockout_remaining(str(user["id"]), "login")
        return _render_template(
            request,
            "mfa_challenge.html",
            {"error": f"Too many attempts. Try again in {remaining} seconds.", "email": user["email"]},
            status_code=429,
        )
    return _render_template(request, "mfa_challenge.html", {"error": "Invalid code", "email": user["email"]}, status_code=401)


@app.get("/ui/profile/mfa", response_class=HTMLResponse, include_in_schema=False)
def mfa_setup_page(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
        
    from .auth import generate_totp_secret, get_totp_uri
    from .mfa import create_mfa_setup_token
    secret = generate_totp_secret()
    uri = get_totp_uri(user.email, secret)
    
    # Generate QR code as base64
    import qrcode
    import io
    import base64
    
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # Store secret server-side; only keep token id in the cookie session.
    token_id = create_mfa_setup_token(str(user.id), str(user.tenant_id), secret, ttl_minutes=10)
    request.session["mfa_setup_token_id"] = token_id
    request.session.pop("mfa_setup_secret", None)  # legacy

    return _render_template(request, "mfa_setup.html", {"user": user, "qr_code": img_b64, "secret": secret})

@app.post("/ui/profile/mfa", response_class=HTMLResponse, include_in_schema=False)
def mfa_enable(
    request: Request,
    code: str = Form(...),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)
    
    from .mfa import get_mfa_setup_secret, consume_mfa_setup_token
    token_id = request.session.get("mfa_setup_token_id")
    if not token_id:
        return RedirectResponse("/ui/profile/mfa", status_code=303)

    secret = get_mfa_setup_secret(token_id, str(user.id))
    if not secret:
        request.session.pop("mfa_setup_token_id", None)
        return RedirectResponse("/ui/profile/mfa", status_code=303)
        
    from .auth import verify_totp, encrypt_mfa_secret
    if verify_totp(secret, code):
        encrypted_secret = encrypt_mfa_secret(secret)
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET mfa_secret = NULL,
                    mfa_secret_encrypted = %s,
                    mfa_enabled = TRUE
                WHERE id = %s
                """,
                (encrypted_secret, user.id),
            )
            _log_action(conn, None, "auth.mfa_enabled", user.email, {"ip": request.client.host}, tenant_id=str(user.tenant_id))
            conn.commit()
        consume_mfa_setup_token(token_id, str(user.id))
        request.session.pop("mfa_setup_token_id", None)
        return RedirectResponse("/ui", status_code=303)

    # Re-render setup page with the same secret/QR for retries.
    from .auth import get_totp_uri
    import qrcode
    import io
    import base64

    uri = get_totp_uri(user.email, secret)
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return _render_template(
        request,
        "mfa_setup.html",
        {"user": user, "error": "Invalid code", "secret": secret, "qr_code": img_b64},
        status_code=400,
    )


@app.get("/ui/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(request: Request):
    """Render login page. In dev mode, redirect straight to dashboard."""
    if settings.app_env == "dev":
        return RedirectResponse("/ui", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "csrf_token": get_csrf_token(request), "csp_nonce": getattr(request.state, "csp_nonce", "")},
    )


@limiter.limit("10/minute")
@app.post("/ui/login", response_class=HTMLResponse, include_in_schema=False)
def login_submit(request: Request, email: str = Form(...), password: str = Form(...), csrf_token: str = Form(...)):
    """Handle login form submission."""
    validate_csrf_token(request, csrf_token)
    try:
        user = authenticate_user(email, password)
        if user:
            request.session["user_id"] = str(user.id)
            return RedirectResponse("/ui", status_code=303)
    except Exception:
        pass
    return RedirectResponse("/ui/login?error=invalid", status_code=303)


@app.get("/ui/logout", include_in_schema=False)
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/ui/login", status_code=303)


@app.get("/ui/step-up", response_class=HTMLResponse, include_in_schema=False)
def step_up_page(request: Request, next: str = "/ui"):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    if not user.mfa_enabled:
        return RedirectResponse("/ui/profile/mfa", status_code=303)
    safe_next = next if next.startswith("/ui/") else "/ui"
    return _render_template(
        request,
        "mfa_challenge.html",
        {"email": user.email, "step_up": True, "action": "/ui/step-up", "next": safe_next},
    )


@limiter.limit("10/minute")
@app.post("/ui/step-up", response_class=HTMLResponse, include_in_schema=False)
def step_up_verify(
    request: Request,
    code: str = Form(...),
    next: str = Form("/ui"),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)
    if not user.mfa_secret:
        return RedirectResponse("/ui/profile/mfa", status_code=303)
    from .auth import verify_totp
    from .mfa import (
        clear_mfa_failures,
        consume_mfa_lockout_expiry,
        get_mfa_lockout_remaining,
        record_mfa_failure,
    )

    safe_next = next if next.startswith("/ui/") else "/ui"
    if consume_mfa_lockout_expiry(str(user.id), "step_up"):
        with get_conn() as conn:
            _log_action(
                conn,
                None,
                "auth.mfa_lockout_cleared",
                user.email,
                {"ip": request.client.host, "flow": "step_up"},
                tenant_id=str(user.tenant_id),
            )
            conn.commit()
    remaining = get_mfa_lockout_remaining(str(user.id), "step_up")
    if remaining > 0:
        return _render_template(
            request,
            "mfa_challenge.html",
            {
                "email": user.email,
                "step_up": True,
                "action": "/ui/step-up",
                "next": safe_next,
                "error": f"Too many attempts. Try again in {remaining} seconds.",
            },
            status_code=429,
        )
    if verify_totp(user.mfa_secret, code):
        clear_mfa_failures(str(user.id), "step_up")
        mark_step_up_verified(request)
        with get_conn() as conn:
            _log_action(
                conn,
                None,
                "auth.step_up_success",
                user.email,
                {"ip": request.client.host, "next": safe_next},
                tenant_id=str(user.tenant_id),
            )
            conn.commit()
        return RedirectResponse(safe_next, status_code=303)
    locked = record_mfa_failure(str(user.id), "step_up")
    if locked:
        with get_conn() as conn:
            _log_action(
                conn,
                None,
                "auth.mfa_lockout",
                user.email,
                {"ip": request.client.host, "flow": "step_up"},
                tenant_id=str(user.tenant_id),
            )
            conn.commit()
        remaining = get_mfa_lockout_remaining(str(user.id), "step_up")
        return _render_template(
            request,
            "mfa_challenge.html",
            {
                "email": user.email,
                "step_up": True,
                "action": "/ui/step-up",
                "next": safe_next,
                "error": f"Too many attempts. Try again in {remaining} seconds.",
            },
            status_code=429,
        )
    return _render_template(
        request,
        "mfa_challenge.html",
        {
            "email": user.email,
            "step_up": True,
            "action": "/ui/step-up",
            "next": safe_next,
            "error": "Invalid code",
        },
        status_code=401,
    )


@app.get("/ui/sso/{provider}/login", include_in_schema=False)
async def sso_login(request: Request, provider: str):
    if not _sso_enabled():
        raise HTTPException(status_code=404, detail="SSO is not enabled")
    return await initiate_sso_login(request, provider=provider)


@app.get("/ui/sso/login", include_in_schema=False)
async def sso_login_default(request: Request):
    if not _sso_enabled():
        raise HTTPException(status_code=404, detail="SSO is not enabled")
    return await initiate_sso_login(request, provider=_default_sso_provider())


@app.get("/auth/callback/{provider}", include_in_schema=False)
async def sso_callback(request: Request, provider: str):
    if not _sso_enabled():
        raise HTTPException(status_code=404, detail="SSO is not enabled")
    try:
        user_info = await handle_sso_callback(request, provider=provider)
        provisioned = provision_sso_user(user_info)
    except Exception as exc:
        with get_conn() as conn:
            append_audit_event(
                conn,
                action="auth.sso_denied",
                resource_type="user",
                outcome="DENIED",
                details={"provider": provider, "reason": str(exc)},
                ip_address=request.client.host if request.client else None,
            )
            conn.commit()
        return RedirectResponse("/ui/login?error=sso", status_code=303)

    request.session.clear()
    renew_session(request)
    request.session["user_id"] = provisioned["user_id"]
    with get_conn() as conn:
        append_audit_event(
            conn,
            action="auth.sso_success",
            resource_type="user",
            outcome="SUCCESS",
            details={"provider": provider, "email": provisioned.get("email")},
            tenant_id=provisioned.get("tenant_id"),
            actor=provisioned.get("email"),
            ip_address=request.client.host if request.client else None,
        )
        conn.commit()
    return RedirectResponse("/ui", status_code=303)


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui_patients(request: Request, page: int = Query(1, ge=1)):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    per_page = 25

    with get_conn() as conn:
        # Total patient count and aggregate stats (for stats cards)
        totals = conn.execute(
            """
            SELECT
                COUNT(DISTINCT p.id) AS total_patients,
                COUNT(d.id) AS total_docs
            FROM patients p
            LEFT JOIN documents d ON p.id = d.patient_id
            WHERE p.tenant_id = %s
            """,
            (user.tenant_id,),
        ).fetchone()
        total_patients = int(totals["total_patients"] or 0)
        total_docs = int(totals["total_docs"] or 0)
        total_pages = max(1, math.ceil(total_patients / per_page))

        # Clamp page to valid range
        if page > total_pages:
            page = total_pages
        offset = (page - 1) * per_page

        # Total reports count
        report_count_row = conn.execute(
            """
            SELECT COUNT(DISTINCT cv.patient_id) AS report_count
            FROM chr_versions cv
            JOIN patients p ON p.id = cv.patient_id
            WHERE p.tenant_id = %s
            """,
            (user.tenant_id,),
        ).fetchone()
        report_count = int(report_count_row["report_count"] or 0)

        # Paginated patient rows (same query + LIMIT/OFFSET)
        rows = conn.execute(
            """
            SELECT
                   p.id,
                   p.full_name,
                   p.dob,
                   p.notes,
                   p.created_at,
                   COUNT(d.id) AS doc_count,
                   COUNT(*) FILTER (
                     WHERE POSITION('pdf' IN COALESCE(lower(d.content_type), '')) > 0
                        OR RIGHT(lower(COALESCE(d.filename, '')), 4) = '.pdf'
                   ) AS pdf_count,
                   COUNT(*) FILTER (
                     WHERE LEFT(COALESCE(lower(d.content_type), ''), 6) = 'image/'
                        OR lower(COALESCE(d.filename, '')) ~ '\\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff)$'
                   ) AS image_count,
                   COUNT(*) FILTER (
                     WHERE LEFT(COALESCE(lower(d.content_type), ''), 5) = 'text/'
                        OR lower(COALESCE(d.filename, '')) ~ '\\.(txt|md|rtf|csv|tsv|json|xml)$'
                   ) AS text_count
            FROM patients p
            LEFT JOIN documents d ON p.id = d.patient_id
            WHERE p.tenant_id = %s
            GROUP BY p.id, p.full_name, p.dob, p.notes, p.created_at
            ORDER BY
                   (
                     (COUNT(*) FILTER (
                       WHERE POSITION('pdf' IN COALESCE(lower(d.content_type), '')) > 0
                          OR RIGHT(lower(COALESCE(d.filename, '')), 4) = '.pdf'
                     ) > 0)::int
                     +
                     (COUNT(*) FILTER (
                       WHERE LEFT(COALESCE(lower(d.content_type), ''), 6) = 'image/'
                          OR lower(COALESCE(d.filename, '')) ~ '\\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff)$'
                     ) > 0)::int
                     +
                     (COUNT(*) FILTER (
                       WHERE LEFT(COALESCE(lower(d.content_type), ''), 5) = 'text/'
                          OR lower(COALESCE(d.filename, '')) ~ '\\.(txt|md|rtf|csv|tsv|json|xml)$'
                     ) > 0)::int
                   ) DESC,
                   doc_count DESC,
                   p.created_at DESC
            LIMIT %s OFFSET %s
            """,
            (user.tenant_id, per_page, offset),
        ).fetchall()

        # Multimodal patients (featured demo section) — always fetched in full
        multimodal_rows = conn.execute(
            """
            SELECT
                   p.id,
                   p.full_name,
                   COUNT(d.id) AS doc_count,
                   COUNT(*) FILTER (
                     WHERE POSITION('pdf' IN COALESCE(lower(d.content_type), '')) > 0
                        OR RIGHT(lower(COALESCE(d.filename, '')), 4) = '.pdf'
                   ) AS pdf_count,
                   COUNT(*) FILTER (
                     WHERE LEFT(COALESCE(lower(d.content_type), ''), 6) = 'image/'
                        OR lower(COALESCE(d.filename, '')) ~ '\\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff)$'
                   ) AS image_count,
                   COUNT(*) FILTER (
                     WHERE LEFT(COALESCE(lower(d.content_type), ''), 5) = 'text/'
                        OR lower(COALESCE(d.filename, '')) ~ '\\.(txt|md|rtf|csv|tsv|json|xml)$'
                   ) AS text_count
            FROM patients p
            LEFT JOIN documents d ON p.id = d.patient_id
            WHERE p.tenant_id = %s
            GROUP BY p.id, p.full_name
            HAVING
                   COUNT(*) FILTER (
                     WHERE POSITION('pdf' IN COALESCE(lower(d.content_type), '')) > 0
                        OR RIGHT(lower(COALESCE(d.filename, '')), 4) = '.pdf'
                   ) > 0
                   AND COUNT(*) FILTER (
                     WHERE LEFT(COALESCE(lower(d.content_type), ''), 6) = 'image/'
                        OR lower(COALESCE(d.filename, '')) ~ '\\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff)$'
                   ) > 0
                   AND COUNT(*) FILTER (
                     WHERE LEFT(COALESCE(lower(d.content_type), ''), 5) = 'text/'
                        OR lower(COALESCE(d.filename, '')) ~ '\\.(txt|md|rtf|csv|tsv|json|xml)$'
                   ) > 0
            ORDER BY doc_count DESC
            """,
            (user.tenant_id,),
        ).fetchall()
        multimodal_patients = [
            {
                "id": str(mr["id"]),
                "full_name": mr["full_name"],
                "doc_count": int(mr["doc_count"] or 0),
                "pdf_count": int(mr["pdf_count"] or 0),
                "image_count": int(mr["image_count"] or 0),
                "text_count": int(mr["text_count"] or 0),
            }
            for mr in multimodal_rows
        ]

        # Get report status and data profiles for the current page
        report_statuses = {}
        doc_counts = {}
        data_profiles = {}
        for r in rows:
            pid = str(r["id"])
            pdf_count = int(r.get("pdf_count") or 0)
            image_count = int(r.get("image_count") or 0)
            text_count = int(r.get("text_count") or 0)
            has_all_data_types = pdf_count > 0 and image_count > 0 and text_count > 0
            data_profiles[pid] = {
                "pdf": pdf_count,
                "image": image_count,
                "text": text_count,
                "has_all": has_all_data_types,
            }
            draft = conn.execute(
                "SELECT status FROM chr_versions WHERE patient_id = %s ORDER BY created_at DESC LIMIT 1",
                (pid,)
            ).fetchone()
            report_statuses[pid] = draft["status"] if draft else None
            doc_counts[pid] = int(r["doc_count"] or 0)

    patients = [_row_to_patient(r) for r in rows]
    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "patients.html",
        {
            "patients": patients,
            "user": user,
            "dev_mode": dev_mode,
            "report_statuses": report_statuses,
            "doc_counts": doc_counts,
            "report_count": report_count,
            "total_docs": total_docs,
            "total_patients": total_patients,
            "data_profiles": data_profiles,
            "multimodal_patients": multimodal_patients,
            "multimodal_count": len(multimodal_patients),
            "page": page,
            "total_pages": total_pages,
            "per_page": per_page,
        },
    )


@app.post("/ui/toggle-dev-mode", include_in_schema=False)
def ui_toggle_dev_mode(request: Request, csrf_token: str = Form(...)):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)
    
    current = request.session.get("dev_mode", False)
    request.session["dev_mode"] = not current
    # Redirect back to referring page or patients list
    referer = request.headers.get("referer", "/ui")
    return RedirectResponse(referer, status_code=303)


@app.get("/ui/embeddings", response_class=HTMLResponse, include_in_schema=False)
def ui_embeddings(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.full_name, d.filename, COUNT(e.id) AS chunk_count, MAX(e.created_at) AS created_at
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            LEFT JOIN embeddings e ON e.document_id = d.id
            WHERE p.tenant_id = %s
            GROUP BY p.full_name, d.filename
            ORDER BY created_at DESC NULLS LAST
            LIMIT 100
            """,
            (user.tenant_id,)
        ).fetchall()

    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "embeddings.html",
        {"rows": rows, "user": user, "dev_mode": dev_mode},
    )


@app.get("/ui/data", response_class=HTMLResponse, include_in_schema=False)
def ui_data_catalog(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                p.id AS patient_id,
                p.full_name,
                d.id AS document_id,
                d.filename,
                d.content_type,
                d.created_at
            FROM patients p
            LEFT JOIN documents d ON d.patient_id = p.id
            WHERE p.tenant_id = %s
            ORDER BY p.full_name ASC, d.created_at DESC NULLS LAST
            """,
            (str(user.tenant_id),),
        ).fetchall()

    def _data_kind(content_type: str | None, filename: str | None) -> str:
        ctype = (content_type or "").lower()
        name = (filename or "").lower()
        if "pdf" in ctype or name.endswith(".pdf"):
            return "pdf"
        if ctype.startswith("image/") or re.search(r"\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff)$", name):
            return "image"
        if ctype.startswith("text/") or re.search(r"\.(txt|md|rtf|csv|tsv|json|xml)$", name):
            return "text"
        return "other"

    grouped: dict[str, dict] = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        if patient_id not in grouped:
            grouped[patient_id] = {
                "patient_id": patient_id,
                "patient_name": row["full_name"],
                "files": [],
                "pdf_count": 0,
                "image_count": 0,
                "text_count": 0,
                "other_count": 0,
            }

        document_id = row.get("document_id")
        if not document_id:
            continue

        kind = _data_kind(row.get("content_type"), row.get("filename"))
        if kind == "pdf":
            grouped[patient_id]["pdf_count"] += 1
        elif kind == "image":
            grouped[patient_id]["image_count"] += 1
        elif kind == "text":
            grouped[patient_id]["text_count"] += 1
        else:
            grouped[patient_id]["other_count"] += 1

        grouped[patient_id]["files"].append(
            {
                "document_id": str(document_id),
                "filename": row.get("filename") or "Unnamed file",
                "content_type": row.get("content_type") or "unknown",
                "created_at": row.get("created_at"),
                "kind": kind,
            }
        )

    patient_groups = list(grouped.values())
    for item in patient_groups:
        item["file_count"] = len(item["files"])
        item["data_type_score"] = sum(
            1
            for key in ("pdf_count", "image_count", "text_count")
            if int(item.get(key, 0)) > 0
        )
        item["has_all_data_types"] = item["data_type_score"] == 3

    patient_groups.sort(
        key=lambda item: (
            -int(item["has_all_data_types"]),
            -int(item["data_type_score"]),
            -int(item["file_count"]),
            (item["patient_name"] or "").lower(),
        )
    )

    featured_patients = [item for item in patient_groups if item["has_all_data_types"]]
    total_files = sum(item["file_count"] for item in patient_groups)

    return _render_template(
        request,
        "data.html",
        {
            "user": user,
            "patient_groups": patient_groups,
            "featured_patients": featured_patients,
            "featured_count": len(featured_patients),
            "total_files": total_files,
            "patient_count": len(patient_groups),
        },
    )


@app.post("/ui/patients", include_in_schema=False)
def ui_create_patient(
    request: Request,
    full_name: str = Form(...),
    dob: str = Form(""),
    notes: str = Form(""),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    payload = PatientCreate(full_name=full_name, dob=dob or None, notes=notes or None)
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO patients (tenant_id, full_name, dob, notes)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (user.tenant_id, payload.full_name, payload.dob, payload.notes),
        ).fetchone()
        _log_action(conn, str(row["id"]), "patient.create", user.email, {"name": payload.full_name}, tenant_id=str(user.tenant_id))
        conn.commit()

    return RedirectResponse("/ui", status_code=303)


@app.get("/ui/embeddings", response_class=HTMLResponse, include_in_schema=False)
def ui_embeddings(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.full_name, d.filename, COUNT(e.id) AS chunk_count, MAX(e.created_at) AS created_at
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            LEFT JOIN embeddings e ON e.document_id = d.id
            WHERE p.tenant_id = %s
            GROUP BY p.full_name, d.filename
            ORDER BY created_at DESC NULLS LAST
            LIMIT 100
            """,
            (user.tenant_id,)
        ).fetchall()

    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "embeddings.html",
        {"rows": rows, "user": user, "dev_mode": dev_mode},
    )


def _get_patient(patient_id: str, tenant_id: str | None = None):
    with get_conn() as conn:
        query = "SELECT id, full_name, dob, notes FROM patients WHERE id = %s"
        params = [patient_id]
        if tenant_id:
            query += " AND tenant_id = %s"
            params.append(tenant_id)
        return conn.execute(query, tuple(params)).fetchone()


def _list_documents(patient_id: str, tenant_id: str | None = None):
    with get_conn() as conn:
        if tenant_id:
            return conn.execute(
                """
                SELECT d.id, d.patient_id, d.filename, d.content_type, d.storage_path
                FROM documents d
                JOIN patients p ON p.id = d.patient_id
                WHERE d.patient_id = %s
                  AND p.tenant_id = %s
                ORDER BY d.created_at DESC
                """,
                (patient_id, tenant_id),
            ).fetchall()
        else:
            return conn.execute(
                """
                SELECT d.id, d.patient_id, d.filename, d.content_type, d.storage_path
                FROM documents d
                WHERE d.patient_id = %s
                ORDER BY d.created_at DESC
                """,
                (patient_id,),
            ).fetchall()


def _latest_draft(patient_id: str):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, draft, status, created_at
            FROM chr_versions
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()


def _audit_logs(patient_id: str):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT actor, action, details, created_at
            FROM audit_logs
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (patient_id,),
        ).fetchall()


def _latest_extraction(patient_id: str, tenant_id: str | None = None):
    with get_conn() as conn:
        if tenant_id:
            return conn.execute(
                """
                SELECT e.raw_text, e.structured
                FROM extractions e
                JOIN documents d ON d.id = e.document_id
                JOIN patients p ON p.id = d.patient_id
                WHERE d.patient_id = %s
                  AND p.tenant_id = %s
                ORDER BY e.created_at DESC
                LIMIT 1
                """,
                (patient_id, tenant_id),
            ).fetchone()
        else:
            return conn.execute(
                """
                SELECT e.raw_text, e.structured
                FROM extractions e
                JOIN documents d ON d.id = e.document_id
                WHERE d.patient_id = %s
                ORDER BY e.created_at DESC
                LIMIT 1
                """,
                (patient_id,),
            ).fetchone()


def _draft_payload(draft_row) -> dict:
    if not draft_row:
        return {}
    payload = draft_row.get("draft")
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_labs(structured: dict | None) -> list[dict]:
    if not structured:
        return []
    labs = structured.get("labs") or structured.get("biomarkers") or []
    normalized = []
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        flag = (lab.get("flag") or "").strip()
        flag_upper = flag.upper()
        if flag_upper in {"H", "HIGH"}:
            flag_label = "High"
        elif flag_upper in {"L", "LOW"}:
            flag_label = "Low"
        else:
            flag_label = "Normal" if flag else ""
        normalized.append(
            {
                "panel": lab.get("panel"),
                "test": lab.get("test") or lab.get("test_name") or lab.get("name"),
                "value": lab.get("value"),
                "unit": lab.get("unit"),
                "range": lab.get("range"),
                "flag": flag_label,
                "abnormal": flag_label in {"High", "Low"},
            }
        )
    return normalized


def _key_findings(labs: list[dict]) -> list[str]:
    findings = []
    for lab in labs:
        if not lab.get("abnormal"):
            continue
        test = lab.get("test") or lab.get("test_name") or lab.get("name") or lab.get("panel") or "Unlabeled Test"
        value = lab.get("value") or ""
        unit = lab.get("unit") or ""
        flag = lab.get("flag") or ""
        findings.append(f"{test}: {value} {unit} ({flag})")
    return findings


def _has_extractions(patient_id: str) -> bool:
    """Check if patient has any extracted documents."""
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM extractions e
            JOIN documents d ON d.id = e.document_id
            WHERE d.patient_id = %s
            """,
            (patient_id,),
        ).fetchone()
    return row and row["cnt"] > 0


@app.get("/ui/patients/{patient_id}", response_class=HTMLResponse, include_in_schema=False)
def ui_patient_detail(request: Request, patient_id: str):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    documents = [_row_to_document(r) for r in _list_documents(patient_id, tenant_id=str(user.tenant_id))]
    draft = _latest_draft(patient_id)
    logs = _audit_logs(patient_id)
    has_extractions = _has_extractions(patient_id)
    dev_mode = request.session.get("dev_mode", False)
    pending_jobs = (
        list_jobs(patient_id=patient_id, tenant_id=str(user.tenant_id), statuses=["pending", "running"])
        if settings.job_queue_enabled
        else []
    )

    # Fetch clinical records for past data display and visualizations
    vitals = []
    lab_results = []
    medications = []
    diagnoses_list = []
    allergies = []
    immunizations = []
    chart_data = {}
    trend_summary = {}

    with get_conn() as conn:
        _log_action(conn, patient_id, "patient.view", user.email, {"ip": request.client.host}, tenant_id=str(user.tenant_id))

        # Vitals
        vitals = conn.execute(
            "SELECT type, value_1, value_2, unit, recorded_at FROM vitals WHERE patient_id = %s ORDER BY recorded_at DESC LIMIT 100",
            (patient_id,),
        ).fetchall()

        # Lab results
        lab_results = conn.execute(
            "SELECT test_name, value, unit, flag, reference_range, test_date, panel FROM lab_results WHERE patient_id = %s ORDER BY test_date DESC NULLS LAST LIMIT 200",
            (patient_id,),
        ).fetchall()

        # Medications
        medications = conn.execute(
            "SELECT name, dosage, frequency, route, start_date, end_date, status FROM medications WHERE patient_id = %s ORDER BY status ASC, start_date DESC NULLS LAST",
            (patient_id,),
        ).fetchall()

        # Diagnoses
        diagnoses_list = conn.execute(
            "SELECT condition, code, status, date_onset FROM diagnoses WHERE patient_id = %s ORDER BY date_onset DESC NULLS LAST",
            (patient_id,),
        ).fetchall()

        # Allergies
        allergies = conn.execute(
            "SELECT substance, reaction, severity, status FROM allergies WHERE patient_id = %s ORDER BY created_at DESC",
            (patient_id,),
        ).fetchall()

        # Immunizations
        immunizations = conn.execute(
            "SELECT vaccine_name, date_administered, status FROM immunizations WHERE patient_id = %s ORDER BY date_administered DESC NULLS LAST",
            (patient_id,),
        ).fetchall()

        conn.commit()

    # Build chart data from lab results for visualization
    try:
        from .trends import analyze_patient_trends, generate_trend_chart_data
        labs_for_trends = [
            {"test_name": r["test_name"], "value": r["value"], "date": r["test_date"].isoformat() if r.get("test_date") else None, "unit": r.get("unit")}
            for r in lab_results if r.get("value")
        ]
        if labs_for_trends:
            trend_result = analyze_patient_trends(labs_for_trends)
            trend_summary = trend_result.get("summary", {})

            # Group lab values by test for chart data
            by_test = {}
            for lab in labs_for_trends:
                t = lab["test_name"]
                if t not in by_test:
                    by_test[t] = []
                by_test[t].append(lab)

            for test_name, values in by_test.items():
                if len(values) >= 2:
                    chart_data[test_name] = generate_trend_chart_data(values)
    except Exception:
        pass  # Trends are optional; don't break the page

    # --- Clinical Safety Alerts ---
    safety_alerts = []
    try:
        from .alerts import check_critical_values, check_drug_interactions, check_allergy_contraindications
        # Critical lab values
        labs_for_alerts = [
            {"test_name": r["test_name"], "value": r["value"], "unit": r.get("unit", "")}
            for r in lab_results if r.get("value")
        ]
        critical = check_critical_values(labs_for_alerts)
        safety_alerts.extend(critical)

        # Drug interactions
        med_names = [r["name"] for r in medications if r.get("name")]
        if len(med_names) >= 2:
            interactions = check_drug_interactions(med_names)
            safety_alerts.extend(interactions)

        # Allergy contraindications
        allergy_names = [r["substance"] for r in allergies if r.get("substance")]
        if allergy_names and med_names:
            contras = check_allergy_contraindications(allergy_names, med_names)
            safety_alerts.extend(contras)
    except Exception:
        pass  # Alerts are optional

    return _render_template(
        request,
        "patient_detail.html",
        {
            "user": user,
            "patient": _row_to_patient(patient_row),
            "documents": documents,
            "draft": draft,
            "logs": logs,
            "has_extractions": has_extractions,
            "pending_jobs": pending_jobs,
            "dev_mode": dev_mode,
            "vitals": [dict(r) for r in vitals],
            "lab_results": [dict(r) for r in lab_results],
            "medications": [dict(r) for r in medications],
            "diagnoses_list": [dict(r) for r in diagnoses_list],
            "allergies": [dict(r) for r in allergies],
            "immunizations": [dict(r) for r in immunizations],
            "chart_data": chart_data,
            "trend_summary": trend_summary,
            "safety_alerts": safety_alerts,
        },
    )


@app.get("/ui/patients/{patient_id}/report", response_class=HTMLResponse, include_in_schema=False)
def ui_patient_report(request: Request, patient_id: str):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    tenant_id = str(user.tenant_id)
    structured, _sources = _aggregate_structured(patient_id, tenant_id=tenant_id)
    labs = _normalize_labs(structured)
    meds = (structured.get("medications") or []) if structured else []
    diagnoses = (structured.get("diagnoses") or []) if structured else []
    allergies_raw = (structured.get("allergies") or []) if structured else []
    vitals_raw = (structured.get("vitals") or []) if structured else []
    extraction_quality = structured.get("extraction_quality") if structured else None

    # Phase 1: Fetch allergies and vitals from dedicated tables
    with get_conn() as conn:
        allergy_rows = conn.execute(
            "SELECT substance, reaction, severity, status FROM allergies WHERE patient_id = %s",
            (patient_id,),
        ).fetchall()
        vital_rows = conn.execute(
            "SELECT type, value_1, value_2, unit, recorded_at FROM vitals WHERE patient_id = %s ORDER BY recorded_at DESC LIMIT 20",
            (patient_id,),
        ).fetchall()

    # Merge allergies: from extraction + from dedicated table (deduplicated)
    seen_allergies = set()
    combined_allergies = []
    for a in allergy_rows:
        key = (a["substance"] or "").lower()
        if key and key not in seen_allergies:
            seen_allergies.add(key)
            combined_allergies.append({
                "substance": a["substance"],
                "reaction": a.get("reaction") or "",
                "severity": a.get("severity") or "",
                "status": a.get("status") or "active",
            })
    for a in allergies_raw:
        key = (a.get("substance") or "").lower()
        if key and key not in seen_allergies:
            seen_allergies.add(key)
            combined_allergies.append(a)

    # Normalize vitals from DB rows
    combined_vitals = []
    for v in vital_rows:
        val = str(v["value_1"])
        if v.get("value_2"):
            val = f"{v['value_1']}/{v['value_2']}"
        combined_vitals.append({
            "type": v["type"],
            "value": val,
            "unit": v.get("unit") or "",
            "recorded_at": str(v["recorded_at"])[:16] if v.get("recorded_at") else "",
        })
    # Also include vitals from extraction if no DB vitals
    if not combined_vitals and vitals_raw:
        for v in vitals_raw:
            combined_vitals.append({
                "type": v.get("type") or "",
                "value": v.get("value") or "",
                "unit": v.get("unit") or "",
                "recorded_at": v.get("date") or "",
            })

    # Phase 1: Run safety checks
    med_names = []
    for m in meds:
        if isinstance(m, dict):
            med_names.append(m.get("name") or "")
        elif isinstance(m, str):
            med_names.append(m)
    allergy_names = [a.get("substance", "") for a in combined_allergies]
    raw_labs = (structured.get("labs") or []) if structured else []
    safety_alerts = run_safety_checks(
        labs=raw_labs,
        medications=med_names,
        allergies=allergy_names,
    )

    # Extraction confidence
    overall_confidence = None
    if extraction_quality and isinstance(extraction_quality, dict):
        overall_confidence = extraction_quality.get("overall_confidence")

    draft_row = _latest_draft(patient_id)
    report_edits = draft_row.get("report_edits") if draft_row else {}
    edited_interpretation = report_edits.get("interpretation") if isinstance(report_edits, dict) else None
    edited_by = report_edits.get("edited_by") if isinstance(report_edits, dict) else None
    if isinstance(report_edits, dict):
        if isinstance(report_edits.get("labs"), list):
            labs = report_edits.get("labs") or labs
        if isinstance(report_edits.get("diagnoses"), list):
            diagnoses = report_edits.get("diagnoses") or diagnoses
    draft_payload = _draft_payload(draft_row)
    summary = draft_payload.get("summary", "")
    citations = draft_payload.get("citations", [])
    
    # Convert markdown to HTML and make citations clickable
    def process_citations(text):
        """Convert [#] references to clickable anchor links."""
        def replace_citation(match):
            num = match.group(1)
            return f'<a href="#cite-{num}" class="citation-link" title="View source">[{num}]</a>'
        return re.sub(r'\[(\d+)\]', replace_citation, text)
    
    if summary:
        summary_html = render_markdown(process_citations(summary))
    else:
        summary_html = ""

    edited_interpretation_html = (
        render_markdown(process_citations(edited_interpretation)) if edited_interpretation else ""
    )

    documents = _list_documents(patient_id, tenant_id=tenant_id)
    findings = _key_findings(labs)

    # Build patient social/lifestyle context
    patient_obj = _row_to_patient(patient_row)
    lifestyle = patient_obj.lifestyle or {}
    # Merge social_history from patient row and lifestyle
    social_history = patient_obj.social_history or {}
    if lifestyle and not social_history:
        social_history = lifestyle

    # Phase 2: Fetch clinical sections from new tables (graceful fallback if tables don't exist)
    encounters = []
    pmh_records = []
    family_history_records = []
    treatment_plans = []
    ros_records = []
    try:
        with get_conn() as conn:
            encounters = conn.execute(
                """SELECT encounter_date, encounter_type, chief_complaint, hpi, provider_name, status
                FROM encounters WHERE patient_id = %s AND tenant_id = %s
                ORDER BY encounter_date DESC LIMIT 5""",
                (patient_id, tenant_id),
            ).fetchall()
            pmh_records = conn.execute(
                """SELECT history_type, condition, icd_code, date_onset, date_resolved, status
                FROM clinical_history WHERE patient_id = %s AND tenant_id = %s
                ORDER BY date_onset DESC""",
                (patient_id, tenant_id),
            ).fetchall()
            family_history_records = conn.execute(
                """SELECT relation, condition, age_at_onset, is_deceased, notes
                FROM family_history WHERE patient_id = %s AND tenant_id = %s""",
                (patient_id, tenant_id),
            ).fetchall()
            treatment_plans = conn.execute(
                """SELECT plan_type, description, priority, status, target_date, created_by
                FROM treatment_plans WHERE patient_id = %s AND tenant_id = %s
                ORDER BY created_at DESC LIMIT 10""",
                (patient_id, tenant_id),
            ).fetchall()
            ros_records = conn.execute(
                """SELECT system_name, findings, is_positive, notes
                FROM review_of_systems WHERE patient_id = %s AND tenant_id = %s
                ORDER BY system_name""",
                (patient_id, tenant_id),
            ).fetchall()
    except Exception:
        # Tables may not exist yet if migration hasn't run
        pass

    with get_conn() as conn:
        _log_action(conn, patient_id, "report.view", user.email, {"ip": request.client.host}, tenant_id=tenant_id)
        conn.commit()

    # Available specialty templates
    specialty_templates = list_templates()

    # Combine PMH from structured patient field + clinical_history table
    combined_pmh = list(patient_obj.past_medical_history or [])
    for rec in pmh_records:
        combined_pmh.append({
            "type": rec.get("history_type", "medical"),
            "condition": rec["condition"],
            "icd_code": rec.get("icd_code"),
            "onset": str(rec["date_onset"]) if rec.get("date_onset") else None,
            "resolved": str(rec["date_resolved"]) if rec.get("date_resolved") else None,
            "status": rec.get("status", "historical"),
        })

    return _render_template(
        request,
        "report.html",
        {
            "user": user.email if user else "System",
            "patient": patient_obj,
            "draft": draft_row,
            "summary": summary,
            "summary_html": summary_html,
            "citations": citations,
            "labs": labs,
            "medications": meds,
            "diagnoses": diagnoses,
            "documents": documents,
            "findings": findings,
            "edited_interpretation_html": edited_interpretation_html,
            "edited_by": edited_by,
            # Phase 1 additions
            "allergies": combined_allergies,
            "vitals": combined_vitals,
            "safety_alerts": safety_alerts,
            "overall_confidence": overall_confidence,
            "social_history": social_history,
            "specialty_templates": specialty_templates,
            # Phase 2 additions
            "encounters": encounters,
            "past_medical_history": combined_pmh,
            "family_history": family_history_records,
            "treatment_plans": treatment_plans,
            "review_of_systems": ros_records,
        },
    )

@app.get("/ui/patients/{patient_id}/report/pdf", include_in_schema=False)
def ui_patient_report_pdf(request: Request, patient_id: str):
    """Generate PDF version of the CHR report."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    # Render the report HTML first
    report_html_response = ui_patient_report(request, patient_id)
    html_content = report_html_response.body.decode("utf-8") if hasattr(report_html_response, "body") else str(report_html_response)

    try:
        from io import BytesIO
        from xhtml2pdf import pisa

        pdf_buffer = BytesIO()
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)

        if pisa_status.err:
            raise HTTPException(status_code=500, detail="PDF generation failed")

        pdf_buffer.seek(0)
        patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
        patient_name = patient_row["full_name"].replace(" ", "_") if patient_row else "patient"
        filename = f"CHR_{patient_name}_{patient_id[:8]}.pdf"

        with get_conn() as conn:
            _log_action(conn, patient_id, "report.pdf_export", user.email, {"ip": request.client.host}, tenant_id=str(user.tenant_id))
            conn.commit()

        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="PDF export not available. Install xhtml2pdf: pip install xhtml2pdf")


@app.get("/ui/patients/{patient_id}/report/share", response_class=HTMLResponse, include_in_schema=False)
def ui_patient_report_share(request: Request, patient_id: str):
    """Patient-friendly shareable report view with plain language."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    require_permission(user.role, "report.share")
    step_up_response = _require_step_up_for_sensitive_action(
        request,
        user,
        f"/ui/patients/{patient_id}/report/share",
    )
    if step_up_response:
        return step_up_response

    patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    draft_row = _latest_draft(patient_id)
    report_edits = draft_row.get("report_edits") if draft_row else {}
    edited_interpretation = report_edits.get("interpretation") if isinstance(report_edits, dict) else None
    edited_by = report_edits.get("edited_by") if isinstance(report_edits, dict) else None

    structured, _sources = _aggregate_structured(patient_id, tenant_id=str(user.tenant_id))
    labs = _normalize_labs(structured)
    diagnoses = (structured.get("diagnoses") or []) if structured else []
    if isinstance(report_edits, dict):
        if isinstance(report_edits.get("labs"), list):
            labs = report_edits.get("labs") or labs
        if isinstance(report_edits.get("diagnoses"), list):
            diagnoses = report_edits.get("diagnoses") or diagnoses
    documents = _list_documents(patient_id, tenant_id=str(user.tenant_id))
    findings = _key_findings(labs)

    with get_conn() as conn:
        _log_action(conn, patient_id, "report.share", user.email, {"ip": request.client.host}, tenant_id=str(user.tenant_id))
        conn.commit()

    return _render_template(
        request,
        "patient_report.html",
        {
            "user": user,
            "patient": _row_to_patient(patient_row),
            "draft": draft_row,
            "labs": labs,
            "diagnoses": diagnoses,
            "documents": documents,
            "findings": findings,
            "edited_interpretation": edited_interpretation,
            "edited_by": edited_by,
        },
    )


@app.get("/ui/patients/{patient_id}/documents/{document_id}/view", response_class=HTMLResponse, include_in_schema=False)
def ui_view_document(request: Request, patient_id: str, document_id: str):
    """Render document content inline in the browser for clinician review."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    with get_conn() as conn:
        doc = conn.execute(
            """
            SELECT d.id, d.filename, d.content_type, d.storage_path
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            WHERE d.id = %s AND d.patient_id = %s AND p.tenant_id = %s
            """,
            (document_id, patient_id, str(user.tenant_id)),
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Fetch extracted text if available
        extraction = conn.execute(
            """
            SELECT raw_text, structured
            FROM extractions
            WHERE document_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()

        _log_action(conn, patient_id, "document.view", user.email, {"document_id": document_id}, tenant_id=str(user.tenant_id))
        conn.commit()

    raw_text = extraction["raw_text"] if extraction else None
    structured = extraction["structured"] if extraction else None

    # Try to generate a signed URL for images/PDFs
    download_url = None
    try:
        download_url = create_signed_download_url(settings.storage_bucket, doc["storage_path"], 600)
    except Exception:
        pass

    return _render_template(
        request,
        "document_viewer.html",
        {
            "user": user,
            "patient": _row_to_patient(patient_row),
            "document": {
                "id": str(doc["id"]),
                "filename": doc["filename"],
                "content_type": doc["content_type"],
            },
            "raw_text": raw_text,
            "structured": structured,
            "download_url": download_url,
        },
    )


@app.get("/ui/patients/{patient_id}/documents/{document_id}/raw", include_in_schema=False)
def ui_download_document_raw(request: Request, patient_id: str, document_id: str):
    """Serve the raw document file for inline viewing (images/PDFs)."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        doc = conn.execute(
            """
            SELECT d.id, d.filename, d.content_type, d.storage_path
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            WHERE d.id = %s AND d.patient_id = %s AND p.tenant_id = %s
            """,
            (document_id, patient_id, str(user.tenant_id)),
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

    try:
        data = download_bytes(settings.storage_bucket, doc["storage_path"])
        from starlette.responses import Response
        return Response(
            content=data,
            media_type=doc["content_type"],
            headers={
                "Content-Disposition": f'inline; filename="{doc["filename"]}"',
            },
        )
    except Exception:
        raise HTTPException(status_code=404, detail="File not available for preview")


@app.post("/ui/patients/{patient_id}/report/query", response_class=HTMLResponse, include_in_schema=False)
def ui_query_report(
    request: Request,
    patient_id: str,
    query: str = Form(""),
    csrf_token: str = Form(...),
):
    """Handle RAG-powered clinical queries on the report page."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = _row_to_patient(patient_row)
    draft_row = _latest_draft(patient_id)
    report_edits = draft_row.get("report_edits") if draft_row else {}
    edited_interpretation = report_edits.get("interpretation") if isinstance(report_edits, dict) else None
    edited_by = report_edits.get("edited_by") if isinstance(report_edits, dict) else None
    
    # Handle empty query - redirect back to report
    if not query or not query.strip():
        return RedirectResponse(f"/ui/patients/{patient_id}/report", status_code=303)
    
    # Retrieve relevant chunks using RAG
    try:
        context_chunks = retrieve_top_chunks(patient_id, query.strip(), top_k=5)
    except Exception:
        context_chunks = []
    
    # Generate AI response
    try:
        result = query_chr(
            query=query.strip(),
            context_chunks=context_chunks,
            patient_name=patient.full_name,
        )
    except Exception as e:
        result = {
            "answer": f"Unable to process query. Please ensure documents are uploaded and processed. Error: {str(e)}",
            "citations": [],
            "query": query,
        }
    
    def process_citations(text):
        """Convert [#] references to clickable anchor links."""
        def replace_citation(match):
            num = match.group(1)
            return f'<a href="#cite-{num}" class="citation-link" title="View source">[{num}]</a>'
        return re.sub(r'\[(\d+)\]', replace_citation, text)

    # Process citations to make them clickable
    def process_query_citations(text):
        def replace_citation(match):
            num = match.group(1)
            return f'<a href="#query-cite-{num}" class="citation-link" title="View source">[{num}]</a>'
        return re.sub(r'\[(\d+)\]', replace_citation, text)
    
    answer_html = render_markdown(process_query_citations(result["answer"]))
    
    # Render the same report page with query results
    structured, _sources = _aggregate_structured(patient_id, tenant_id=str(user.tenant_id))
    labs = _normalize_labs(structured)
    meds = (structured.get("medications") or []) if structured else []
    diagnoses = (structured.get("diagnoses") or []) if structured else []
    if isinstance(report_edits, dict):
        if isinstance(report_edits.get("labs"), list):
            labs = report_edits.get("labs") or labs
        if isinstance(report_edits.get("diagnoses"), list):
            diagnoses = report_edits.get("diagnoses") or diagnoses
    draft_payload = _draft_payload(draft_row)
    summary = draft_payload.get("summary", "")
    citations = draft_payload.get("citations", [])
    
    if summary:
        summary_html = render_markdown(process_citations(summary))
    else:
        summary_html = ""

    edited_interpretation_html = (
        render_markdown(process_citations(edited_interpretation)) if edited_interpretation else ""
    )

    documents = _list_documents(patient_id, tenant_id=str(user.tenant_id))
    findings = _key_findings(labs)

    with get_conn() as conn:
        _log_action(
            conn,
            patient_id,
            "report.query",
            user.email,
            {"query": query.strip()[:120], "ip": request.client.host},
            tenant_id=str(user.tenant_id),
        )
        conn.commit()

    return _render_template(
        request,
        "report.html",
        {
            "user": user,
            "patient": patient,
            "draft": draft_row,
            "summary": summary,
            "summary_html": summary_html,
            "edited_interpretation_html": edited_interpretation_html,
            "edited_by": edited_by,
            "citations": citations,
            "labs": labs,
            "medications": meds,
            "diagnoses": diagnoses,
            "documents": documents,
            "findings": findings,
            # Query results
            "query_result": {
                "query": query,
                "answer_html": answer_html,
                "citations": result["citations"],
            },
        },
    )


@app.get("/ui/patients/{patient_id}/rag", response_class=HTMLResponse, include_in_schema=False)
def ui_rag_view(request: Request, patient_id: str, q: str = "", k: int = 5):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id, tenant_id=str(user.tenant_id))
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    top_k = max(1, min(k, 20))
    query = q.strip()

    if not query:
        structured, _sources = _aggregate_structured(patient_id, tenant_id=str(user.tenant_id))
        if structured:
            query_payload = dict(structured)
            query_payload.pop("documents", None)
            query = build_query(query_payload, patient_row.get("notes"))

    chunks = retrieve_top_chunks(patient_id, query, top_k=top_k) if query else []

    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "rag_view.html",
        {
            "user": user,
            "patient": _row_to_patient(patient_row),
            "chunks": chunks,
            "query": query,
            "top_k": top_k,
            "dev_mode": dev_mode,
        },
    )


@app.post("/ui/patients/{patient_id}/upload", include_in_schema=False)
async def ui_upload_document(
    request: Request,
    patient_id: str,
    files: list[UploadFile] = File(...),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    for upload in files:
        if not upload.filename:
            continue
        # Upload document
        doc = _upload_document(patient_id, upload, actor=user.email, tenant_id=str(user.tenant_id))

        if settings.job_queue_enabled:
            enqueue_job(
                "extract",
                {"document_id": doc.id, "actor": user.email, "tenant_id": str(user.tenant_id), "auto_embed": True},
                tenant_id=str(user.tenant_id),
                document_id=doc.id,
            )
        else:
            # Auto-process: Extract text from document
            try:
                _extract_document(doc.id, actor=user.email, tenant_id=str(user.tenant_id))
                # Auto-process: Generate embeddings after extraction
                try:
                    _embed_document(doc.id, actor=user.email, tenant_id=str(user.tenant_id))
                except Exception:
                    # Embedding may fail if extraction didn't produce text
                    pass
            except Exception:
                # Extraction may fail for some document types
                pass
            
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/documents/{document_id}/extract", include_in_schema=False)
def ui_extract_document(
    request: Request,
    document_id: str,
    patient_id: str = Form(...),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    if settings.job_queue_enabled:
        enqueue_job(
            "extract",
            {"document_id": document_id, "actor": user.email, "tenant_id": str(user.tenant_id)},
            tenant_id=str(user.tenant_id),
            document_id=document_id,
        )
    else:
        _extract_document(document_id, actor=user.email, tenant_id=str(user.tenant_id))
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/documents/{document_id}/embed", include_in_schema=False)
def ui_embed_document(
    request: Request,
    document_id: str,
    patient_id: str = Form(...),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    if settings.job_queue_enabled:
        enqueue_job(
            "embed",
            {"document_id": document_id, "actor": user.email, "tenant_id": str(user.tenant_id)},
            tenant_id=str(user.tenant_id),
            document_id=document_id,
        )
    else:
        _embed_document(document_id, actor=user.email, tenant_id=str(user.tenant_id))
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/patients/{patient_id}/draft", include_in_schema=False)
def ui_draft_chr(
    request: Request,
    patient_id: str,
    notes: str = Form(""),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    if settings.job_queue_enabled:
        enqueue_job(
            "draft_chr",
            {"patient_id": patient_id, "notes": notes or None, "actor": user.email, "tenant_id": str(user.tenant_id)},
            tenant_id=str(user.tenant_id),
            patient_id=patient_id,
        )
    else:
        _draft_chr(patient_id, notes or None, actor=user.email, tenant_id=str(user.tenant_id))
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/patients/{patient_id}/lifestyle", include_in_schema=False)
def ui_save_lifestyle(
    request: Request,
    patient_id: str,
    diet: str = Form(""),
    exercise: str = Form(""),
    stress: str = Form(""),
    sleep: str = Form(""),
    smoking: str = Form(""),
    alcohol: str = Form(""),
    environmental: str = Form(""),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    lifestyle_data = {
        "diet": diet,
        "exercise": exercise,
        "stress_level": stress,
        "sleep_quality": sleep,
        "smoking": smoking,
        "alcohol": alcohol,
        "environmental_exposures": environmental,
    }

    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE patients SET lifestyle = %s WHERE id = %s AND tenant_id = %s",
            (Json(lifestyle_data), patient_id, str(user.tenant_id)),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Patient not found")
        _log_action(conn, patient_id, "lifestyle.updated", user.email, lifestyle_data, tenant_id=str(user.tenant_id))
        conn.commit()

    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/patients/{patient_id}/report/save", include_in_schema=False)
def ui_save_report_edits(
    request: Request,
    patient_id: str,
    labs: str = Form(""),
    diagnoses: str = Form(""),
    interpretation: str = Form(""),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    validate_csrf_token(request, csrf_token)

    if not _get_patient(patient_id, tenant_id=str(user.tenant_id)):
        raise HTTPException(status_code=404, detail="Patient not found")

    import json
    try:
        labs_data = json.loads(labs) if labs else []
    except json.JSONDecodeError:
        labs_data = []
    try:
        diagnoses_data = json.loads(diagnoses) if diagnoses else []
    except json.JSONDecodeError:
        diagnoses_data = []

    edits = {
        "labs": labs_data,
        "diagnoses": diagnoses_data,
        "interpretation": interpretation,
        "edited_by": user.email,
    }

    with get_conn() as conn:
        # Update the latest chr_version with edits
        conn.execute(
            """
            WITH latest AS (
                SELECT id
                FROM chr_versions
                WHERE patient_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            )
            UPDATE chr_versions
            SET report_edits = %s
            FROM latest
            WHERE chr_versions.id = latest.id
            """,
            (patient_id, Json(edits)),
        )
        _log_action(conn, patient_id, "report.edited", user.email, {"fields_edited": list(edits.keys())}, tenant_id=str(user.tenant_id))
        conn.commit()

    return RedirectResponse(f"/ui/patients/{patient_id}/report", status_code=303)


@app.post("/ui/patients/{patient_id}/report/finalize", include_in_schema=False)
def ui_finalize_report(request: Request, patient_id: str, csrf_token: str = Form(...)):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    require_permission(user.role, "report.finalize")
    step_up_response = _require_step_up_for_sensitive_action(
        request,
        user,
        f"/ui/patients/{patient_id}/report/finalize",
    )
    if step_up_response:
        return step_up_response
    validate_csrf_token(request, csrf_token)

    if not _get_patient(patient_id, tenant_id=str(user.tenant_id)):
        raise HTTPException(status_code=404, detail="Patient not found")

    with get_conn() as conn:
        conn.execute(
            """
            WITH latest AS (
                SELECT id
                FROM chr_versions
                WHERE patient_id = %s
                AND status = 'draft'
                ORDER BY created_at DESC
                LIMIT 1
            )
            UPDATE chr_versions
            SET status = 'finalized', finalized_at = NOW()
            FROM latest
            WHERE chr_versions.id = latest.id
            """,
            (patient_id,),
        )
        _log_action(conn, patient_id, "report.finalized", user.email, {}, tenant_id=str(user.tenant_id))
        conn.commit()

    return RedirectResponse(f"/ui/patients/{patient_id}/report", status_code=303)
@app.get("/ui/admin", response_class=HTMLResponse, include_in_schema=False)
def ui_admin(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    if user.role != "admin":
        return RedirectResponse("/ui", status_code=303)

    with get_conn() as conn:
        users = conn.execute(
            """
            SELECT u.id, u.email, u.role, u.created_at, MAX(a.created_at) as last_active
            FROM users u
            LEFT JOIN audit_logs a ON a.actor = u.email
            WHERE u.tenant_id = %s
            GROUP BY u.id, u.email, u.role, u.created_at
            ORDER BY u.created_at DESC
            """, 
            (user.tenant_id,)
        ).fetchall()
        
        metrics = conn.execute(
            """
            SELECT 
                (SELECT COUNT(*) FROM chr_versions v JOIN patients p ON p.id = v.patient_id WHERE p.tenant_id = %s) as total_reports,
                (SELECT COUNT(*) FROM documents d JOIN patients p ON p.id = d.patient_id WHERE p.tenant_id = %s) as total_documents,
                (SELECT COUNT(*) FROM extractions e JOIN documents d ON d.id = e.document_id JOIN patients p ON p.id = d.patient_id WHERE p.tenant_id = %s) as total_extractions
            """,
            (str(user.tenant_id), str(user.tenant_id), str(user.tenant_id))
        ).fetchone()

    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "admin.html",
        {
            "user": user,
            "users": users,
            "metrics": metrics,
            "dev_mode": dev_mode,
        },
    )


@app.post("/ui/admin/users", include_in_schema=False)
def ui_create_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    csrf_token: str = Form(...)
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    require_permission(user.role, "user.manage")
    validate_csrf_token(request, csrf_token)

    pw_hash = get_password_hash(password)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, role, tenant_id) VALUES (%s, %s, %s, %s)",
                (email, pw_hash, role, user.tenant_id)
            )
            _log_action(conn, str(user.tenant_id), "user.create", user.email, {"created_email": email, "role": role}, tenant_id=str(user.tenant_id))
            conn.commit()
    except Exception:
        pass 
    
    return RedirectResponse("/ui/admin", status_code=303)


@app.post("/ui/admin/ip-allowlist", include_in_schema=False)
def ui_update_ip_allowlist(
    request: Request,
    ip_whitelist: str = Form(""),
    csrf_token: str = Form(...),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    require_permission(user.role, "user.manage")
    validate_csrf_token(request, csrf_token)

    entries = [line.strip() for line in ip_whitelist.replace(",", "\n").splitlines() if line.strip()]
    try:
        with get_conn() as conn:
            update_tenant_whitelist(str(user.tenant_id), entries, conn)
            _log_action(
                conn,
                None,
                "tenant.ip_allowlist_updated",
                user.email,
                {"entries": entries, "count": len(entries)},
                tenant_id=str(user.tenant_id),
            )
            conn.commit()
    except ValueError as exc:
        with get_conn() as conn:
            users = conn.execute(
                "SELECT id, email, role, created_at FROM users WHERE tenant_id = %s ORDER BY created_at DESC",
                (user.tenant_id,),
            ).fetchall()
            whitelist_entries = get_tenant_whitelist(str(user.tenant_id), conn)
        return _render_template(
            request,
            "admin.html",
            {
                "user": user,
                "users": users,
                "dev_mode": request.session.get("dev_mode", False),
                "ip_whitelist_text": "\n".join(whitelist_entries),
                "allowlist_error": str(exc),
            },
            status_code=400,
        )

    return RedirectResponse("/ui/admin", status_code=303)

app.add_middleware(ServerSideSessionMiddleware, **session_kwargs)
