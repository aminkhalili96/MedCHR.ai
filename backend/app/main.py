from pathlib import Path
from uuid import uuid4
import logging
import re
import secrets
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from psycopg.types.json import Json
from starlette.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from .config import get_settings
from .db import get_conn, close_pool, clear_tenant_context
from .schemas import (
    PatientCreate,
    Patient,
    Document,
    SignedUploadRequest,
    SignedUploadRegistration,
    SignedUploadResponse,
    SignedDownloadResponse,
    ExtractionResult,
    ChrDraftRequest,
    ChrDraft,
    JobStatus,
    EmbedResult,
)
from .storage import (
    upload_bytes_via_signed_url,
    download_bytes,
    ensure_bucket,
    storage_health,
    delete_bytes,
    create_signed_upload_url,
    create_signed_download_url,
)
from .ocr import extract_text
from .extract import extract_structured
from .embeddings import embed_texts
from .rag import build_query, retrieve_top_chunks, retrieve_hybrid
from .chr import generate_chr_draft, query_chr
from .jobs import enqueue_job, list_jobs, get_job
from .observability import metrics, record_request
from . import clinical
from . import gap_features
from .security import (
    require_api_key,
    require_read_scope,
    require_write_scope,
    get_csrf_token,
    validate_csrf_token,
    render_markdown,
    validate_production_settings,
    allowed_hosts,
    cors_origins,
)
from .auth import authenticate_user, get_current_user, User, require_admin, get_password_hash
from .audit_events import append_audit_event
from .authz import (
    require_tenant_id,
    require_permission,
    mark_step_up_verified,
    is_step_up_verified,
)
from .phi import ensure_phi_processor
from .server_session import ServerSideSessionMiddleware, renew_session
from .sso import configure_sso, handle_sso_callback, initiate_sso_login, provision_sso_user
from .uploads import read_upload_bytes, sanitize_filename, resolve_content_type
from .ip_whitelist import get_tenant_whitelist, update_tenant_whitelist

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

session_kwargs = {
    "secret_key": settings.app_secret_key,
    "session_cookie": settings.session_cookie_name,
    "max_age": settings.session_max_age_minutes * 60,
    "same_site": "strict" if settings.app_env == "prod" or settings.hipaa_mode else "lax",
    "https_only": settings.app_env == "prod" or settings.hipaa_mode,
}
app.add_middleware(ServerSideSessionMiddleware, **session_kwargs)

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

app.include_router(clinical.router)
app.include_router(gap_features.router)


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
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def session_timeout_middleware(request: Request, call_next):
    """
    Enforce session timeout for HIPAA compliance.
    Sessions expire after 15 minutes of inactivity.
    """
    import time
    
    # Session timeout policy applies to UI surface only.
    if not request.url.path.startswith("/ui"):
        return await call_next(request)
    
    session = request.session
    current_time = int(time.time())
    last_activity = session.get("_last_activity", 0)
    
    # Check for timeout (15 minutes = 900 seconds)
    SESSION_TIMEOUT = 900
    
    if last_activity and (current_time - last_activity) > SESSION_TIMEOUT:
        # Session expired - clear and redirect to login for UI routes
        session.clear()
        if request.url.path.startswith("/ui") and request.url.path != "/ui/login":
            from starlette.responses import RedirectResponse
            return RedirectResponse("/ui/login?timeout=1", status_code=303)
    
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
    ensure_bucket(settings.storage_bucket)


@app.on_event("shutdown")
def shutdown() -> None:
    close_pool()


def _row_to_patient(row) -> Patient:
    return Patient(
        id=str(row["id"]),
        full_name=row["full_name"],
        dob=row.get("dob"),
        notes=row.get("notes"),
        lifestyle=row.get("lifestyle") or {},
        genetics=row.get("genetics") or {},
    )


def _row_to_document(row) -> Document:
    return Document(
        id=str(row["id"]),
        patient_id=str(row["patient_id"]),
        filename=row["filename"],
        content_type=row["content_type"],
        storage_path=row["storage_path"],
    )


def _log_action(conn, patient_id: str | None, action: str, actor: str, details: dict | None = None, tenant_id: str | None = None):
    if action.startswith("patient."):
        resource_type = "patient"
    elif action.startswith("document.") or action.startswith("storage."):
        resource_type = "document"
    elif action.startswith("report.") or action.startswith("chr."):
        resource_type = "chr"
    elif action.startswith("auth.") or action.startswith("user."):
        resource_type = "user"
    else:
        resource_type = "system"

    try:
        append_audit_event(
            conn,
            action=action,
            resource_type=resource_type,
            resource_id=patient_id,
            details=details or {},
            tenant_id=tenant_id,
            actor=actor,
        )
    except Exception:
        # Keep legacy audit_logs write path available for older schemas.
        pass
    conn.execute(
        """
        INSERT INTO audit_logs (patient_id, actor, action, details, tenant_id)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (patient_id, actor, action, Json(details) if details else None, tenant_id),
    )


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


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    dependencies=[Depends(require_api_key), Depends(require_read_scope)],
)
def job_status(request: Request, job_id: str):
    tenant_id = require_tenant_id(request)
    actor = getattr(request.state, "actor", "api")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tenant_id and job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tenant_id:
        return JobStatus(job_id=job.id, status=job.status)
    with get_conn() as conn:
        allowed = None
        if job.patient_id:
            allowed = conn.execute(
                "SELECT 1 FROM patients WHERE id = %s AND tenant_id = %s",
                (job.patient_id, tenant_id),
            ).fetchone()
        elif job.document_id:
            allowed = conn.execute(
                """
                SELECT 1
                FROM documents d
                JOIN patients p ON p.id = d.patient_id
                WHERE d.id = %s AND p.tenant_id = %s
                """,
                (job.document_id, tenant_id),
            ).fetchone()
        if not allowed:
            raise HTTPException(status_code=404, detail="Job not found")
        _log_action(
            conn,
            str(job.patient_id) if job.patient_id else None,
            "job.status_view",
            actor,
            {"job_id": job.id, "status": job.status},
            tenant_id=tenant_id,
        )
        conn.commit()
        return JobStatus(job_id=job.id, status=job.status)
    with get_conn() as conn:
        _log_action(
            conn,
            str(job.patient_id) if job.patient_id else None,
            "job.status_view",
            actor,
            {"job_id": job.id, "status": job.status},
            tenant_id=tenant_id,
        )
        conn.commit()
    return JobStatus(job_id=job.id, status=job.status)


@app.post(
    "/patients",
    response_model=Patient,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def create_patient(payload: PatientCreate, request: Request):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO patients (tenant_id, full_name, dob, notes)
            VALUES (%s, %s, %s, %s)
            RETURNING id, full_name, dob, notes, lifestyle, genetics
            """,
            (tenant_id, payload.full_name, payload.dob, payload.notes),
        ).fetchone()
        _log_action(conn, str(row["id"]), "patient.create", actor, {"name": payload.full_name}, tenant_id=tenant_id)
        conn.commit()
    return _row_to_patient(row)


@app.get(
    "/patients",
    response_model=list[Patient],
    dependencies=[Depends(require_api_key), Depends(require_read_scope)],
)
def list_patients(request: Request):
    tenant_id = require_tenant_id(request)
    actor = getattr(request.state, "actor", "api")
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, full_name, dob, notes, lifestyle, genetics
            FROM patients
            WHERE tenant_id = %s
            ORDER BY created_at DESC
            """,
            (tenant_id,),
        ).fetchall()
        _log_action(
            conn,
            None,
            "patient.list",
            actor,
            {"count": len(rows)},
            tenant_id=tenant_id,
        )
        conn.commit()
    return [_row_to_patient(r) for r in rows]


@app.delete(
    "/patients/{patient_id}",
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def delete_patient(request: Request, patient_id: str):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        patient = conn.execute(
            "SELECT id FROM patients WHERE id = %s AND tenant_id = %s",
            (patient_id, tenant_id),
        ).fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        rows = conn.execute(
            "SELECT storage_path FROM documents WHERE patient_id = %s",
            (patient_id,),
        ).fetchall()
        paths = [row["storage_path"] for row in rows]
        _log_action(conn, patient_id, "patient.delete", actor, {"files": len(paths)}, tenant_id=tenant_id)
        conn.execute("DELETE FROM patients WHERE id = %s AND tenant_id = %s", (patient_id, tenant_id))
        conn.commit()

    if paths:
        try:
            delete_bytes(settings.storage_bucket, paths)
        except Exception as exc:
            with get_conn() as conn:
                _log_action(
                    conn,
                    None,
                    "storage.delete_failed",
                    actor,
                    {"patient_id": patient_id, "error": str(exc)},
                    tenant_id=tenant_id,
                )
                conn.commit()
            raise
    return {"status": "deleted", "patient_id": patient_id, "files_deleted": len(paths)}


@app.post(
    "/patients/{patient_id}/documents",
    response_model=Document,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
async def upload_document(request: Request, patient_id: str, file: UploadFile = File(...)):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    doc = _upload_document(patient_id, file, actor=actor, tenant_id=tenant_id)
    return doc


@app.post(
    "/patients/{patient_id}/documents/presign-upload",
    response_model=SignedUploadResponse,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def create_document_upload_url(request: Request, patient_id: str, payload: SignedUploadRequest):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    return _issue_signed_upload(
        patient_id,
        payload.filename,
        payload.content_type,
        actor=actor,
        tenant_id=tenant_id,
        expires_in_seconds=payload.expires_in_seconds,
    )


@app.post(
    "/patients/{patient_id}/documents/register-upload",
    response_model=Document,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def register_document_upload(request: Request, patient_id: str, payload: SignedUploadRegistration):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    return _register_signed_upload(patient_id, payload, actor=actor, tenant_id=tenant_id)


@app.get(
    "/documents/{document_id}/download-url",
    response_model=SignedDownloadResponse,
    dependencies=[Depends(require_api_key), Depends(require_read_scope)],
)
def document_download_url(request: Request, document_id: str, expires_in_seconds: int = 300):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    ttl = max(60, min(expires_in_seconds, 3600))
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT d.id, d.patient_id, d.storage_path
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            WHERE d.id = %s AND p.tenant_id = %s
            """,
            (document_id, tenant_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        download_url = create_signed_download_url(settings.storage_bucket, row["storage_path"], ttl)
        _log_action(
            conn,
            str(row["patient_id"]),
            "document.download_url_issued",
            actor,
            {"document_id": document_id, "expires_in_seconds": ttl},
            tenant_id=tenant_id,
        )
        conn.commit()
    return SignedDownloadResponse(
        document_id=str(row["id"]),
        storage_path=row["storage_path"],
        download_url=download_url,
        expires_in_seconds=ttl,
    )


@app.delete(
    "/documents/{document_id}",
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def delete_document(request: Request, document_id: str):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        doc = conn.execute(
            """
            SELECT id, patient_id, storage_path
            FROM documents
            WHERE id = %s
            """,
            (document_id,),
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        allowed = conn.execute(
            "SELECT 1 FROM patients WHERE id = %s AND tenant_id = %s",
            (str(doc["patient_id"]), tenant_id),
        ).fetchone()
        if not allowed:
            raise HTTPException(status_code=404, detail="Document not found")
        _log_action(
            conn,
            str(doc["patient_id"]),
            "document.delete",
            actor,
            {"document_id": document_id},
            tenant_id=tenant_id,
        )
        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
        conn.commit()

    try:
        delete_bytes(settings.storage_bucket, [doc["storage_path"]])
    except Exception as exc:
        with get_conn() as conn:
            _log_action(
                conn,
                None,
                "storage.delete_failed",
                actor,
                {"document_id": document_id, "error": str(exc)},
                tenant_id=tenant_id,
            )
            conn.commit()
        raise
    return {"status": "deleted", "document_id": document_id}


def _upload_document(patient_id: str, file: UploadFile, actor: str = "system", tenant_id: str | None = None) -> Document:
    filename = sanitize_filename(getattr(file, "filename", "upload.bin"))
    if hasattr(file, "file"):
        data, content_type, _size = read_upload_bytes(file)
    else:
        data = file
        content_type = resolve_content_type(filename, None)

    with get_conn() as conn:
        query = "SELECT id FROM patients WHERE id = %s"
        params: list[str] = [patient_id]
        if tenant_id:
            query += " AND tenant_id = %s"
            params.append(tenant_id)
        patient = conn.execute(query, tuple(params)).fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        storage_path = f"{patient_id}/{uuid4()}_{filename}"
        upload_bytes_via_signed_url(settings.storage_bucket, storage_path, data, content_type)

        row = conn.execute(
            """
            INSERT INTO documents (patient_id, filename, content_type, storage_path)
            VALUES (%s, %s, %s, %s)
            RETURNING id, patient_id, filename, content_type, storage_path
            """,
            (patient_id, filename, content_type, storage_path),
        ).fetchone()
        _log_action(
            conn,
            patient_id,
            "document.upload",
            actor,
            {"document_id": str(row["id"])},
            tenant_id=tenant_id,
        )
        conn.commit()

    return _row_to_document(row)


def _validate_storage_path_for_patient(patient_id: str, storage_path: str) -> str:
    path = storage_path.strip().lstrip("/")
    if ".." in path:
        raise HTTPException(status_code=400, detail="Invalid storage path")
    expected_prefix = f"{patient_id}/"
    if not path.startswith(expected_prefix):
        raise HTTPException(status_code=400, detail="Storage path does not belong to patient")
    return path


def _issue_signed_upload(
    patient_id: str,
    filename: str,
    content_type: str | None,
    *,
    actor: str,
    tenant_id: str,
    expires_in_seconds: int,
) -> SignedUploadResponse:
    safe_filename = sanitize_filename(filename)
    resolved_content_type = resolve_content_type(safe_filename, content_type)
    with get_conn() as conn:
        patient = conn.execute(
            "SELECT id FROM patients WHERE id = %s AND tenant_id = %s",
            (patient_id, tenant_id),
        ).fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

    storage_path = f"{patient_id}/{uuid4()}_{safe_filename}"
    signed = create_signed_upload_url(settings.storage_bucket, storage_path)
    resolved_path = _validate_storage_path_for_patient(patient_id, signed.get("path") or storage_path)
    with get_conn() as conn:
        _log_action(
            conn,
            patient_id,
            "document.upload_presigned_issued",
            actor,
            {"storage_path": resolved_path, "expires_in_seconds": expires_in_seconds},
            tenant_id=tenant_id,
        )
        conn.commit()

    return SignedUploadResponse(
        patient_id=patient_id,
        filename=safe_filename,
        content_type=resolved_content_type,
        storage_path=resolved_path,
        upload_url=str(signed["upload_url"]),
        upload_token=str(signed["token"]),
        expires_in_seconds=expires_in_seconds,
    )


def _register_signed_upload(
    patient_id: str,
    payload: SignedUploadRegistration,
    *,
    actor: str,
    tenant_id: str,
) -> Document:
    filename = sanitize_filename(payload.filename)
    content_type = resolve_content_type(filename, payload.content_type)
    storage_path = _validate_storage_path_for_patient(patient_id, payload.storage_path)
    with get_conn() as conn:
        patient = conn.execute(
            "SELECT id FROM patients WHERE id = %s AND tenant_id = %s",
            (patient_id, tenant_id),
        ).fetchone()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        row = conn.execute(
            """
            INSERT INTO documents (patient_id, filename, content_type, storage_path)
            VALUES (%s, %s, %s, %s)
            RETURNING id, patient_id, filename, content_type, storage_path
            """,
            (patient_id, filename, content_type, storage_path),
        ).fetchone()
        _log_action(
            conn,
            patient_id,
            "document.upload_registered",
            actor,
            {"document_id": str(row["id"]), "storage_path": storage_path},
            tenant_id=tenant_id,
        )
        conn.commit()
    return _row_to_document(row)


@app.post(
    "/documents/{document_id}/extract",
    response_model=ExtractionResult | JobStatus,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def extract_document(request: Request, document_id: str, async_process: bool = False):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        allowed = conn.execute(
            """
            SELECT 1
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            WHERE d.id = %s AND p.tenant_id = %s
            """,
            (document_id, tenant_id),
        ).fetchone()
    if not allowed:
        raise HTTPException(status_code=404, detail="Document not found")
    if settings.job_queue_enabled or async_process:
        job_id = enqueue_job(
            "extract",
            {"document_id": document_id, "actor": actor, "tenant_id": tenant_id},
            tenant_id=tenant_id,
            document_id=document_id,
        )
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=status.HTTP_202_ACCEPTED)
    return _extract_document(document_id, actor=actor, tenant_id=tenant_id)


def _extract_document(document_id: str, actor: str = "system", tenant_id: str | None = None) -> ExtractionResult:
    with get_conn() as conn:
        doc = conn.execute(
            """
            SELECT id, patient_id, storage_path, content_type
            FROM documents
            WHERE id = %s
            """,
            (document_id,),
        ).fetchone()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if tenant_id:
        with get_conn() as conn:
            allowed = conn.execute(
                "SELECT 1 FROM patients WHERE id = %s AND tenant_id = %s",
                (str(doc["patient_id"]), tenant_id),
            ).fetchone()
        if not allowed:
            raise HTTPException(status_code=404, detail="Document not found")

    data = download_bytes(settings.storage_bucket, doc["storage_path"])
    raw_text = extract_text(data, doc["content_type"])
    # structured is a dict (ExtractionData().dict())
    structured = extract_structured(raw_text)

    with get_conn() as conn:
        # 1. Insert into extractions (Legacy/Backup JSONB)
        extraction_row = conn.execute(
            """
            INSERT INTO extractions (document_id, raw_text, structured)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (document_id, raw_text, Json(structured)),
        ).fetchone()
        extraction_id = extraction_row["id"]

        # 2. Insert into structured tables
        patient_id = doc["patient_id"]
        
        # Labs
        if "labs" in structured and structured["labs"]:
            for lab in structured["labs"]:
                conn.execute(
                    """
                    INSERT INTO lab_results 
                    (patient_id, extraction_id, test_name, value, unit, flag, reference_range, test_date, panel)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        patient_id, extraction_id, 
                        lab.get("test_name"), lab.get("value"), lab.get("unit"), 
                        lab.get("flag"), lab.get("reference_range"), 
                        lab.get("date"), lab.get("panel")
                    )
                )

        # Medications
        if "medications" in structured and structured["medications"]:
            for med in structured["medications"]:
                conn.execute(
                    """
                    INSERT INTO medications
                    (patient_id, extraction_id, name, dosage, frequency, route, start_date, end_date, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        patient_id, extraction_id,
                        med.get("name"), med.get("dosage"), med.get("frequency"),
                        med.get("route"), med.get("start_date"), med.get("end_date"),
                        med.get("status", "active")
                    )
                )

        # Diagnoses
        if "diagnoses" in structured and structured["diagnoses"]:
            for dx in structured["diagnoses"]:
                conn.execute(
                    """
                    INSERT INTO diagnoses
                    (patient_id, extraction_id, condition, code, status, date_onset)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        patient_id, extraction_id,
                        dx.get("condition"), dx.get("code"), dx.get("status"),
                        dx.get("date_onset")
                    )
                )

        _log_action(
            conn,
            str(doc["patient_id"]),
            "document.extract",
            actor,
            {"document_id": document_id},
            tenant_id=tenant_id,
        )
        conn.commit()

    # Re-validate to return Pydantic model (ExtractionResult expects structured as ExtractionData)
    # Since schemas were updated, we need to ensure this return value matches.
    # structured is a dict, ExtractionResult.structured is ExtractionData type.
    # Pydantic should auto-cast dict to model if passed to constructor.
    # But currently the return type is ExtractionResult.
    
    from .schemas import ExtractionData # Import locally to avoid circulars if any
    
    return ExtractionResult(
        document_id=document_id,
        raw_text=raw_text,
        structured=ExtractionData(**structured),
    )


def _chunk_text(text: str):
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    chunk_index = 0
    while start < length:
        end = min(start + size, length)
        raw_chunk = text[start:end]
        if end < length:
            last_space = raw_chunk.rfind(" ")
            if last_space > 0 and last_space > size * 0.6:
                end = start + last_space
                raw_chunk = text[start:end]
        leading_ws = len(raw_chunk) - len(raw_chunk.lstrip())
        trailing_ws = len(raw_chunk) - len(raw_chunk.rstrip())
        chunk = raw_chunk.strip()
        if chunk:
            chunk_start = start + leading_ws
            chunk_end = end - trailing_ws
            chunks.append(
                {
                    "chunk_text": chunk,
                    "chunk_index": chunk_index,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                }
            )
            chunk_index += 1
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


@app.post(
    "/documents/{document_id}/embed",
    response_model=EmbedResult | JobStatus,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def embed_document(request: Request, document_id: str, async_process: bool = False):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        allowed = conn.execute(
            """
            SELECT 1
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            WHERE d.id = %s AND p.tenant_id = %s
            """,
            (document_id, tenant_id),
        ).fetchone()
    if not allowed:
        raise HTTPException(status_code=404, detail="Document not found")
    if settings.job_queue_enabled or async_process:
        job_id = enqueue_job(
            "embed",
            {"document_id": document_id, "actor": actor, "tenant_id": tenant_id},
            tenant_id=tenant_id,
            document_id=document_id,
        )
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=status.HTTP_202_ACCEPTED)
    return _embed_document(document_id, actor=actor, tenant_id=tenant_id)


def _embed_document(document_id: str, actor: str = "system", tenant_id: str | None = None):
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT e.id as extraction_id, e.raw_text, d.patient_id
            FROM extractions e
            JOIN documents d ON d.id = e.document_id
            JOIN patients p ON p.id = d.patient_id
            WHERE e.document_id = %s
              AND (%s IS NULL OR p.tenant_id = %s)
            ORDER BY e.created_at DESC
            LIMIT 1
            """,
            (document_id, tenant_id, tenant_id),
        ).fetchone()

    if not row or not row.get("raw_text"):
        raise HTTPException(status_code=404, detail="No extraction found for document")

    chunks = _chunk_text(row["raw_text"])
    if not chunks:
        raise HTTPException(status_code=400, detail="No text available for embedding")
    vectors = embed_texts([chunk["chunk_text"] for chunk in chunks])

    with get_conn() as conn:
        conn.execute("DELETE FROM embeddings WHERE document_id = %s", (document_id,))
        for chunk, vector in zip(chunks, vectors, strict=False):
            conn.execute(
                """
                INSERT INTO embeddings (document_id, extraction_id, chunk_index, chunk_start, chunk_end, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    row["extraction_id"],
                    chunk["chunk_index"],
                    chunk["chunk_start"],
                    chunk["chunk_end"],
                    chunk["chunk_text"],
                    vector,
                ),
            )
        _log_action(
            conn,
            str(row["patient_id"]),
            "document.embed",
            actor,
            {"document_id": document_id},
            tenant_id=tenant_id,
        )
        conn.commit()

    return {"document_id": document_id, "chunks": len(chunks)}


@app.post(
    "/chr/draft",
    response_model=ChrDraft | JobStatus,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def draft_chr(request: Request, payload: ChrDraftRequest, async_process: bool = False):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        allowed = conn.execute(
            "SELECT 1 FROM patients WHERE id = %s AND tenant_id = %s",
            (payload.patient_id, tenant_id),
        ).fetchone()
    if not allowed:
        raise HTTPException(status_code=404, detail="Patient not found")
    if settings.job_queue_enabled or async_process:
        job_id = enqueue_job(
            "draft_chr",
            {"patient_id": payload.patient_id, "notes": payload.notes, "actor": actor, "tenant_id": tenant_id},
            tenant_id=tenant_id,
            patient_id=payload.patient_id,
        )
        return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=status.HTTP_202_ACCEPTED)
    return _draft_chr(payload.patient_id, payload.notes, actor=actor, tenant_id=tenant_id)


def _draft_chr(patient_id: str, notes: str | None, actor: str = "system", tenant_id: str | None = None) -> ChrDraft:
    structured, _sources = _aggregate_structured(patient_id, tenant_id=tenant_id)
    if not structured:
        raise HTTPException(status_code=404, detail="No extraction found for patient")

    query_payload = dict(structured)
    query_payload.pop("documents", None)
    query = build_query(query_payload, notes)
    context_chunks = retrieve_hybrid(patient_id, query, top_k=5)
    draft = generate_chr_draft(query_payload, notes, context_chunks)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO chr_versions (patient_id, draft, status)
            VALUES (%s, %s, %s)
            """,
            (patient_id, Json(draft), "draft"),
        )
        _log_action(conn, patient_id, "chr.draft", actor, {"chunks": len(context_chunks)}, tenant_id=tenant_id)
        conn.commit()

    return ChrDraft(
        patient_id=patient_id,
        draft=draft,
        citations=draft.get("citations", []),
    )


# -------------------- UI --------------------

def _require_ui_user(request: Request) -> User | None:
    try:
        return get_current_user(request)
    except HTTPException:
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
    if status_code is None:
        return templates.TemplateResponse(template, payload)
    return templates.TemplateResponse(template, payload, status_code=status_code)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui", status_code=302)


@app.get("/ui/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request):
    return _render_template(request, "login.html", {"sso_enabled": _sso_enabled()})


@limiter.limit("10/minute")
@app.post("/ui/login", response_class=HTMLResponse, include_in_schema=False)
def login(
    request: Request,
    username: str = Form(..., alias="email"), # Using email as username
    password: str = Form(...),
    csrf_token: str = Form(...),
):
    validate_csrf_token(request, csrf_token)
    user = authenticate_user(username, password)
    if user:
        request.session.clear()
        renew_session(request)
        request.session.pop("step_up_verified_at", None)
        if user.get("mfa_enabled"):
            request.session["partial_auth_user_id"] = str(user["id"])
            return _render_template(request, "mfa_challenge.html", {"email": username})
            
        request.session["user_id"] = str(user["id"])
        with get_conn() as conn:
            _log_action(conn, None, "auth.login_success", user["email"], {"ip": request.client.host}, tenant_id=str(user["tenant_id"]))
            conn.commit()
        return RedirectResponse("/ui", status_code=303)
    
    # ... logging logic ...
    return _render_template(
        request,
        "login.html",
        {"error": "Invalid credentials", "email": username, "sso_enabled": _sso_enabled()},
        status_code=401,
    )

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
def ui_patients(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, full_name, dob, notes FROM patients WHERE tenant_id = %s ORDER BY created_at DESC",
            (user.tenant_id,)
        ).fetchall()

    patients = [_row_to_patient(r) for r in rows]
    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "patients.html",
        {"patients": patients, "user": user, "dev_mode": dev_mode},
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
        return conn.execute(
            """
            SELECT d.id, d.patient_id, d.filename, d.content_type, d.storage_path
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            WHERE d.patient_id = %s
              AND (%s IS NULL OR p.tenant_id = %s)
            ORDER BY d.created_at DESC
            """,
            (patient_id, tenant_id, tenant_id),
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
        return conn.execute(
            """
            SELECT e.raw_text, e.structured
            FROM extractions e
            JOIN documents d ON d.id = e.document_id
            JOIN patients p ON p.id = d.patient_id
            WHERE d.patient_id = %s
              AND (%s IS NULL OR p.tenant_id = %s)
            ORDER BY e.created_at DESC
            LIMIT 1
            """,
            (patient_id, tenant_id, tenant_id),
        ).fetchone()


def _aggregate_structured(patient_id: str, tenant_id: str | None = None) -> tuple[dict | None, list[dict]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                d.id as document_id,
                d.filename,
                d.content_type,
                d.created_at as document_created_at,
                e.id as extraction_id,
                e.structured,
                e.raw_text,
                e.created_at as extracted_at
            FROM documents d
            JOIN patients p ON p.id = d.patient_id
            JOIN LATERAL (
                SELECT id, structured, raw_text, created_at
                FROM extractions
                WHERE document_id = d.id
                ORDER BY created_at DESC
                LIMIT 1
            ) e ON true
            WHERE d.patient_id = %s
              AND (%s IS NULL OR p.tenant_id = %s)
            ORDER BY d.created_at DESC
            """,
            (patient_id, tenant_id, tenant_id),
        ).fetchall()

    if not rows:
        return None, []

    labs: list[dict] = []
    diagnoses: list[str] = []
    medications: list[str] = []
    procedures: list[str] = []
    genetics: list[dict] = []
    notes_parts: list[str] = []
    sources: list[dict] = []

    seen_labs: set[tuple] = set()
    seen_dx: set[str] = set()
    seen_meds: set[str] = set()
    seen_proc: set[str] = set()
    seen_gen: set[tuple] = set()

    for row in rows:
        structured = row.get("structured") or {}
        sources.append(
            {
                "document_id": str(row["document_id"]),
                "filename": row["filename"],
                "content_type": row["content_type"],
                "extraction_id": str(row["extraction_id"]),
                "extracted_at": row["extracted_at"].isoformat() if row.get("extracted_at") else None,
            }
        )

        for lab in structured.get("labs") or structured.get("biomarkers") or []:
            if not isinstance(lab, dict):
                continue
            key = (
                lab.get("panel"),
                lab.get("test"),
                lab.get("value"),
                lab.get("unit"),
                lab.get("range"),
                lab.get("flag"),
            )
            if key in seen_labs:
                continue
            seen_labs.add(key)
            labs.append(lab)

        for dx in structured.get("diagnoses") or []:
            if not isinstance(dx, str):
                continue
            key = dx.strip().lower()
            if not key or key in seen_dx:
                continue
            seen_dx.add(key)
            diagnoses.append(dx)

        for med in structured.get("medications") or []:
            if not isinstance(med, str):
                continue
            key = med.strip().lower()
            if not key or key in seen_meds:
                continue
            seen_meds.add(key)
            medications.append(med)

        for proc in structured.get("procedures") or []:
            if not isinstance(proc, str):
                continue
            key = proc.strip().lower()
            if not key or key in seen_proc:
                continue
            seen_proc.add(key)
            procedures.append(proc)

        for gene in structured.get("genetics") or []:
            if not isinstance(gene, dict):
                continue
            key = (gene.get("gene"), gene.get("variant"), gene.get("impact"))
            if key in seen_gen:
                continue
            seen_gen.add(key)
            genetics.append(gene)

        note = structured.get("notes") or ""
        if note:
            notes_parts.append(f"{row['filename']}: {note}")

    combined_notes = "\n".join(notes_parts)
    if len(combined_notes) > settings.aggregate_notes_max_chars:
        combined_notes = combined_notes[: settings.aggregate_notes_max_chars] + "…"

    aggregated = {
        "labs": labs,
        "biomarkers": labs,
        "diagnoses": diagnoses,
        "medications": medications,
        "procedures": procedures,
        "genetics": genetics,
        "notes": combined_notes,
        "documents": sources,
    }
    return aggregated, sources


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
                "test": lab.get("test"),
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
        test = lab.get("test") or "Unknown"
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

    documents = [_row_to_document(r) for r in _list_documents(patient_id)]
    draft = _latest_draft(patient_id)
    logs = _audit_logs(patient_id)
    has_extractions = _has_extractions(patient_id)
    dev_mode = request.session.get("dev_mode", False)
    pending_jobs = (
        list_jobs(patient_id=patient_id, tenant_id=str(user.tenant_id), statuses=["pending", "running"])
        if settings.job_queue_enabled
        else []
    )

    with get_conn() as conn:
        _log_action(conn, patient_id, "patient.view", user.email, {"ip": request.client.host}, tenant_id=str(user.tenant_id))
        conn.commit()

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

    structured, _sources = _aggregate_structured(patient_id, tenant_id=str(user.tenant_id))
    labs = _normalize_labs(structured)
    meds = structured.get("medications") if structured else []
    diagnoses = structured.get("diagnoses") if structured else []

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

    documents = _list_documents(patient_id, tenant_id=str(user.tenant_id))
    findings = _key_findings(labs)

    with get_conn() as conn:
        _log_action(conn, patient_id, "report.view", user.email, {"ip": request.client.host}, tenant_id=str(user.tenant_id))
        conn.commit()

    return _render_template(
        request,
        "report.html",
        {
            "user": user,
            "patient": _row_to_patient(patient_row),
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
        },
    )


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
    diagnoses = structured.get("diagnoses") if structured else []
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
    meds = structured.get("medications") if structured else []
    diagnoses = structured.get("diagnoses") if structured else []
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
            "SELECT id, email, role, created_at FROM users WHERE tenant_id = %s ORDER BY created_at DESC", 
            (user.tenant_id,)
        ).fetchall()
        whitelist_entries = get_tenant_whitelist(str(user.tenant_id), conn)

    dev_mode = request.session.get("dev_mode", False)
    return _render_template(
        request,
        "admin.html",
        {
            "user": user,
            "users": users,
            "dev_mode": dev_mode,
            "ip_whitelist_text": "\n".join(whitelist_entries),
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
