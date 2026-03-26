"""
Data API routes extracted from main.py.

These are the programmatic API routes that require API key authentication
(dependencies=[Depends(require_api_key)]).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse

from .config import get_settings
from .db import get_conn
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
    delete_bytes,
    create_signed_download_url,
)
from .jobs import enqueue_job, get_job
from .security import (
    require_api_key,
    require_read_scope,
    require_write_scope,
)
from .authz import require_tenant_id
from .helpers import (
    _row_to_patient,
    _log_action,
    _upload_document,
    _issue_signed_upload,
    _register_signed_upload,
    _extract_document,
    _embed_document,
    _draft_chr,
)

settings = get_settings()

router = APIRouter(tags=["API"])


@router.get(
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


@router.post(
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


@router.get(
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


@router.delete(
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


@router.post(
    "/patients/{patient_id}/documents",
    response_model=Document,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
async def upload_document(request: Request, patient_id: str, file: UploadFile = File(...)):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    doc = _upload_document(patient_id, file, actor=actor, tenant_id=tenant_id)
    return doc


@router.post(
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
    )


@router.post(
    "/patients/{patient_id}/documents/register-upload",
    response_model=Document,
    dependencies=[Depends(require_api_key), Depends(require_write_scope)],
)
def register_document_upload(request: Request, patient_id: str, payload: SignedUploadRegistration):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    return _register_signed_upload(patient_id, payload, actor=actor, tenant_id=tenant_id)


@router.get(
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


@router.delete(
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


@router.post(
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


@router.post(
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


@router.post(
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
