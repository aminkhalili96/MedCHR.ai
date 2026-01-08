from pathlib import Path
from uuid import uuid4
import re

import markdown
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from psycopg.types.json import Json
from starlette.middleware.sessions import SessionMiddleware

from .config import get_settings
from .db import get_conn
from .schemas import (
    PatientCreate,
    Patient,
    Document,
    ExtractionResult,
    ChrDraftRequest,
    ChrDraft,
)
from .storage import upload_bytes, download_bytes, ensure_bucket
from .ocr import extract_text
from .extract import extract_structured
from .embeddings import embed_texts
from .rag import build_query, retrieve_top_chunks
from .chr import generate_chr_draft, query_chr

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="MedCHR API")
settings = get_settings()
app.add_middleware(SessionMiddleware, secret_key=settings.app_secret_key)


@app.on_event("startup")
def startup() -> None:
    # Ensure the storage bucket exists for uploads
    ensure_bucket(settings.storage_bucket)


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


def _log_action(conn, patient_id: str | None, action: str, actor: str, details: dict | None = None):
    conn.execute(
        """
        INSERT INTO audit_logs (patient_id, actor, action, details)
        VALUES (%s, %s, %s, %s)
        """,
        (patient_id, actor, action, Json(details) if details else None),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/patients", response_model=Patient)
def create_patient(payload: PatientCreate):
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO patients (full_name, dob, notes)
            VALUES (%s, %s, %s)
            RETURNING id, full_name, dob, notes
            """,
            (payload.full_name, payload.dob, payload.notes),
        ).fetchone()
        _log_action(conn, str(row["id"]), "patient.create", "api", {"name": payload.full_name})
        conn.commit()
    return _row_to_patient(row)


@app.get("/patients", response_model=list[Patient])
def list_patients():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, full_name, dob, notes FROM patients ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_patient(r) for r in rows]


@app.post("/patients/{patient_id}/documents", response_model=Document)
async def upload_document(patient_id: str, file: UploadFile = File(...)):
    doc = await _upload_document(patient_id, file, actor="api")
    return doc


def _upload_document(patient_id: str, file: UploadFile, actor: str = "system") -> Document:
    if hasattr(file, "file"):
        file.file.seek(0)
        data = file.file.read()
    else:
        data = file
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    storage_path = f"{patient_id}/{uuid4()}_{file.filename}"
    upload_bytes(settings.storage_bucket, storage_path, data, file.content_type)

    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (patient_id, filename, content_type, storage_path)
            VALUES (%s, %s, %s, %s)
            RETURNING id, patient_id, filename, content_type, storage_path
            """,
            (patient_id, file.filename, file.content_type, storage_path),
        ).fetchone()
        _log_action(conn, patient_id, "document.upload", actor, {"document_id": str(row["id"])})
        conn.commit()

    return _row_to_document(row)


@app.post("/documents/{document_id}/extract", response_model=ExtractionResult)
def extract_document(document_id: str):
    return _extract_document(document_id, actor="api")


def _extract_document(document_id: str, actor: str = "system") -> ExtractionResult:
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

    data = download_bytes(settings.storage_bucket, doc["storage_path"])
    raw_text = extract_text(data, doc["content_type"])
    structured = extract_structured(raw_text)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO extractions (document_id, raw_text, structured)
            VALUES (%s, %s, %s)
            """,
            (document_id, raw_text, Json(structured)),
        )
        _log_action(conn, str(doc["patient_id"]), "document.extract", actor, {"document_id": document_id})
        conn.commit()

    return ExtractionResult(
        document_id=document_id,
        raw_text=raw_text,
        structured=structured,
    )


def _chunk_text(text: str, size: int = 1000):
    return [text[i : i + size] for i in range(0, len(text), size) if text[i : i + size]]


@app.post("/documents/{document_id}/embed")
def embed_document(document_id: str):
    return _embed_document(document_id, actor="api")


def _embed_document(document_id: str, actor: str = "system"):
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT e.raw_text, d.patient_id
            FROM extractions e
            JOIN documents d ON d.id = e.document_id
            WHERE e.document_id = %s
            ORDER BY e.created_at DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()

    if not row or not row.get("raw_text"):
        raise HTTPException(status_code=404, detail="No extraction found for document")

    chunks = _chunk_text(row["raw_text"])
    vectors = embed_texts(chunks)

    with get_conn() as conn:
        for chunk, vector in zip(chunks, vectors, strict=False):
            conn.execute(
                """
                INSERT INTO embeddings (document_id, chunk_text, embedding)
                VALUES (%s, %s, %s)
                """,
                (document_id, chunk, vector),
            )
        _log_action(conn, str(row["patient_id"]), "document.embed", actor, {"document_id": document_id})
        conn.commit()

    return {"document_id": document_id, "chunks": len(chunks)}


@app.post("/chr/draft", response_model=ChrDraft)
def draft_chr(payload: ChrDraftRequest):
    return _draft_chr(payload.patient_id, payload.notes, actor="api")


def _draft_chr(patient_id: str, notes: str | None, actor: str = "system") -> ChrDraft:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT e.structured
            FROM extractions e
            JOIN documents d ON d.id = e.document_id
            WHERE d.patient_id = %s
            ORDER BY e.created_at DESC
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()

    if not row or not row.get("structured"):
        raise HTTPException(status_code=404, detail="No extraction found for patient")

    query = build_query(row["structured"], notes)
    context_chunks = retrieve_top_chunks(patient_id, query, top_k=5)
    draft = generate_chr_draft(row["structured"], notes, context_chunks)

    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO chr_versions (patient_id, draft, status)
            VALUES (%s, %s, %s)
            """,
            (patient_id, Json(draft), "draft"),
        )
        _log_action(conn, patient_id, "chr.draft", actor, {"chunks": len(context_chunks)})
        conn.commit()

    return ChrDraft(
        patient_id=patient_id,
        draft=draft,
        citations=draft.get("citations", []),
    )


# -------------------- UI --------------------

def _require_ui_user(request: Request):
    user = request.session.get("user")
    if not user:
        return None
    return user


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui", status_code=302)


@app.get("/ui/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/ui/login", response_class=HTMLResponse, include_in_schema=False)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == settings.app_username and password == settings.app_password:
        request.session["user"] = username
        return RedirectResponse("/ui", status_code=303)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid credentials"},
        status_code=401,
    )


@app.get("/ui/logout", include_in_schema=False)
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/ui/login", status_code=303)


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui_patients(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, full_name, dob, notes FROM patients ORDER BY created_at DESC"
        ).fetchall()

    patients = [_row_to_patient(r) for r in rows]
    dev_mode = request.session.get("dev_mode", False)
    return templates.TemplateResponse(
        "patients.html",
        {"request": request, "patients": patients, "user": user, "dev_mode": dev_mode},
    )


@app.post("/ui/toggle-dev-mode", include_in_schema=False)
def ui_toggle_dev_mode(request: Request):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)
    
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
            GROUP BY p.full_name, d.filename
            ORDER BY created_at DESC NULLS LAST
            LIMIT 100
            """
        ).fetchall()

    dev_mode = request.session.get("dev_mode", False)
    return templates.TemplateResponse(
        "embeddings.html",
        {"request": request, "rows": rows, "user": user, "dev_mode": dev_mode},
    )


@app.post("/ui/patients", include_in_schema=False)
def ui_create_patient(
    request: Request,
    full_name: str = Form(...),
    dob: str = Form(""),
    notes: str = Form(""),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    payload = PatientCreate(full_name=full_name, dob=dob or None, notes=notes or None)
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO patients (full_name, dob, notes)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (payload.full_name, payload.dob, payload.notes),
        ).fetchone()
        _log_action(conn, str(row["id"]), "patient.create", user, {"name": payload.full_name})
        conn.commit()

    return RedirectResponse("/ui", status_code=303)


def _get_patient(patient_id: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, full_name, dob, notes FROM patients WHERE id = %s",
            (patient_id,),
        ).fetchone()


def _list_documents(patient_id: str):
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, patient_id, filename, content_type, storage_path
            FROM documents
            WHERE patient_id = %s
            ORDER BY created_at DESC
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


def _latest_extraction(patient_id: str):
    with get_conn() as conn:
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

    patient_row = _get_patient(patient_id)
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    documents = [_row_to_document(r) for r in _list_documents(patient_id)]
    draft = _latest_draft(patient_id)
    logs = _audit_logs(patient_id)
    has_extractions = _has_extractions(patient_id)
    dev_mode = request.session.get("dev_mode", False)

    return templates.TemplateResponse(
        "patient_detail.html",
        {
            "request": request,
            "user": user,
            "patient": _row_to_patient(patient_row),
            "documents": documents,
            "draft": draft,
            "logs": logs,
            "has_extractions": has_extractions,
            "dev_mode": dev_mode,
        },
    )


@app.get("/ui/patients/{patient_id}/report", response_class=HTMLResponse, include_in_schema=False)
def ui_patient_report(request: Request, patient_id: str):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id)
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    extraction = _latest_extraction(patient_id)
    structured = extraction.get("structured") if extraction else None
    labs = _normalize_labs(structured)
    meds = structured.get("medications") if structured else []
    diagnoses = structured.get("diagnoses") if structured else []

    draft_row = _latest_draft(patient_id)
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
    
    # Render markdown to HTML
    if summary:
        summary_html = markdown.markdown(
            summary,
            extensions=['tables', 'fenced_code', 'sane_lists']
        )
        summary_html = process_citations(summary_html)
    else:
        summary_html = ""

    documents = _list_documents(patient_id)
    findings = _key_findings(labs)

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
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
        },
    )


@app.get("/ui/patients/{patient_id}/report/share", response_class=HTMLResponse, include_in_schema=False)
def ui_patient_report_share(request: Request, patient_id: str):
    """Patient-friendly shareable report view with plain language."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id)
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    extraction = _latest_extraction(patient_id)
    structured = extraction.get("structured") if extraction else None
    labs = _normalize_labs(structured)
    diagnoses = structured.get("diagnoses") if structured else []

    draft_row = _latest_draft(patient_id)
    documents = _list_documents(patient_id)
    findings = _key_findings(labs)

    return templates.TemplateResponse(
        "patient_report.html",
        {
            "request": request,
            "user": user,
            "patient": _row_to_patient(patient_row),
            "draft": draft_row,
            "labs": labs,
            "diagnoses": diagnoses,
            "documents": documents,
            "findings": findings,
        },
    )


@app.post("/ui/patients/{patient_id}/report/query", response_class=HTMLResponse, include_in_schema=False)
def ui_query_report(request: Request, patient_id: str, query: str = Form("")):
    """Handle RAG-powered clinical queries on the report page."""
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    patient_row = _get_patient(patient_id)
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient = _row_to_patient(patient_row)
    
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
    
    # Process answer for display (convert markdown to HTML)
    answer_html = markdown.markdown(
        result["answer"],
        extensions=['tables', 'fenced_code', 'sane_lists']
    )
    
    # Process citations to make them clickable
    def process_query_citations(text):
        def replace_citation(match):
            num = match.group(1)
            return f'<a href="#query-cite-{num}" class="citation-link" title="View source">[{num}]</a>'
        return re.sub(r'\[(\d+)\]', replace_citation, text)
    
    answer_html = process_query_citations(answer_html)
    
    # Render the same report page with query results
    extraction = _latest_extraction(patient_id)
    structured = extraction.get("structured") if extraction else None
    labs = _normalize_labs(structured)
    meds = structured.get("medications") if structured else []
    diagnoses = structured.get("diagnoses") if structured else []
    draft_row = _latest_draft(patient_id)
    draft_payload = draft_row.get("draft") if draft_row else {}
    summary = draft_payload.get("summary", "")
    citations = draft_payload.get("citations", [])
    
    if summary:
        summary_html = markdown.markdown(
            summary,
            extensions=['tables', 'fenced_code', 'sane_lists']
        )
        summary_html = process_query_citations(summary_html)
    else:
        summary_html = ""

    documents = _list_documents(patient_id)
    findings = _key_findings(labs)

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "user": user,
            "patient": patient,
            "draft": draft_row,
            "summary": summary,
            "summary_html": summary_html,
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

    patient_row = _get_patient(patient_id)
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")

    top_k = max(1, min(k, 20))
    query = q.strip()

    if not query:
        extraction = _latest_extraction(patient_id)
        if extraction and extraction.get("structured"):
            query = build_query(extraction["structured"], patient_row.get("notes"))

    chunks = retrieve_top_chunks(patient_id, query, top_k=top_k) if query else []

    dev_mode = request.session.get("dev_mode", False)
    return templates.TemplateResponse(
        "rag_view.html",
        {
            "request": request,
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
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    for upload in files:
        if not upload.filename:
            continue
        # Upload document
        doc = _upload_document(patient_id, upload, actor=user)
        
        # Auto-process: Extract text from document
        try:
            _extract_document(doc.id, actor=user)
            # Auto-process: Generate embeddings after extraction
            try:
                _embed_document(doc.id, actor=user)
            except Exception:
                # Embedding may fail if extraction didn't produce text
                pass
        except Exception:
            # Extraction may fail for some document types
            pass
            
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/documents/{document_id}/extract", include_in_schema=False)
def ui_extract_document(request: Request, document_id: str, patient_id: str = Form(...)):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    _extract_document(document_id, actor=user)
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/documents/{document_id}/embed", include_in_schema=False)
def ui_embed_document(request: Request, document_id: str, patient_id: str = Form(...)):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    _embed_document(document_id, actor=user)
    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/patients/{patient_id}/draft", include_in_schema=False)
def ui_draft_chr(request: Request, patient_id: str, notes: str = Form("")):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    _draft_chr(patient_id, notes or None, actor=user)
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
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

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
        conn.execute(
            "UPDATE patients SET lifestyle = %s WHERE id = %s",
            (Json(lifestyle_data), patient_id),
        )
        _log_action(conn, patient_id, "lifestyle.updated", user, lifestyle_data)
        conn.commit()

    return RedirectResponse(f"/ui/patients/{patient_id}", status_code=303)


@app.post("/ui/patients/{patient_id}/report/save", include_in_schema=False)
def ui_save_report_edits(
    request: Request,
    patient_id: str,
    labs: str = Form(""),
    diagnoses: str = Form(""),
    interpretation: str = Form(""),
):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

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
        "edited_by": user,
    }

    with get_conn() as conn:
        # Update the latest chr_version with edits
        conn.execute(
            """
            UPDATE chr_versions
            SET report_edits = %s
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (Json(edits), patient_id),
        )
        _log_action(conn, patient_id, "report.edited", user, {"fields_edited": list(edits.keys())})
        conn.commit()

    return RedirectResponse(f"/ui/patients/{patient_id}/report", status_code=303)


@app.post("/ui/patients/{patient_id}/report/finalize", include_in_schema=False)
def ui_finalize_report(request: Request, patient_id: str):
    user = _require_ui_user(request)
    if not user:
        return RedirectResponse("/ui/login", status_code=303)

    with get_conn() as conn:
        conn.execute(
            """
            UPDATE chr_versions
            SET status = 'finalized', finalized_at = NOW()
            WHERE patient_id = %s
            AND status = 'draft'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (patient_id,),
        )
        _log_action(conn, patient_id, "report.finalized", user, {})
        conn.commit()

    return RedirectResponse(f"/ui/patients/{patient_id}/report", status_code=303)
