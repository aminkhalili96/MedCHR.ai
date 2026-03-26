"""Shared helper functions used by both API routes and UI handlers."""

from uuid import uuid4

from fastapi import UploadFile, HTTPException
from psycopg.types.json import Json

from .config import get_settings
from .db import get_conn
from .schemas import (
    Patient,
    Document,
    SignedUploadResponse,
    SignedUploadRegistration,
    ExtractionResult,
    ChrDraft,
)
from .storage import (
    upload_bytes_via_signed_url,
    download_bytes,
    create_signed_upload_url,
)
from .ocr import extract_text
from .extract import extract_structured
from .embeddings import embed_texts
from .rag import build_query, retrieve_hybrid
from .chr import generate_chr_draft
from .audit_events import append_audit_event
from .uploads import read_upload_bytes, sanitize_filename, resolve_content_type

settings = get_settings()


def _row_to_patient(row) -> Patient:
    return Patient(
        id=str(row["id"]),
        full_name=row["full_name"],
        dob=row.get("dob"),
        notes=row.get("notes"),
        lifestyle=row.get("lifestyle") or {},
        genetics=row.get("genetics") or {},
        gender=row.get("gender"),
        phone=row.get("phone"),
        email=row.get("email"),
        emergency_contact=row.get("emergency_contact"),
        insurance=row.get("insurance"),
        social_history=row.get("social_history") or {},
        past_medical_history=row.get("past_medical_history") or [],
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
            {"storage_path": resolved_path},
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


def _aggregate_structured(patient_id: str, tenant_id: str | None = None) -> tuple[dict | None, list[dict]]:
    with get_conn() as conn:
        if tenant_id:
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
                  AND p.tenant_id = %s
                ORDER BY d.created_at DESC
                """,
                (patient_id, tenant_id),
            ).fetchall()
        else:
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
                JOIN LATERAL (
                    SELECT id, structured, raw_text, created_at
                    FROM extractions
                    WHERE document_id = d.id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) e ON true
                WHERE d.patient_id = %s
                ORDER BY d.created_at DESC
                """,
                (patient_id,),
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
            dx_str = dx.get("condition") if isinstance(dx, dict) else dx if isinstance(dx, str) else None
            if not dx_str:
                continue
            key = dx_str.strip().lower()
            if not key or key in seen_dx:
                continue
            seen_dx.add(key)
            diagnoses.append(dx_str)

        for med in structured.get("medications") or []:
            med_str = med.get("name") if isinstance(med, dict) else med if isinstance(med, str) else None
            if not med_str:
                continue
            key = med_str.strip().lower()
            if not key or key in seen_meds:
                continue
            seen_meds.add(key)
            medications.append(med_str)

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
        combined_notes = combined_notes[: settings.aggregate_notes_max_chars] + "\u2026"

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
