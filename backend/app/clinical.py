from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from psycopg.rows import dict_row
from uuid import uuid4

from .db import get_conn
from .authz import require_tenant_id, get_patient_row
from .security import require_api_key, require_read_scope, require_write_scope
from .schemas import Allergy, Vital, Immunization
from .audit_events import append_audit_event

router = APIRouter()


def _audit_clinical_event(
    conn,
    *,
    tenant_id: str,
    actor: str,
    patient_id: str,
    action: str,
    details: dict | None = None,
    outcome: str = "SUCCESS",
):
    append_audit_event(
        conn,
        action=action,
        resource_type="patient",
        resource_id=patient_id,
        outcome=outcome,
        details=details or {},
        tenant_id=tenant_id,
        actor=actor,
    )

# -----------------------------------------------------------------------------
# Clinical Data Endpoints
# -----------------------------------------------------------------------------

@router.get("/patients/{patient_id}/allergies", tags=["Clinical"])
def get_allergies(
    request: Request,
    patient_id: str,
    _=Depends(require_api_key),
    __=Depends(require_read_scope),
):
    tenant_id = require_tenant_id(request)
    actor = getattr(request.state, "actor", "api")
    with get_conn() as conn:
        get_patient_row(conn, patient_id, tenant_id)
        rows = conn.execute(
            "SELECT id, substance, reaction, severity, status FROM allergies WHERE patient_id = %s",
            (patient_id,)
        ).fetchall()
        _audit_clinical_event(
            conn,
            tenant_id=tenant_id,
            actor=actor,
            patient_id=patient_id,
            action="patient.allergies_view",
            details={"count": len(rows)},
        )
        conn.commit()
    return rows

@router.post("/patients/{patient_id}/allergies", response_model=Allergy, tags=["Clinical"])
def create_allergy(
    patient_id: str,
    allergy: Allergy,
    request: Request,
    _=Depends(require_api_key),
    __=Depends(require_write_scope),
):
    actor = getattr(request.state, "actor", "api")
    tenant_id = require_tenant_id(request)
    with get_conn() as conn:
        get_patient_row(conn, patient_id, tenant_id)
        row = conn.execute(
            """
            INSERT INTO allergies (patient_id, substance, reaction, severity, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (patient_id, allergy.substance, allergy.reaction, allergy.severity, allergy.status)
        ).fetchone()
        _audit_clinical_event(
            conn,
            tenant_id=tenant_id,
            actor=actor,
            patient_id=patient_id,
            action="patient.allergy_create",
            details={"allergy_id": str(row["id"]), "substance": allergy.substance},
        )
        conn.commit()
    allergy.id = str(row["id"])
    return allergy

@router.get("/patients/{patient_id}/vitals", tags=["Clinical"])
def get_vitals(
    request: Request,
    patient_id: str,
    _=Depends(require_api_key),
    __=Depends(require_read_scope),
):
    tenant_id = require_tenant_id(request)
    actor = getattr(request.state, "actor", "api")
    with get_conn() as conn:
        get_patient_row(conn, patient_id, tenant_id)
        rows = conn.execute(
            "SELECT id, type, value_1, value_2, unit, recorded_at FROM vitals WHERE patient_id = %s ORDER BY recorded_at DESC",
            (patient_id,)
        ).fetchall()
        _audit_clinical_event(
            conn,
            tenant_id=tenant_id,
            actor=actor,
            patient_id=patient_id,
            action="patient.vitals_view",
            details={"count": len(rows)},
        )
        conn.commit()
    return rows

@router.post("/patients/{patient_id}/vitals", response_model=Vital, tags=["Clinical"])
def create_vital(
    patient_id: str,
    vital: Vital,
    request: Request,
    _=Depends(require_api_key),
    __=Depends(require_write_scope),
):
     tenant_id = require_tenant_id(request)
     actor = getattr(request.state, "actor", "api")
     with get_conn() as conn:
        get_patient_row(conn, patient_id, tenant_id)
        row = conn.execute(
            """
            INSERT INTO vitals (patient_id, type, value_1, value_2, unit, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (patient_id, vital.type, vital.value_1, vital.value_2, vital.unit, vital.recorded_at)
        ).fetchone()
        _audit_clinical_event(
            conn,
            tenant_id=tenant_id,
            actor=actor,
            patient_id=patient_id,
            action="patient.vital_create",
            details={"vital_id": str(row["id"]), "type": vital.type},
        )
        conn.commit()
     vital.id = str(row["id"])
     return vital

# -----------------------------------------------------------------------------
# FHIR Interoperability
# -----------------------------------------------------------------------------

@router.get("/fhir/Patient/{patient_id}", tags=["FHIR"])
def get_fhir_patient(
    request: Request,
    patient_id: str,
    _=Depends(require_api_key),
    __=Depends(require_read_scope),
):
    """
    Returns a FHIR R4 Patient resource.
    """
    from fhir.resources.patient import Patient
    
    tenant_id = require_tenant_id(request)
    actor = getattr(request.state, "actor", "api")
    with get_conn() as conn:
        p = get_patient_row(conn, patient_id, tenant_id)
        _audit_clinical_event(
            conn,
            tenant_id=tenant_id,
            actor=actor,
            patient_id=patient_id,
            action="patient.fhir_view",
            details={"resource": "Patient"},
        )
        conn.commit()
        
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")

    fhir_p = Patient.construct()
    fhir_p.id = str(p["id"])
    fhir_p.name = [{"text": p["full_name"], "use": "official"}]
    if p["dob"]:
        fhir_p.birthDate = p["dob"]
    
    # In a real impl, we would map gender, address, telecom from p["json_fields"] if they existed.
    
    return fhir_p.dict()
