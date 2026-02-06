"""
Gap Features API Router
Contains endpoints for: Trends, Timeline, Suggestions, Genetics, Rules
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import date

from .db import get_conn
from .auth import get_current_user, User
from .normalizer import normalize_lab_list, get_loinc_code
from .genetics_interpreter import interpret_patient_genetics, check_drug_gene_interactions, format_genetics_for_chr
from .rules_engine import evaluate_rules, format_rules_for_chr
from .diagnosis_suggester import generate_diagnosis_suggestions, format_suggestions_for_chr
from .timeline import extract_events_from_document, format_timeline_for_display
from .audit_events import append_audit_event

router = APIRouter(prefix="/api/gap", tags=["Gap Features"])


def _audit_gap_event(
    conn,
    *,
    tenant_id: str,
    actor: str,
    actor_id: str | None,
    patient_id: str,
    action: str,
    details: dict | None = None,
) -> None:
    append_audit_event(
        conn,
        action=action,
        resource_type="patient",
        resource_id=patient_id,
        outcome="SUCCESS",
        details=details or {},
        tenant_id=tenant_id,
        actor=actor,
        actor_id=actor_id,
    )

def _require_patient_in_tenant(conn, patient_id: str, tenant_id) -> None:
    row = conn.execute(
        "SELECT 1 FROM patients WHERE id = %s AND tenant_id = %s",
        (patient_id, tenant_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")


# ============== GAP 1: LONGITUDINAL TRENDS ==============

class TrendPoint(BaseModel):
    date: str
    value: float
    document_id: Optional[str] = None
    filename: Optional[str] = None


class TrendData(BaseModel):
    test_name: str
    canonical_name: str
    unit: Optional[str] = None
    data_points: list[TrendPoint]


@router.get("/patients/{patient_id}/trends", response_model=list[TrendData])
async def get_patient_trends(patient_id: str, user: User = Depends(get_current_user)):
    """
    Get longitudinal lab trends for a patient.
    Groups lab values by canonical test name across all documents.
    """
    with get_conn() as conn:
        _require_patient_in_tenant(conn, patient_id, user.tenant_id)
        # Get all extractions for this patient
        rows = conn.execute(
            """
            SELECT 
                d.id as document_id,
                d.filename,
                d.created_at,
                e.structured,
                e.service_date
            FROM documents d
            JOIN extractions e ON e.document_id = d.id
            WHERE d.patient_id = %s
            ORDER BY COALESCE(e.service_date, d.created_at::date) DESC
            """,
            (patient_id,)
        ).fetchall()
        _audit_gap_event(
            conn,
            tenant_id=str(user.tenant_id),
            actor=user.email,
            actor_id=str(user.id),
            patient_id=patient_id,
            action="patient.gap_trends_view",
            details={"records": len(rows)},
        )
        conn.commit()
    
    # Aggregate labs by canonical name
    trends_map: dict[str, TrendData] = {}
    
    for row in rows:
        structured = row.get("structured") or {}
        labs = structured.get("labs") or structured.get("biomarkers") or []
        
        # Normalize lab names
        normalized_labs = normalize_lab_list(labs)
        
        event_date = row.get("service_date") or row.get("created_at")
        if hasattr(event_date, "strftime"):
            date_str = event_date.strftime("%Y-%m-%d")
        else:
            date_str = str(event_date)[:10] if event_date else ""
        
        for lab in normalized_labs:
            canonical = lab.get("canonical_name", lab.get("test", "Unknown"))
            value_str = lab.get("value", "")
            
            # Try to parse numeric value
            try:
                numeric_val = float(''.join(c for c in str(value_str) if c.isdigit() or c == '.'))
            except (ValueError, TypeError):
                continue  # Skip non-numeric values
            
            if canonical not in trends_map:
                trends_map[canonical] = TrendData(
                    test_name=lab.get("original_name", canonical),
                    canonical_name=canonical,
                    unit=lab.get("unit", ""),
                    data_points=[]
                )
            
            trends_map[canonical].data_points.append(TrendPoint(
                date=date_str,
                value=numeric_val,
                document_id=str(row.get("document_id", "")),
                filename=row.get("filename", "")
            ))
    
    # Sort data points within each trend
    for trend in trends_map.values():
        trend.data_points.sort(key=lambda x: x.date)
    
    return list(trends_map.values())


# ============== GAP 4: GENETICS INTERPRETATION ==============

class GeneticsInterpretation(BaseModel):
    gene: str
    variant: str
    phenotype: str
    drugs_affected: list[str]
    recommendation: str


class DrugInteractionAlert(BaseModel):
    severity: str
    gene: str
    variant: str
    drug: str
    phenotype: str
    recommendation: str
    message: str


@router.get("/patients/{patient_id}/genetics", response_model=list[GeneticsInterpretation])
async def get_genetics_interpretation(patient_id: str, user: User = Depends(get_current_user)):
    """
    Get clinical interpretation of patient's genetic data.
    """
    with get_conn() as conn:
        _require_patient_in_tenant(conn, patient_id, user.tenant_id)
        row = conn.execute(
            "SELECT genetics FROM patients WHERE id = %s",
            (patient_id,)
        ).fetchone()
        _audit_gap_event(
            conn,
            tenant_id=str(user.tenant_id),
            actor=user.email,
            actor_id=str(user.id),
            patient_id=patient_id,
            action="patient.gap_genetics_view",
        )
        conn.commit()
    
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    genetics = row.get("genetics") or {}
    interpretations = interpret_patient_genetics(genetics)
    
    return [GeneticsInterpretation(**interp) for interp in interpretations]


@router.get("/patients/{patient_id}/drug-interactions", response_model=list[DrugInteractionAlert])
async def check_patient_drug_interactions(patient_id: str, user: User = Depends(get_current_user)):
    """
    Check for drug-gene interactions based on patient's genetics and current medications.
    """
    with get_conn() as conn:
        _require_patient_in_tenant(conn, patient_id, user.tenant_id)
        patient = conn.execute(
            "SELECT genetics FROM patients WHERE id = %s",
            (patient_id,)
        ).fetchone()
        
        # Get medications from latest extractions
        meds_rows = conn.execute(
            """
            SELECT e.structured
            FROM documents d
            JOIN extractions e ON e.document_id = d.id
            WHERE d.patient_id = %s
            ORDER BY e.created_at DESC
            LIMIT 10
            """,
            (patient_id,)
        ).fetchall()
        _audit_gap_event(
            conn,
            tenant_id=str(user.tenant_id),
            actor=user.email,
            actor_id=str(user.id),
            patient_id=patient_id,
            action="patient.gap_drug_interactions_view",
            details={"source_documents": len(meds_rows)},
        )
        conn.commit()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    genetics = patient.get("genetics") or {}
    
    # Aggregate medications
    medications = set()
    for row in meds_rows:
        structured = row.get("structured") or {}
        for med in structured.get("medications") or []:
            if isinstance(med, dict):
                medications.add(med.get("name", ""))
            else:
                medications.add(str(med))
    
    alerts = check_drug_gene_interactions(genetics, list(medications))
    
    return [DrugInteractionAlert(**alert) for alert in alerts]


# ============== GAP 5: DIAGNOSIS SUGGESTIONS ==============

class DiagnosisSuggestion(BaseModel):
    diagnosis: str
    rationale: str
    icd10: Optional[str] = None
    confidence: str


@router.get("/patients/{patient_id}/suggested-diagnoses", response_model=list[DiagnosisSuggestion])
async def get_diagnosis_suggestions(patient_id: str, user: User = Depends(get_current_user)):
    """
    Get AI-generated diagnosis suggestions based on patient data.
    """
    with get_conn() as conn:
        _require_patient_in_tenant(conn, patient_id, user.tenant_id)
        # Get patient info
        patient = conn.execute(
            "SELECT full_name, notes, genetics FROM patients WHERE id = %s",
            (patient_id,)
        ).fetchone()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get extractions
        extractions = conn.execute(
            """
            SELECT e.structured
            FROM documents d
            JOIN extractions e ON e.document_id = d.id
            WHERE d.patient_id = %s
            ORDER BY e.created_at DESC
            """,
            (patient_id,)
        ).fetchall()
        _audit_gap_event(
            conn,
            tenant_id=str(user.tenant_id),
            actor=user.email,
            actor_id=str(user.id),
            patient_id=patient_id,
            action="patient.gap_suggested_diagnoses_view",
            details={"extractions": len(extractions)},
        )
        conn.commit()
    
    # Aggregate data
    labs = []
    medications = []
    existing_diagnoses = []
    
    for row in extractions:
        structured = row.get("structured") or {}
        labs.extend(structured.get("labs") or structured.get("biomarkers") or [])
        
        for med in structured.get("medications") or []:
            if isinstance(med, dict):
                medications.append(med.get("name", ""))
            else:
                medications.append(str(med))
        
        for dx in structured.get("diagnoses") or []:
            if isinstance(dx, dict):
                existing_diagnoses.append(dx.get("condition", str(dx)))
            else:
                existing_diagnoses.append(str(dx))
    
    suggestions = generate_diagnosis_suggestions(
        patient_summary=f"{patient.get('full_name', '')}. {patient.get('notes', '')}",
        labs=labs[:30],  # Limit
        medications=list(set(medications)),
        genetics=patient.get("genetics") or {},
        existing_diagnoses=list(set(existing_diagnoses))
    )
    
    return [DiagnosisSuggestion(**s) for s in suggestions]


# ============== GAP 2: CLINICAL RULES ==============

class TriggeredRule(BaseModel):
    rule_id: str
    recommendation: str


@router.get("/patients/{patient_id}/clinical-insights", response_model=list[TriggeredRule])
async def get_clinical_insights(patient_id: str, user: User = Depends(get_current_user)):
    """
    Get clinical insights based on rule engine evaluation.
    """
    with get_conn() as conn:
        _require_patient_in_tenant(conn, patient_id, user.tenant_id)
        patient = conn.execute(
            "SELECT genetics FROM patients WHERE id = %s",
            (patient_id,)
        ).fetchone()
        
        extractions = conn.execute(
            """
            SELECT e.structured
            FROM documents d
            JOIN extractions e ON e.document_id = d.id
            WHERE d.patient_id = %s
            """,
            (patient_id,)
        ).fetchall()
        _audit_gap_event(
            conn,
            tenant_id=str(user.tenant_id),
            actor=user.email,
            actor_id=str(user.id),
            patient_id=patient_id,
            action="patient.gap_clinical_insights_view",
            details={"extractions": len(extractions)},
        )
        conn.commit()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Aggregate labs
    labs = []
    for row in extractions:
        structured = row.get("structured") or {}
        labs.extend(structured.get("labs") or structured.get("biomarkers") or [])
    
    # Normalize labs
    normalized_labs = normalize_lab_list(labs)
    
    triggered = evaluate_rules(normalized_labs, patient.get("genetics") or {})
    
    return [TriggeredRule(**r) for r in triggered]


# ============== GAP 7: PATIENT TIMELINE ==============

class TimelineEvent(BaseModel):
    id: Optional[str] = None
    event_type: str
    event_date: str
    display_date: Optional[str] = None
    title: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    source_document_id: Optional[str] = None


@router.get("/patients/{patient_id}/timeline", response_model=list[TimelineEvent])
async def get_patient_timeline(patient_id: str, user: User = Depends(get_current_user)):
    """
    Get chronological timeline of patient events.
    """
    with get_conn() as conn:
        _require_patient_in_tenant(conn, patient_id, user.tenant_id)
        # First try the patient_events table
        events_rows = conn.execute(
            """
            SELECT id, event_type, event_date, title, description, source_document_id
            FROM patient_events
            WHERE patient_id = %s
            ORDER BY event_date DESC
            """,
            (patient_id,)
        ).fetchall()
        _audit_gap_event(
            conn,
            tenant_id=str(user.tenant_id),
            actor=user.email,
            actor_id=str(user.id),
            patient_id=patient_id,
            action="patient.gap_timeline_view",
            details={"stored_events": len(events_rows)},
        )
        conn.commit()
        
        # Also generate events from documents if table is empty
        if not events_rows:
            doc_rows = conn.execute(
                """
                SELECT 
                    d.id as document_id,
                    d.filename,
                    d.created_at,
                    e.raw_text,
                    e.structured,
                    e.service_date
                FROM documents d
                LEFT JOIN extractions e ON e.document_id = d.id
                WHERE d.patient_id = %s
                ORDER BY COALESCE(e.service_date, d.created_at::date) DESC
                """,
                (patient_id,)
            ).fetchall()
            
            events = []
            for row in doc_rows:
                doc_events = extract_events_from_document(
                    document_id=str(row.get("document_id", "")),
                    filename=row.get("filename", ""),
                    raw_text=row.get("raw_text", "") or "",
                    structured_data=row.get("structured") or {}
                )
                events.extend(doc_events)
            
            formatted = format_timeline_for_display(events)
            return [TimelineEvent(**e) for e in formatted]
    
    # Format stored events
    events = [dict(row) for row in events_rows]
    formatted = format_timeline_for_display(events)
    
    return [TimelineEvent(**e) for e in formatted]


# ============== GAP 6: BIOMARKER PROVENANCE (handled in aggregation) ==============
# Provenance is added during extraction - see normalizer integration
