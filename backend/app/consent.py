"""
Consent Management Module

Manages patient consent records for data processing.

Gap Reference: S10
"""

from datetime import datetime, timedelta
from typing import Optional
from enum import Enum


class ConsentType(str, Enum):
    TREATMENT = "treatment"
    DATA_SHARING = "data_sharing"
    AI_PROCESSING = "ai_processing"
    RESEARCH = "research"
    MARKETING = "marketing"


class ConsentStatus(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


def record_consent(
    patient_id: str,
    consent_type: ConsentType,
    granted: bool,
    granted_by: str,
    expiry_days: int = 365,
    document_url: str = None,
    notes: str = None,
    conn = None
) -> str:
    """
    Record patient consent.
    """
    from .db import get_conn
    
    if conn is None:
        conn = get_conn()
    
    expires_at = datetime.utcnow() + timedelta(days=expiry_days) if expiry_days else None
    
    result = conn.execute("""
        INSERT INTO patient_consents (
            patient_id, consent_type, granted, granted_by,
            granted_at, expires_at, document_url, notes
        )
        VALUES (%s, %s, %s, %s, NOW(), %s, %s, %s)
        ON CONFLICT (patient_id, consent_type) DO UPDATE SET
            granted = EXCLUDED.granted,
            granted_by = EXCLUDED.granted_by,
            granted_at = NOW(),
            expires_at = EXCLUDED.expires_at,
            document_url = EXCLUDED.document_url,
            notes = EXCLUDED.notes
        RETURNING id
    """, (
        patient_id, consent_type.value, granted, granted_by,
        expires_at, document_url, notes
    )).fetchone()
    
    conn.commit()
    return str(result["id"])


def check_consent(
    patient_id: str,
    consent_type: ConsentType,
    conn = None
) -> dict:
    """
    Check if patient has granted consent for a specific type.
    """
    from .db import get_conn
    
    if conn is None:
        conn = get_conn()
    
    row = conn.execute("""
        SELECT id, granted, granted_by, granted_at, expires_at, notes
        FROM patient_consents
        WHERE patient_id = %s AND consent_type = %s
    """, (patient_id, consent_type.value)).fetchone()
    
    if not row:
        return {
            "has_consent": False,
            "status": ConsentStatus.DENIED,
            "message": "No consent record found"
        }
    
    # Check expiry
    if row["expires_at"] and row["expires_at"] < datetime.utcnow():
        return {
            "has_consent": False,
            "status": ConsentStatus.EXPIRED,
            "message": "Consent has expired",
            "expired_at": row["expires_at"].isoformat()
        }
    
    return {
        "has_consent": row["granted"],
        "status": ConsentStatus.GRANTED if row["granted"] else ConsentStatus.DENIED,
        "granted_by": row["granted_by"],
        "granted_at": row["granted_at"].isoformat() if row["granted_at"] else None,
        "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None
    }


def require_consent(consent_type: ConsentType):
    """
    Decorator to require consent before processing.
    """
    def decorator(func):
        async def wrapper(patient_id: str, *args, **kwargs):
            from .db import get_conn
            
            with get_conn() as conn:
                consent = check_consent(patient_id, consent_type, conn)
                
                if not consent["has_consent"]:
                    raise PermissionError(
                        f"Patient consent required for {consent_type.value}. "
                        f"Status: {consent['status']}"
                    )
            
            return await func(patient_id, *args, **kwargs)
        return wrapper
    return decorator


def get_patient_consents(patient_id: str, conn = None) -> list:
    """
    Get all consent records for a patient.
    """
    from .db import get_conn
    
    if conn is None:
        conn = get_conn()
    
    rows = conn.execute("""
        SELECT consent_type, granted, granted_by, granted_at, expires_at, notes
        FROM patient_consents
        WHERE patient_id = %s
    """, (patient_id,)).fetchall()
    
    consents = []
    for row in rows:
        status = ConsentStatus.GRANTED if row["granted"] else ConsentStatus.DENIED
        
        if row["expires_at"] and row["expires_at"] < datetime.utcnow():
            status = ConsentStatus.EXPIRED
        
        consents.append({
            "type": row["consent_type"],
            "granted": row["granted"],
            "status": status,
            "granted_by": row["granted_by"],
            "granted_at": row["granted_at"].isoformat() if row["granted_at"] else None,
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "notes": row["notes"]
        })
    
    return consents


def withdraw_consent(
    patient_id: str,
    consent_type: ConsentType,
    withdrawn_by: str,
    reason: str = None,
    conn = None
):
    """
    Withdraw patient consent.
    """
    from .db import get_conn
    
    if conn is None:
        conn = get_conn()
    
    conn.execute("""
        UPDATE patient_consents
        SET granted = FALSE, notes = %s, granted_by = %s, granted_at = NOW()
        WHERE patient_id = %s AND consent_type = %s
    """, (f"Withdrawn: {reason}" if reason else "Consent withdrawn", withdrawn_by, patient_id, consent_type.value))
    
    conn.commit()


def check_ai_processing_allowed(patient_id: str, conn = None) -> bool:
    """
    Check if AI processing is allowed for patient data.
    """
    consent = check_consent(patient_id, ConsentType.AI_PROCESSING, conn)
    return consent["has_consent"]
