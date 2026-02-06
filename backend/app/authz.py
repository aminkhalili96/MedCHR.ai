from __future__ import annotations

import time

from fastapi import HTTPException, Request, status


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "patient.read",
        "patient.write",
        "report.share",
        "report.finalize",
        "user.manage",
        "tenant.policy.manage",
    },
    "clinician": {
        "patient.read",
        "patient.write",
        "report.share",
        "report.finalize",
    },
}


def has_permission(role: str, permission: str) -> bool:
    allowed = ROLE_PERMISSIONS.get((role or "").lower(), set())
    return permission in allowed


def require_permission(role: str, permission: str) -> None:
    if has_permission(role, permission):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Missing permission: {permission}",
    )


def mark_step_up_verified(request: Request) -> None:
    request.session["step_up_verified_at"] = int(time.time())


def is_step_up_verified(request: Request, max_age_minutes: int = 15) -> bool:
    verified_at = request.session.get("step_up_verified_at")
    if not verified_at:
        return False
    now = int(time.time())
    return (now - int(verified_at)) <= max(1, max_age_minutes) * 60


def require_tenant_id(request: Request) -> str:
    """
    Return the authenticated tenant_id for API-key requests.

    UI requests should not use this; they rely on the logged-in user session.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant context missing for API request",
        )
    return str(tenant_id)


def get_patient_row(conn, patient_id: str, tenant_id: str):
    row = conn.execute(
        """
        SELECT id, tenant_id, full_name, dob, notes, lifestyle, genetics
        FROM patients
        WHERE id = %s AND tenant_id = %s
        """,
        (patient_id, tenant_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Patient not found")
    return row


def get_document_row(conn, document_id: str, tenant_id: str):
    row = conn.execute(
        """
        SELECT d.id, d.patient_id, d.filename, d.content_type, d.storage_path
        FROM documents d
        JOIN patients p ON p.id = d.patient_id
        WHERE d.id = %s AND p.tenant_id = %s
        """,
        (document_id, tenant_id),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return row
