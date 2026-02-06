"""
Security Audit helpers backed by the append-only `audit_events` table.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .audit_events import append_audit_event
from .db import get_conn


class AuditAction(str, Enum):
    LOGIN_SUCCESS = "login.success"
    LOGIN_FAILED = "login.failed"
    LOGOUT = "logout"
    MFA_ENABLED = "mfa.enabled"
    MFA_DISABLED = "mfa.disabled"
    PATIENT_VIEW = "patient.view"
    PATIENT_CREATE = "patient.create"
    PATIENT_UPDATE = "patient.update"
    PATIENT_DELETE = "patient.delete"
    DOCUMENT_UPLOAD = "document.upload"
    DOCUMENT_VIEW = "document.view"
    DOCUMENT_DELETE = "document.delete"
    EXTRACTION_RUN = "extraction.run"
    CHR_GENERATE = "chr.generate"
    CHR_VIEW = "chr.view"
    CHR_SIGN = "chr.sign"
    CHR_AMEND = "chr.amend"
    CHR_EXPORT = "chr.export"
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    ROLE_CHANGE = "role.change"
    EXPORT_BULK = "export.bulk"


def log_audit_event(
    *,
    action: AuditAction | str,
    tenant_id: str | None,
    actor_id: str | None = None,
    actor_email: str | None = None,
    resource_type: str = "system",
    resource_id: str | None = None,
    details: dict | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    outcome: str = "SUCCESS",
    conn=None,
) -> None:
    owns_conn = conn is None
    if owns_conn:
        cm = get_conn()
        conn = cm.__enter__()
    try:
        append_audit_event(
            conn,
            action=action.value if isinstance(action, AuditAction) else str(action),
            tenant_id=tenant_id,
            actor_id=actor_id,
            actor=actor_email,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=client_ip,
            user_agent=user_agent,
            request_id=request_id,
            outcome=outcome,
        )
        if owns_conn:
            conn.commit()
    finally:
        if owns_conn:
            cm.__exit__(None, None, None)


def query_audit_log(
    *,
    tenant_id: str,
    action: str | None = None,
    actor_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
    conn=None,
) -> list[dict]:
    owns_conn = conn is None
    if owns_conn:
        cm = get_conn()
        conn = cm.__enter__()
    try:
        query = """
            SELECT id, event_time, actor_id, actor, tenant_id, ip_address, user_agent,
                   resource_type, resource_id, action, outcome, details, request_id, event_hash
            FROM audit_events
            WHERE tenant_id = %s
        """
        params: list = [tenant_id]
        if action:
            query += " AND action = %s"
            params.append(action)
        if actor_id:
            query += " AND actor_id = %s"
            params.append(actor_id)
        if resource_type:
            query += " AND resource_type = %s"
            params.append(resource_type)
        if resource_id:
            query += " AND resource_id = %s"
            params.append(resource_id)
        if start_date:
            query += " AND event_time >= %s"
            params.append(start_date)
        if end_date:
            query += " AND event_time <= %s"
            params.append(end_date)
        query += " ORDER BY event_time DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        if owns_conn:
            cm.__exit__(None, None, None)


def export_audit_log(
    *,
    tenant_id: str,
    start_date: datetime,
    end_date: datetime,
    conn=None,
) -> list[dict]:
    return query_audit_log(
        tenant_id=tenant_id,
        start_date=start_date,
        end_date=end_date,
        limit=100000,
        conn=conn,
    )
