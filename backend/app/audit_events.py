from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg.types.json import Json

from .db import get_actor_context, get_tenant_context


def _to_uuid(value: str | None):
    if not value:
        return None
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def append_audit_event(
    conn,
    *,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    outcome: str = "SUCCESS",
    details: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    actor: str | None = None,
    actor_id: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> None:
    """
    Append an audit event entry.

    This intentionally performs INSERT-only writes; immutability is enforced in SQL migration triggers.
    """
    resolved_tenant_id = tenant_id or get_tenant_context()
    resolved_actor_id = actor_id or get_actor_context()
    event_details = dict(details or {})
    if resource_id and _to_uuid(resource_id) is None:
        event_details.setdefault("resource_id_raw", resource_id)
    conn.execute(
        """
        INSERT INTO audit_events (
            tenant_id,
            actor_id,
            actor,
            ip_address,
            user_agent,
            request_id,
            resource_type,
            resource_id,
            action,
            outcome,
            details
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            resolved_tenant_id,
            _to_uuid(resolved_actor_id),
            actor,
            ip_address,
            user_agent,
            request_id,
            resource_type,
            _to_uuid(resource_id),
            action,
            outcome,
            Json(event_details),
        ),
    )
