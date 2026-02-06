from unittest.mock import MagicMock

from backend.app.audit_events import append_audit_event


def test_append_audit_event_handles_non_uuid_resource_id():
    conn = MagicMock()
    append_audit_event(
        conn,
        action="report.share",
        resource_type="chr",
        resource_id="not-a-uuid",
        details={"note": "x"},
        tenant_id="00000000-0000-0000-0000-000000000001",
        actor="user@example.com",
    )

    params = conn.execute.call_args[0][1]
    assert params[7] is None
