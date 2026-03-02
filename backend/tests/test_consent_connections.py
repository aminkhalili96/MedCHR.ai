from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from backend.app.consent import ConsentType, check_consent, record_consent


def _mock_conn_ctx(conn):
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def test_record_consent_uses_context_manager_when_conn_missing():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"id": "consent-1"}
    with patch("backend.app.db.get_conn", return_value=_mock_conn_ctx(conn)):
        consent_id = record_consent(
            patient_id="patient-1",
            consent_type=ConsentType.AI_PROCESSING,
            granted=True,
            granted_by="clinician@example.com",
        )

    assert consent_id == "consent-1"
    conn.execute.assert_called_once()
    conn.commit.assert_called_once()


def test_check_consent_handles_timezone_aware_timestamps():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "id": "consent-1",
        "granted": True,
        "granted_by": "clinician@example.com",
        "granted_at": datetime.now(timezone.utc) - timedelta(days=1),
        "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
        "notes": "",
    }

    result = check_consent("patient-1", ConsentType.AI_PROCESSING, conn)

    assert result["has_consent"] is True
    assert result["status"] == "granted"
