from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from backend.app.mfa import get_mfa_lockout_remaining, record_mfa_failure


def test_get_mfa_lockout_remaining_returns_positive_seconds():
    future = datetime.now(timezone.utc) + timedelta(seconds=120)
    with patch("backend.app.mfa.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = {"lockout_until": future}
        remaining = get_mfa_lockout_remaining("00000000-0000-0000-0000-000000000001", "login")
    assert remaining > 0


def test_record_mfa_failure_locks_after_threshold():
    with patch("backend.app.mfa.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value={"failed_attempts": 5, "lockout_until": None})),
            MagicMock(),
        ]
        locked = record_mfa_failure(
            "00000000-0000-0000-0000-000000000001",
            "login",
            max_attempts=5,
            lockout_seconds=300,
        )
    assert locked is True
