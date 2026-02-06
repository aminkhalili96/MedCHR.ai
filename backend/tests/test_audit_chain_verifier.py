from unittest.mock import MagicMock, patch

from backend.scripts.verify_audit_chain import verify_chain


def _mock_conn(events, chain_state):
    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(fetchall=MagicMock(return_value=events)),
        MagicMock(fetchall=MagicMock(return_value=chain_state)),
    ]
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def test_verify_chain_passes_for_linear_chain():
    events = [
        {
            "id": "e1",
            "tenant_id": "00000000-0000-0000-0000-000000000111",
            "event_time": "2026-01-01T00:00:00Z",
            "prev_hash": "",
            "event_hash": "h1",
        },
        {
            "id": "e2",
            "tenant_id": "00000000-0000-0000-0000-000000000111",
            "event_time": "2026-01-01T00:00:01Z",
            "prev_hash": "h1",
            "event_hash": "h2",
        },
    ]
    chain_state = [{"tenant_key": "00000000-0000-0000-0000-000000000111", "last_hash": "h2"}]
    with patch("backend.scripts.verify_audit_chain.get_conn", return_value=_mock_conn(events, chain_state)):
        ok, errors = verify_chain()
    assert ok is True
    assert errors == []


def test_verify_chain_fails_on_prev_hash_mismatch():
    events = [
        {
            "id": "e1",
            "tenant_id": "00000000-0000-0000-0000-000000000111",
            "event_time": "2026-01-01T00:00:00Z",
            "prev_hash": "wrong",
            "event_hash": "h1",
        },
    ]
    chain_state = [{"tenant_key": "00000000-0000-0000-0000-000000000111", "last_hash": "h1"}]
    with patch("backend.scripts.verify_audit_chain.get_conn", return_value=_mock_conn(events, chain_state)):
        ok, errors = verify_chain()
    assert ok is False
    assert any("prev_hash mismatch" in err for err in errors)
