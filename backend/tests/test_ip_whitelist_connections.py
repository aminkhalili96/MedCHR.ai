from unittest.mock import MagicMock, patch

from backend.app.ip_whitelist import get_tenant_whitelist, update_tenant_whitelist


def _mock_conn_ctx(conn):
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def test_get_tenant_whitelist_uses_context_manager_when_conn_missing():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"ip_whitelist": "[\"10.0.0.1\", \"10.0.0.0/24\"]"}

    with patch("backend.app.db.get_conn", return_value=_mock_conn_ctx(conn)):
        whitelist = get_tenant_whitelist("tenant-1")

    assert whitelist == ["10.0.0.1", "10.0.0.0/24"]
    conn.execute.assert_called_once()


def test_update_tenant_whitelist_uses_context_manager_when_conn_missing():
    conn = MagicMock()
    with patch("backend.app.db.get_conn", return_value=_mock_conn_ctx(conn)):
        update_tenant_whitelist("tenant-1", ["10.0.0.1"])

    conn.execute.assert_called_once()
    conn.commit.assert_called_once()
