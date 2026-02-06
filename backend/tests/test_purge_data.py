from unittest.mock import MagicMock, patch

from backend.scripts.purge_data import _enforce_immutable_exports, _export_rows, purge_table


def test_export_rows_returns_count_in_dry_run():
    with patch("backend.scripts.purge_data.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = {"cnt": 4}

        count, output, checksum = _export_rows("audit_logs", 30, "data/exports", dry_run=True)

        assert count == 4
        assert output is None
        assert checksum is None


def test_purge_table_returns_zero_on_missing_table():
    with patch("backend.scripts.purge_data.get_conn", side_effect=RuntimeError("no db")):
        assert purge_table("missing_table", 30, dry_run=True) == 0


def test_immutable_exports_skip_when_no_candidates():
    uris = _enforce_immutable_exports(
        immutable_dir="",
        audit_days=30,
        export_counts={"audit_logs": 0, "audit_events": 0, "phi_egress_events": 0},
        export_files={"audit_logs": None, "audit_events": None, "phi_egress_events": None},
        checksums={"audit_logs": None, "audit_events": None, "phi_egress_events": None},
    )
    assert uris == {}
