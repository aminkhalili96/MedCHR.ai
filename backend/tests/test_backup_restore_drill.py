from unittest.mock import MagicMock, patch

from backend.scripts.backup_restore_drill import CRITICAL_TABLES, run_backup_restore_drill


def _mock_conn_ctx(conn):
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def test_backup_restore_drill_passes_when_all_table_drills_pass():
    conn = MagicMock()
    with patch("backend.scripts.backup_restore_drill.get_conn", return_value=_mock_conn_ctx(conn)), patch(
        "backend.scripts.backup_restore_drill._table_exists", return_value=True
    ), patch(
        "backend.scripts.backup_restore_drill._run_table_copy_drill",
        return_value={
            "table": "dummy",
            "source_rows": 1,
            "restored_rows": 1,
            "expected_rows": 1,
            "status": "PASS",
        },
    ) as mock_copy:
        report = run_backup_restore_drill(sample_limit=5)

    assert report["overall_status"] == "PASS"
    assert report["tables_checked"] == len(CRITICAL_TABLES)
    assert mock_copy.call_count == len(CRITICAL_TABLES)


def test_backup_restore_drill_fails_when_any_table_copy_raises():
    conn = MagicMock()

    def _copy(conn_obj, table_name, sample_limit):
        if table_name == "documents":
            raise RuntimeError("restore copy failed")
        return {
            "table": table_name,
            "source_rows": 1,
            "restored_rows": 1,
            "expected_rows": 1,
            "status": "PASS",
        }

    with patch("backend.scripts.backup_restore_drill.get_conn", return_value=_mock_conn_ctx(conn)), patch(
        "backend.scripts.backup_restore_drill._table_exists", return_value=True
    ), patch("backend.scripts.backup_restore_drill._run_table_copy_drill", side_effect=_copy):
        report = run_backup_restore_drill(sample_limit=5)

    assert report["overall_status"] == "FAIL"
    assert report["failed_tables"] >= 1
