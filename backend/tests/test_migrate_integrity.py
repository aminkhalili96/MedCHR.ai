from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from backend.scripts import migrate


def _mock_conn_ctx(conn):
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    return cm


def test_migrate_blocks_checksum_mismatch():
    with TemporaryDirectory() as tmp:
        migration_file = Path(tmp) / "001_test.sql"
        migration_file.write_text("SELECT 1;", encoding="utf-8")
        conn = MagicMock()
        with patch("backend.scripts.migrate.get_conn", return_value=_mock_conn_ctx(conn)), patch(
            "backend.scripts.migrate._migration_files", return_value=[migration_file]
        ), patch("backend.scripts.migrate._applied_versions", return_value={"001_test": "invalid-checksum"}):
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                migrate.main()


def test_migrate_backfills_missing_checksum():
    with TemporaryDirectory() as tmp:
        migration_file = Path(tmp) / "001_test.sql"
        migration_file.write_text("SELECT 1;", encoding="utf-8")
        conn = MagicMock()
        with patch("backend.scripts.migrate.get_conn", return_value=_mock_conn_ctx(conn)), patch(
            "backend.scripts.migrate._migration_files", return_value=[migration_file]
        ), patch("backend.scripts.migrate._applied_versions", return_value={"001_test": None}):
            migrate.main()
        executed_sql = [call.args[0] for call in conn.execute.call_args_list]
        assert any("UPDATE schema_migrations SET checksum" in sql for sql in executed_sql)
