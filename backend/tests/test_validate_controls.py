from types import SimpleNamespace
from unittest.mock import patch

from backend.scripts.validate_controls import collect_control_validation


def _settings(**overrides):
    base = {
        "app_env": "prod",
        "hipaa_mode": True,
        "app_secret_key": "super-secret-value",
        "mfa_secret_key": "mfa-secret-value",
        "api_keys": "00000000-0000-0000-0000-000000000000:test-key",
        "trust_proxy_headers": True,
        "trusted_proxy_ips": "10.0.0.0/24",
        "phi_processors": "openai,supabase",
        "retention_immutable_dir": "data/retention_immutable",
        "api_docs_enabled": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _find_check(report: dict, control_id: str) -> dict:
    for check in report["checks"]:
        if check["id"] == control_id:
            return check
    raise AssertionError(f"Missing check {control_id}")


def test_validate_controls_fails_on_untrusted_proxy_config():
    with patch("backend.scripts.validate_controls.get_settings", return_value=_settings(trust_proxy_headers=True, trusted_proxy_ips="")), patch(
        "backend.scripts.validate_controls.get_conn", side_effect=RuntimeError("db unavailable")
    ):
        report = collect_control_validation(require_db=False)

    assert report["overall_status"] == "FAIL"
    assert _find_check(report, "C-NET-001")["status"] == "FAIL"


def test_validate_controls_warns_when_db_missing_but_not_required():
    with patch("backend.scripts.validate_controls.get_settings", return_value=_settings()), patch(
        "backend.scripts.validate_controls.get_conn", side_effect=RuntimeError("db unavailable")
    ):
        report = collect_control_validation(require_db=False)

    assert report["overall_status"] == "PASS"
    assert _find_check(report, "C-DB-000")["status"] == "WARN"


def test_validate_controls_fails_when_db_required_and_unavailable():
    with patch("backend.scripts.validate_controls.get_settings", return_value=_settings()), patch(
        "backend.scripts.validate_controls.get_conn", side_effect=RuntimeError("db unavailable")
    ):
        report = collect_control_validation(require_db=True)

    assert report["overall_status"] == "FAIL"
    assert _find_check(report, "C-DB-000")["status"] == "FAIL"


def test_validate_controls_warns_without_key_source_when_db_unavailable():
    with patch("backend.scripts.validate_controls.get_settings", return_value=_settings(api_keys="")), patch(
        "backend.scripts.validate_controls.get_conn", side_effect=RuntimeError("db unavailable")
    ):
        report = collect_control_validation(require_db=False)

    assert report["overall_status"] == "PASS"
    assert _find_check(report, "C-IAM-001")["status"] == "WARN"


def test_validate_controls_fails_without_key_source_when_db_available():
    class _Cursor:
        def __init__(self, row: dict | None = None):
            self._row = row or {}

        def fetchone(self):
            return self._row

    class _Conn:
        def execute(self, query, params=None):
            if "FROM schema_migrations" in query:
                return _Cursor({"total": 1, "with_checksum": 1})
            if "FROM api_keys" in query and "revoked_at IS NULL" in query:
                return _Cursor({"cnt": 0})
            raise AssertionError(f"Unexpected query in test: {query}")

    class _ConnCtx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch("backend.scripts.validate_controls.get_settings", return_value=_settings(api_keys="")), patch(
        "backend.scripts.validate_controls._db_ready", return_value=(True, "")
    ), patch("backend.scripts.validate_controls.get_conn", side_effect=lambda: _ConnCtx()), patch(
        "backend.scripts.validate_controls._table_exists", return_value=True
    ), patch("backend.scripts.validate_controls.verify_chain", return_value=(True, [])):
        report = collect_control_validation(require_db=True)

    assert report["overall_status"] == "FAIL"
    assert _find_check(report, "C-IAM-001")["status"] == "FAIL"
