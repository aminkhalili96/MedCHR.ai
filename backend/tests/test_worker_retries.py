from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.scripts import worker


def _settings():
    return SimpleNamespace(log_level="INFO", job_poll_interval_seconds=0, job_max_attempts=3)


def test_worker_marks_failed_when_attempts_exceed_threshold():
    job = SimpleNamespace(id="job-1", job_type="extract", payload={}, tenant_id=None, attempts=4)
    with patch("backend.scripts.worker.get_settings", return_value=_settings()), patch(
        "backend.scripts.worker.claim_next_job_from_queue", side_effect=[job, KeyboardInterrupt]
    ), patch("backend.scripts.worker.claim_next_job", return_value=None), patch(
        "backend.scripts.worker.mark_job_failed"
    ) as mock_failed, patch(
        "backend.scripts.worker.handle_job"
    ) as mock_handle, patch(
        "backend.scripts.worker.mark_job_done"
    ), patch(
        "backend.scripts.worker.clear_tenant_context"
    ):
        with pytest.raises(KeyboardInterrupt):
            worker.main()

    mock_failed.assert_called_once_with("job-1", "max attempts exceeded")
    mock_handle.assert_not_called()


def test_worker_marks_failed_when_job_handler_raises():
    job = SimpleNamespace(id="job-2", job_type="extract", payload={}, tenant_id=None, attempts=1)
    with patch("backend.scripts.worker.get_settings", return_value=_settings()), patch(
        "backend.scripts.worker.claim_next_job_from_queue", side_effect=[job, KeyboardInterrupt]
    ), patch("backend.scripts.worker.claim_next_job", return_value=None), patch(
        "backend.scripts.worker.handle_job", side_effect=RuntimeError("boom")
    ), patch("backend.scripts.worker.mark_job_done") as mock_done, patch(
        "backend.scripts.worker.mark_job_failed"
    ) as mock_failed, patch(
        "backend.scripts.worker.time.sleep", return_value=None
    ), patch(
        "backend.scripts.worker.clear_tenant_context"
    ):
        with pytest.raises(KeyboardInterrupt):
            worker.main()

    mock_done.assert_not_called()
    assert mock_failed.call_count == 1
    assert mock_failed.call_args[0][0] == "job-2"
    assert "boom" in mock_failed.call_args[0][1]
