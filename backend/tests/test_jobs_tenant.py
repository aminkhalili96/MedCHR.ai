from unittest.mock import MagicMock, patch

from backend.app.jobs import claim_next_job_from_queue, enqueue_job, get_job, list_jobs


def test_enqueue_job_persists_tenant_id():
    with patch("backend.app.jobs.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = {"id": "job-123"}

        job_id = enqueue_job(
            "extract",
            {"document_id": "doc-1"},
            tenant_id="tenant-1",
            document_id="doc-1",
        )

        assert job_id == "job-123"
        sql = mock_conn.execute.call_args[0][0]
        params = mock_conn.execute.call_args[0][1]
        assert "tenant_id" in sql
        assert params[0] == "tenant-1"


def test_list_jobs_can_filter_by_tenant_id():
    with patch("backend.app.jobs.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        list_jobs(patient_id="patient-1", tenant_id="tenant-1", statuses=["pending"], limit=10)

        sql = mock_conn.execute.call_args[0][0]
        assert "tenant_id = %s" in sql


def test_get_job_returns_tenant_id():
    with patch("backend.app.jobs.get_conn") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = {
            "id": "job-1",
            "tenant_id": "tenant-1",
            "job_type": "extract",
            "status": "pending",
            "payload": {},
            "patient_id": None,
            "document_id": None,
            "attempts": 0,
        }

        job = get_job("job-1")
        assert job is not None
        assert job.tenant_id == "tenant-1"


def test_claim_next_job_from_queue_uses_job_id():
    with patch("backend.app.jobs._redis_client") as mock_redis_client, patch(
        "backend.app.jobs.get_conn"
    ) as mock_get_conn:
        mock_client = MagicMock()
        mock_client.brpop.return_value = ("medchr:jobs", "job-queue-1")
        mock_redis_client.return_value = mock_client

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = {
            "id": "job-queue-1",
            "tenant_id": "tenant-1",
            "job_type": "extract",
            "status": "running",
            "payload": {},
            "patient_id": None,
            "document_id": None,
            "attempts": 1,
        }

        job = claim_next_job_from_queue(timeout_seconds=1)

        assert job is not None
        assert job.id == "job-queue-1"
        assert job.tenant_id == "tenant-1"


def test_claim_next_job_from_queue_returns_none_for_stale_queue_entry():
    with patch("backend.app.jobs._redis_client") as mock_redis_client, patch(
        "backend.app.jobs.get_conn"
    ) as mock_get_conn:
        mock_client = MagicMock()
        mock_client.brpop.return_value = ("medchr:jobs", "job-missing")
        mock_redis_client.return_value = mock_client

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None

        job = claim_next_job_from_queue(timeout_seconds=1)
        assert job is None
