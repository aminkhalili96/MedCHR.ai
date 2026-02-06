from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import redis
from psycopg.types.json import Json

from .config import get_settings
from .db import get_conn


@dataclass(frozen=True)
class Job:
    id: str
    tenant_id: str | None
    job_type: str
    status: str
    payload: dict[str, Any]
    patient_id: str | None
    document_id: str | None
    attempts: int


def _redis_client() -> redis.Redis | None:
    settings = get_settings()
    if not settings.redis_url:
        return None
    try:
        return redis.Redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:
        return None


def _queue_job_id(job_id: str) -> None:
    client = _redis_client()
    if not client:
        return
    settings = get_settings()
    try:
        # LPUSH + BRPOP yields FIFO semantics.
        client.lpush(settings.redis_queue_name, job_id)
    except Exception:
        return


def enqueue_job(
    job_type: str,
    payload: dict[str, Any],
    *,
    tenant_id: str | None = None,
    patient_id: str | None = None,
    document_id: str | None = None,
) -> str:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO jobs (tenant_id, job_type, payload, patient_id, document_id)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (tenant_id, job_type, Json(payload), patient_id, document_id),
        ).fetchone()
        conn.commit()
    job_id = str(row["id"])
    _queue_job_id(job_id)
    return job_id


def get_job(job_id: str) -> Job | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, tenant_id, job_type, status, payload, patient_id, document_id, attempts
            FROM jobs
            WHERE id = %s
            """,
            (job_id,),
        ).fetchone()
    if not row:
        return None
    return Job(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
        job_type=row["job_type"],
        status=row["status"],
        payload=row.get("payload") or {},
        patient_id=str(row["patient_id"]) if row.get("patient_id") else None,
        document_id=str(row["document_id"]) if row.get("document_id") else None,
        attempts=row.get("attempts", 0),
    )


def list_jobs(
    *,
    patient_id: str | None = None,
    tenant_id: str | None = None,
    statuses: list[str] | None = None,
    limit: int = 50,
) -> list[Job]:
    clauses = []
    params: list[Any] = []
    if patient_id:
        clauses.append("patient_id = %s")
        params.append(patient_id)
    if tenant_id:
        clauses.append("tenant_id = %s")
        params.append(tenant_id)
    if statuses:
        clauses.append("status = ANY(%s)")
        params.append(statuses)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT id, tenant_id, job_type, status, payload, patient_id, document_id, attempts
        FROM jobs
        {where}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        Job(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
            job_type=row["job_type"],
            status=row["status"],
            payload=row.get("payload") or {},
            patient_id=str(row["patient_id"]) if row.get("patient_id") else None,
            document_id=str(row["document_id"]) if row.get("document_id") else None,
            attempts=row.get("attempts", 0),
        )
        for row in rows
    ]


def claim_next_job() -> Job | None:
    with get_conn() as conn:
        row = conn.execute(
            """
            WITH next_job AS (
                SELECT id
                FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
            SET status = 'running', started_at = NOW(), updated_at = NOW(), attempts = attempts + 1
            WHERE id IN (SELECT id FROM next_job)
            RETURNING id, tenant_id, job_type, status, payload, patient_id, document_id, attempts
            """
        ).fetchone()
        conn.commit()
    if not row:
        return None
    return Job(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
        job_type=row["job_type"],
        status=row["status"],
        payload=row.get("payload") or {},
        patient_id=str(row["patient_id"]) if row.get("patient_id") else None,
        document_id=str(row["document_id"]) if row.get("document_id") else None,
        attempts=row.get("attempts", 0),
    )


def claim_next_job_from_queue(timeout_seconds: int = 1) -> Job | None:
    client = _redis_client()
    if not client:
        return None
    settings = get_settings()
    try:
        popped = client.brpop(settings.redis_queue_name, timeout=max(1, int(timeout_seconds)))
    except Exception:
        return None
    if not popped:
        return None
    _, job_id = popped

    with get_conn() as conn:
        row = conn.execute(
            """
            UPDATE jobs
            SET status = 'running', started_at = NOW(), updated_at = NOW(), attempts = attempts + 1
            WHERE id = %s
              AND status = 'pending'
            RETURNING id, tenant_id, job_type, status, payload, patient_id, document_id, attempts
            """,
            (job_id,),
        ).fetchone()
        conn.commit()
    if not row:
        return None
    return Job(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]) if row.get("tenant_id") else None,
        job_type=row["job_type"],
        status=row["status"],
        payload=row.get("payload") or {},
        patient_id=str(row["patient_id"]) if row.get("patient_id") else None,
        document_id=str(row["document_id"]) if row.get("document_id") else None,
        attempts=row.get("attempts", 0),
    )


def mark_job_done(job_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'done', finished_at = NOW(), updated_at = NOW()
            WHERE id = %s
            """,
            (job_id,),
        )
        conn.commit()


def mark_job_failed(job_id: str, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', last_error = %s, finished_at = NOW(), updated_at = NOW()
            WHERE id = %s
            """,
            (error[:800], job_id),
        )
        conn.commit()
