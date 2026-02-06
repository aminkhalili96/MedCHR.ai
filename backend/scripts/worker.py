from __future__ import annotations

import logging
import time

from backend.app.config import get_settings
from backend.app.db import clear_tenant_context, set_tenant_context
from backend.app.jobs import (
    claim_next_job,
    claim_next_job_from_queue,
    enqueue_job,
    mark_job_done,
    mark_job_failed,
)
from backend.app.main import _draft_chr, _embed_document, _extract_document


def handle_job(job) -> None:
    payload = job.payload or {}
    actor = payload.get("actor", "worker")
    tenant_id = payload.get("tenant_id") or getattr(job, "tenant_id", None)
    set_tenant_context(tenant_id)

    if job.job_type == "extract":
        document_id = payload.get("document_id") or job.document_id
        if not document_id:
            raise ValueError("extract job missing document_id")
        _extract_document(document_id, actor=actor, tenant_id=tenant_id)
        if payload.get("auto_embed"):
            enqueue_job(
                "embed",
                {"document_id": document_id, "actor": actor, "tenant_id": tenant_id},
                tenant_id=tenant_id,
                document_id=document_id,
            )
        return

    if job.job_type == "embed":
        document_id = payload.get("document_id") or job.document_id
        if not document_id:
            raise ValueError("embed job missing document_id")
        _embed_document(document_id, actor=actor, tenant_id=tenant_id)
        return

    if job.job_type == "draft_chr":
        patient_id = payload.get("patient_id") or job.patient_id
        if not patient_id:
            raise ValueError("draft_chr job missing patient_id")
        _draft_chr(patient_id, payload.get("notes"), actor=actor, tenant_id=tenant_id)
        return

    raise ValueError(f"Unknown job type: {job.job_type}")


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logger = logging.getLogger("medchr.worker")
    logger.info("Worker starting")

    while True:
        job = claim_next_job_from_queue(timeout_seconds=settings.job_poll_interval_seconds)
        if not job:
            job = claim_next_job()
        if not job:
            time.sleep(settings.job_poll_interval_seconds)
            continue

        if job.attempts > settings.job_max_attempts:
            mark_job_failed(job.id, "max attempts exceeded")
            continue

        try:
            handle_job(job)
            mark_job_done(job.id)
        except Exception as exc:
            logger.exception("Job failed", extra={"job_id": job.id, "job_type": job.job_type})
            mark_job_failed(job.id, str(exc))
            time.sleep(1)
        finally:
            clear_tenant_context()


if __name__ == "__main__":
    main()
