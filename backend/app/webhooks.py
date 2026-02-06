"""
Webhook Handler Module

Provides webhook dispatch for key events:
- Extraction complete
- Critical value detected
- Document signed
- CHR generated

Gap Reference: F06
"""

import json
import hashlib
import hmac
import time
from typing import Optional
from enum import Enum

import httpx

from .config import get_settings
from .db import get_conn


class WebhookEvent(str, Enum):
    EXTRACTION_COMPLETE = "extraction.complete"
    CRITICAL_VALUE = "critical_value.detected"
    DOCUMENT_SIGNED = "document.signed"
    CHR_GENERATED = "chr.generated"
    PATIENT_UPDATED = "patient.updated"


class WebhookDeliveryStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


def register_webhook(
    tenant_id: str,
    url: str,
    events: list,
    secret: str = None,
    conn = None
) -> str:
    """
    Register a new webhook endpoint.
    """
    import secrets
    
    if secret is None:
        secret = secrets.token_hex(32)
    
    if conn is None:
        conn = get_conn()
    
    result = conn.execute("""
        INSERT INTO webhooks (tenant_id, url, events, secret, active)
        VALUES (%s, %s, %s, %s, TRUE)
        RETURNING id
    """, (tenant_id, url, json.dumps(events), secret)).fetchone()
    
    conn.commit()
    return str(result["id"])


def get_webhooks_for_event(tenant_id: str, event: str, conn = None) -> list:
    """
    Get all active webhooks subscribed to an event.
    """
    if conn is None:
        conn = get_conn()
    
    rows = conn.execute("""
        SELECT id, url, secret
        FROM webhooks
        WHERE tenant_id = %s AND active = TRUE
          AND events @> %s::jsonb
    """, (tenant_id, json.dumps([event]))).fetchall()
    
    return [dict(r) for r in rows]


def generate_signature(payload: str, secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.
    """
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()


async def dispatch_webhook(
    webhook_id: str,
    event: str,
    payload: dict,
    conn = None
) -> bool:
    """
    Dispatch a webhook to the registered endpoint.
    """
    if conn is None:
        conn = get_conn()
    
    # Get webhook details
    webhook = conn.execute(
        "SELECT url, secret FROM webhooks WHERE id = %s",
        (webhook_id,)
    ).fetchone()
    
    if not webhook:
        return False
    
    # Prepare payload
    full_payload = {
        "event": event,
        "timestamp": int(time.time()),
        "data": payload
    }
    payload_json = json.dumps(full_payload, default=str)
    
    # Generate signature
    signature = generate_signature(payload_json, webhook["secret"])
    
    # Send webhook
    headers = {
        "Content-Type": "application/json",
        "X-MedCHR-Event": event,
        "X-MedCHR-Signature": f"sha256={signature}",
        "X-MedCHR-Timestamp": str(full_payload["timestamp"])
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                webhook["url"],
                content=payload_json,
                headers=headers
            )
        
        status = WebhookDeliveryStatus.SUCCESS if response.is_success else WebhookDeliveryStatus.FAILED
        
        # Log delivery
        conn.execute("""
            INSERT INTO webhook_deliveries (
                webhook_id, event, payload, status_code, 
                response_body, status, delivered_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (
            webhook_id, event, payload_json,
            response.status_code, response.text[:1000],
            status.value
        ))
        conn.commit()
        
        return response.is_success
        
    except Exception as e:
        # Log failure
        conn.execute("""
            INSERT INTO webhook_deliveries (
                webhook_id, event, payload, status, error, delivered_at
            )
            VALUES (%s, %s, %s, %s, %s, NOW())
        """, (
            webhook_id, event, payload_json,
            WebhookDeliveryStatus.FAILED.value, str(e)
        ))
        conn.commit()
        
        return False


async def dispatch_event(
    tenant_id: str,
    event: WebhookEvent,
    payload: dict,
    conn = None
):
    """
    Dispatch an event to all subscribed webhooks.
    """
    if conn is None:
        conn = get_conn()
    
    webhooks = get_webhooks_for_event(tenant_id, event.value, conn)
    
    for webhook in webhooks:
        await dispatch_webhook(
            webhook["id"],
            event.value,
            payload,
            conn
        )


# Convenience dispatchers
async def notify_extraction_complete(
    tenant_id: str,
    patient_id: str,
    document_id: str,
    extraction_id: str,
    summary: dict,
    conn = None
):
    """Notify that extraction is complete."""
    await dispatch_event(
        tenant_id,
        WebhookEvent.EXTRACTION_COMPLETE,
        {
            "patient_id": patient_id,
            "document_id": document_id,
            "extraction_id": extraction_id,
            "summary": summary
        },
        conn
    )


async def notify_critical_value(
    tenant_id: str,
    patient_id: str,
    alert: dict,
    conn = None
):
    """Notify that a critical value was detected."""
    await dispatch_event(
        tenant_id,
        WebhookEvent.CRITICAL_VALUE,
        {
            "patient_id": patient_id,
            "alert": alert
        },
        conn
    )


async def notify_chr_generated(
    tenant_id: str,
    patient_id: str,
    chr_id: str,
    conn = None
):
    """Notify that a CHR was generated."""
    await dispatch_event(
        tenant_id,
        WebhookEvent.CHR_GENERATED,
        {
            "patient_id": patient_id,
            "chr_id": chr_id
        },
        conn
    )
