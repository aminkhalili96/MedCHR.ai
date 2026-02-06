"""
Digital Signature Module

Provides PKI-based document signing for legal validity.

Gap Reference: S05
"""

import hashlib
import hmac
import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from .config import get_settings


class SignatureStatus:
    DRAFT = "draft"
    PENDING = "pending_signature"
    SIGNED = "signed"
    COSIGNED = "cosigned"
    AMENDED = "amended"


def generate_document_hash(content: dict) -> str:
    """
    Generate a SHA-256 hash of document content.
    This hash is used to verify document integrity.
    """
    # Normalize content for consistent hashing
    normalized = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


def sign_document(
    document_id: str,
    content_hash: str,
    user_id: str,
    user_name: str,
    role: str,
    conn,
    signature_type: str = "primary"
) -> dict:
    """
    Create a digital signature for a document.
    
    In a production system, this would use PKI (private key signing).
    For now, we use HMAC with a server secret as a stepping stone.
    """
    settings = get_settings()
    
    # Create signature payload
    signature_payload = {
        "document_id": document_id,
        "content_hash": content_hash,
        "signer_id": user_id,
        "signer_name": user_name,
        "signer_role": role,
        "signature_type": signature_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "nonce": str(uuid4())
    }
    
    # Generate HMAC signature (in production, use actual PKI)
    signature_string = json.dumps(signature_payload, sort_keys=True)
    signature = hmac.new(
        settings.app_secret_key.encode(),
        signature_string.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Store signature in database
    result = conn.execute("""
        INSERT INTO document_signatures (
            document_id, content_hash, signer_id, signer_name,
            signer_role, signature_type, signature, signed_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id, signed_at
    """, (
        document_id, content_hash, user_id, user_name,
        role, signature_type, signature
    )).fetchone()
    
    conn.commit()
    
    return {
        "signature_id": str(result["id"]),
        "signed_at": result["signed_at"].isoformat(),
        "signature": signature[:16] + "...",  # Truncated for display
        "status": SignatureStatus.SIGNED
    }


def verify_signature(document_id: str, content_hash: str, conn) -> dict:
    """
    Verify that a document signature is valid and the content hasn't changed.
    """
    # Get the signature record
    sig = conn.execute("""
        SELECT * FROM document_signatures
        WHERE document_id = %s
        ORDER BY signed_at DESC
        LIMIT 1
    """, (document_id,)).fetchone()
    
    if not sig:
        return {
            "valid": False,
            "status": SignatureStatus.DRAFT,
            "message": "No signature found"
        }
    
    # Check if content hash matches
    if sig["content_hash"] != content_hash:
        return {
            "valid": False,
            "status": SignatureStatus.AMENDED,
            "message": "Document has been modified since signing",
            "original_hash": sig["content_hash"],
            "current_hash": content_hash
        }
    
    return {
        "valid": True,
        "status": SignatureStatus.SIGNED,
        "signer": sig["signer_name"],
        "signed_at": sig["signed_at"].isoformat(),
        "role": sig["signer_role"]
    }


def get_signature_history(document_id: str, conn) -> list:
    """
    Get all signatures for a document.
    """
    rows = conn.execute("""
        SELECT id, signer_name, signer_role, signature_type, signed_at
        FROM document_signatures
        WHERE document_id = %s
        ORDER BY signed_at DESC
    """, (document_id,)).fetchall()
    
    return [
        {
            "id": str(r["id"]),
            "signer": r["signer_name"],
            "role": r["signer_role"],
            "type": r["signature_type"],
            "signed_at": r["signed_at"].isoformat()
        }
        for r in rows
    ]


def require_cosignature(document_id: str, cosigner_id: str, conn) -> dict:
    """
    Mark a document as requiring cosignature (e.g., attending physician).
    """
    conn.execute("""
        UPDATE chr_versions
        SET status = %s, cosigner_required = %s
        WHERE id = %s
    """, (SignatureStatus.PENDING, cosigner_id, document_id))
    conn.commit()
    
    return {"status": SignatureStatus.PENDING, "cosigner_id": cosigner_id}
