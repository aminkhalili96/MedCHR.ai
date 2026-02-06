from __future__ import annotations

import argparse
import secrets

from psycopg.types.json import Json

from backend.app.db import get_conn
from backend.app.security import hash_api_key


def build_api_key() -> str:
    return f"medchr_{secrets.token_urlsafe(32)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tenant-scoped API key in the database.")
    parser.add_argument("--tenant-id", required=True, help="Tenant UUID for this key.")
    parser.add_argument("--name", required=True, help="Friendly key name for auditability.")
    parser.add_argument(
        "--scopes",
        default="read",
        help="Comma-separated scopes (for example: read,write).",
    )
    parser.add_argument("--rate-limit", type=int, default=1000, help="Per-key rate limit.")
    parser.add_argument(
        "--key",
        default=None,
        help="Optional plaintext key to import. If omitted, a secure random key is generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plaintext_key = args.key or build_api_key()
    key_hash = hash_api_key(plaintext_key)
    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    if not scopes:
        scopes = ["read"]

    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO api_keys (tenant_id, name, key_hash, scopes, rate_limit)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (args.tenant_id, args.name, key_hash, Json(scopes), args.rate_limit),
        ).fetchone()
        conn.commit()

    print("API key created")
    print(f"id: {row['id']}")
    print(f"tenant_id: {args.tenant_id}")
    print(f"name: {args.name}")
    print(f"scopes: {','.join(scopes)}")
    print(f"rate_limit: {args.rate_limit}")
    print(f"key: {plaintext_key}")
    print("Store this key securely. It will not be shown again.")


if __name__ == "__main__":
    main()
