from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from backend.app.db import get_conn


_SYSTEM_TENANT_KEY = "00000000-0000-0000-0000-000000000000"


def _tenant_key(value) -> str:
    return str(value) if value else _SYSTEM_TENANT_KEY


def verify_chain(*, tenant_id: str | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    with get_conn() as conn:
        params: list[str] = []
        where_sql = ""
        if tenant_id:
            where_sql = "WHERE tenant_id = %s"
            params.append(tenant_id)
        rows = conn.execute(
            f"""
            SELECT id, tenant_id, event_time, prev_hash, event_hash
            FROM audit_events
            {where_sql}
            ORDER BY tenant_id NULLS FIRST, event_time ASC, id ASC
            """,
            tuple(params),
        ).fetchall()

        expected_prev_by_tenant: dict[str, str] = defaultdict(str)
        for row in rows:
            key = _tenant_key(row.get("tenant_id"))
            event_id = str(row.get("id"))
            prev_hash = row.get("prev_hash") or ""
            event_hash = row.get("event_hash") or ""
            expected_prev = expected_prev_by_tenant[key]
            if not event_hash:
                errors.append(f"{key}: event {event_id} missing event_hash")
            if prev_hash != expected_prev:
                errors.append(
                    f"{key}: event {event_id} prev_hash mismatch (expected={expected_prev!r}, got={prev_hash!r})"
                )
            expected_prev_by_tenant[key] = event_hash

        state_rows = conn.execute(
            """
            SELECT tenant_key, last_hash
            FROM audit_event_chain_state
            """
        ).fetchall()
        state = {str(row["tenant_key"]): row.get("last_hash") or "" for row in state_rows}
        for tenant_key, expected_last in expected_prev_by_tenant.items():
            actual_last = state.get(tenant_key)
            if actual_last != expected_last:
                errors.append(
                    f"{tenant_key}: chain_state last_hash mismatch (expected={expected_last!r}, got={actual_last!r})"
                )
    return len(errors) == 0, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify append-only audit hash chain integrity.")
    parser.add_argument("--tenant-id", default=None, help="Optional tenant UUID to scope verification.")
    args = parser.parse_args()

    ok, errors = verify_chain(tenant_id=args.tenant_id)
    if ok:
        print("Audit chain verification passed.")
        return

    print("Audit chain verification failed.")
    for err in errors:
        print(f"- {err}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
