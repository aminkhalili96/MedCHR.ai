from __future__ import annotations

import argparse
import os
import sys

from backend.app.auth import get_password_hash
from backend.app.db import get_conn

SYSTEM_TENANT_ID = "00000000-0000-0000-0000-000000000000"


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap a tenant and the first admin user (for /ui login)."
    )
    parser.add_argument(
        "--tenant-name",
        default=_env("BOOTSTRAP_TENANT_NAME") or "System Tenant",
        help="Tenant name to use/create (default: System Tenant).",
    )
    parser.add_argument(
        "--tenant-id",
        default=_env("BOOTSTRAP_TENANT_ID") or SYSTEM_TENANT_ID,
        help="Tenant UUID to use/create (default: system tenant UUID).",
    )
    parser.add_argument(
        "--admin-email",
        default=_env("BOOTSTRAP_ADMIN_EMAIL"),
        help="Admin email address (or set BOOTSTRAP_ADMIN_EMAIL).",
    )
    parser.add_argument(
        "--admin-password",
        default=_env("BOOTSTRAP_ADMIN_PASSWORD"),
        help="Admin password (or set BOOTSTRAP_ADMIN_PASSWORD). Avoid passing via shell history in production.",
    )
    parser.add_argument(
        "--promote-admin",
        action="store_true",
        help="If the user exists, ensure role is admin.",
    )
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="If the user exists, reset password_hash using --admin-password/BOOTSTRAP_ADMIN_PASSWORD.",
    )
    args = parser.parse_args()

    if not args.admin_email:
        print("Missing admin email. Set BOOTSTRAP_ADMIN_EMAIL or pass --admin-email.")
        sys.exit(2)

    if args.reset_password and not args.admin_password:
        print("Missing admin password for reset. Set BOOTSTRAP_ADMIN_PASSWORD or pass --admin-password.")
        sys.exit(2)

    with get_conn() as conn:
        # Ensure tenant exists
        tenant = conn.execute("SELECT id FROM tenants WHERE id = %s", (args.tenant_id,)).fetchone()
        if not tenant:
            conn.execute(
                "INSERT INTO tenants (id, name) VALUES (%s, %s)",
                (args.tenant_id, args.tenant_name),
            )
            tenant_id = args.tenant_id
            tenant_action = "created"
        else:
            tenant_id = str(tenant["id"])
            tenant_action = "exists"

        # Ensure user exists
        user = conn.execute(
            "SELECT id, role FROM users WHERE email = %s",
            (args.admin_email,),
        ).fetchone()

        if not user:
            if not args.admin_password:
                print("Missing admin password. Set BOOTSTRAP_ADMIN_PASSWORD or pass --admin-password.")
                sys.exit(2)
            pw_hash = get_password_hash(args.admin_password)
            row = conn.execute(
                """
                INSERT INTO users (email, password_hash, role, tenant_id)
                VALUES (%s, %s, 'admin', %s)
                RETURNING id
                """,
                (args.admin_email, pw_hash, tenant_id),
            ).fetchone()
            user_id = str(row["id"])
            user_action = "created"
        else:
            user_id = str(user["id"])
            user_action = "exists"

            if args.promote_admin and user.get("role") != "admin":
                conn.execute(
                    "UPDATE users SET role = 'admin' WHERE id = %s",
                    (user_id,),
                )
                user_action = "promoted"

            if args.reset_password:
                pw_hash = get_password_hash(args.admin_password)
                conn.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (pw_hash, user_id),
                )
                user_action = "password_reset" if user_action == "exists" else user_action

        conn.commit()

    print(f"Tenant: {tenant_action} id={tenant_id} name={args.tenant_name}")
    print(f"User: {user_action} id={user_id} email={args.admin_email}")


if __name__ == "__main__":
    main()

