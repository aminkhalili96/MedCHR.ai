from __future__ import annotations

import hashlib
import os
from pathlib import Path

from backend.app.db import get_conn


def _checksum_for(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _ensure_migrations_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          version TEXT PRIMARY KEY,
          applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          checksum TEXT,
          applied_by TEXT
        )
        """
    )
    conn.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum TEXT")
    conn.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS applied_by TEXT")


def _applied_versions(conn) -> dict[str, str | None]:
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {row["version"]: row.get("checksum") for row in rows}


def _migration_files() -> list[Path]:
    base = Path(__file__).resolve().parents[1] / "sql" / "migrations"
    return sorted(base.glob("*.sql"))


def main() -> None:
    migrations = _migration_files()
    if not migrations:
        print("No migrations found.")
        return

    applied_by = os.getenv("MIGRATION_APPLIED_BY", "backend.scripts.migrate")
    with get_conn() as conn:
        _ensure_migrations_table(conn)
        applied = _applied_versions(conn)

        for path in migrations:
            version = path.stem
            sql = path.read_text(encoding="utf-8")
            checksum = _checksum_for(sql)
            recorded_checksum = applied.get(version)

            if version in applied:
                if recorded_checksum and recorded_checksum != checksum:
                    raise RuntimeError(
                        f"Migration checksum mismatch for {version}. "
                        "Historical migration was modified after apply."
                    )
                if not recorded_checksum:
                    conn.execute(
                        "UPDATE schema_migrations SET checksum = %s, applied_by = COALESCE(applied_by, %s) WHERE version = %s",
                        (checksum, applied_by, version),
                    )
                continue

            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, checksum, applied_by) VALUES (%s, %s, %s)",
                (version, checksum, applied_by),
            )
        conn.commit()

    print("Migrations applied.")


if __name__ == "__main__":
    main()
