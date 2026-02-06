from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from psycopg.types.json import Json

from backend.app.config import get_settings
from backend.app.db import get_conn


_TIME_COLUMNS = {
    "audit_logs": "created_at",
    "audit_events": "event_time",
    "phi_egress_events": "event_time",
    "jobs": "created_at",
}

_IMMUTABLE_EXPORT_TABLES = ("audit_logs", "audit_events", "phi_egress_events")


def _export_rows(table: str, days: int, export_dir: str, *, dry_run: bool) -> tuple[int, str | None, str | None]:
    if table not in _TIME_COLUMNS:
        return 0, None, None
    time_column = _TIME_COLUMNS[table]
    try:
        with get_conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE {time_column} < NOW() - (%s * INTERVAL '1 day')",
                (days,),
            ).fetchone()
            count = int(row["cnt"]) if row else 0
            if dry_run or count == 0:
                return count, None, None
            rows = conn.execute(
                f"""
                SELECT row_to_json(t) AS payload
                FROM (
                  SELECT *
                  FROM {table}
                  WHERE {time_column} < NOW() - (%s * INTERVAL '1 day')
                  ORDER BY {time_column} ASC
                ) t
                """,
                (days,),
            ).fetchall()
    except Exception:
        return 0, None, None

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_file = export_path / f"{table}_{timestamp}.jsonl"

    hasher = hashlib.sha256()
    with output_file.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = row.get("payload") if isinstance(row, dict) else row[0]
            line = json.dumps(payload, separators=(",", ":"), default=str)
            fh.write(line + "\n")
            hasher.update((line + "\n").encode("utf-8"))

    checksum = hasher.hexdigest()
    checksum_file = output_file.with_suffix(output_file.suffix + ".sha256")
    checksum_file.write_text(f"{checksum}  {output_file.name}\n", encoding="utf-8")
    return count, str(output_file), checksum


def _copy_to_immutable_sink(output_file: str, immutable_dir: str) -> str:
    src = Path(output_file)
    checksum_src = src.with_suffix(src.suffix + ".sha256")
    sink_root = Path(immutable_dir)
    sink_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    sink_dir = sink_root / stamp
    sink_dir.mkdir(parents=True, exist_ok=True)

    target_file = sink_dir / src.name
    target_checksum = sink_dir / checksum_src.name
    shutil.copy2(src, target_file)
    shutil.copy2(checksum_src, target_checksum)

    target_file.chmod(0o444)
    target_checksum.chmod(0o444)
    return f"file://{target_file.resolve()}"


def _insert_retention_manifest(
    *,
    table: str,
    retention_days: int,
    row_count: int,
    export_uri: str,
    checksum: str,
    immutable_confirmed: bool,
    metadata: dict,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO retention_manifests (
                table_name,
                retention_days,
                row_count,
                export_uri,
                checksum,
                immutable_confirmed,
                manifest
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                table,
                retention_days,
                row_count,
                export_uri,
                checksum,
                immutable_confirmed,
                Json(metadata),
            ),
        )
        conn.commit()


def purge_table(table: str, days: int, dry_run: bool, *, time_column: str = "created_at") -> int:
    try:
        with get_conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM {table} WHERE {time_column} < NOW() - (%s * INTERVAL '1 day')",
                (days,),
            ).fetchone()
            count = int(row["cnt"]) if row else 0
            if not dry_run and count:
                conn.execute(
                    f"DELETE FROM {table} WHERE {time_column} < NOW() - (%s * INTERVAL '1 day')",
                    (days,),
                )
                conn.commit()
        return count
    except Exception:
        return 0


def purge_jobs(days: int, dry_run: bool) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM jobs
            WHERE status IN ('done', 'failed')
              AND created_at < NOW() - (%s * INTERVAL '1 day')
            """,
            (days,),
        ).fetchone()
        count = int(row["cnt"]) if row else 0
        if not dry_run and count:
            conn.execute(
                """
                DELETE FROM jobs
                WHERE status IN ('done', 'failed')
                  AND created_at < NOW() - (%s * INTERVAL '1 day')
                """,
                (days,),
            )
            conn.commit()
    return count


def purge_mfa_setup_tokens(dry_run: bool) -> int:
    try:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM mfa_setup_tokens
                WHERE used_at IS NOT NULL OR expires_at < NOW()
                """
            ).fetchone()
            count = int(row["cnt"]) if row else 0
            if not dry_run and count:
                conn.execute(
                    "DELETE FROM mfa_setup_tokens WHERE used_at IS NOT NULL OR expires_at < NOW()"
                )
                conn.commit()
        return count
    except Exception:
        return 0


def purge_ui_sessions(dry_run: bool) -> int:
    try:
        with get_conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM ui_sessions
                WHERE expires_at < NOW() OR revoked_at IS NOT NULL
                """
            ).fetchone()
            count = int(row["cnt"]) if row else 0
            if not dry_run and count:
                conn.execute(
                    "DELETE FROM ui_sessions WHERE expires_at < NOW() OR revoked_at IS NOT NULL"
                )
                conn.commit()
        return count
    except Exception:
        return 0


def _enforce_immutable_exports(
    *,
    immutable_dir: str,
    audit_days: int,
    export_counts: dict[str, int],
    export_files: dict[str, str | None],
    checksums: dict[str, str | None],
) -> dict[str, str]:
    has_export_candidates = any(export_counts.get(table, 0) > 0 for table in _IMMUTABLE_EXPORT_TABLES)
    if not has_export_candidates:
        return {}
    if not immutable_dir:
        raise RuntimeError("Immutable retention sink is required for execute mode when PHI audit data is purged.")

    manifest_uris: dict[str, str] = {}
    for table in _IMMUTABLE_EXPORT_TABLES:
        count = export_counts.get(table, 0)
        output_file = export_files.get(table)
        checksum = checksums.get(table)
        if count <= 0 or not output_file or not checksum:
            continue
        uri = _copy_to_immutable_sink(output_file, immutable_dir)
        _insert_retention_manifest(
            table=table,
            retention_days=audit_days,
            row_count=count,
            export_uri=uri,
            checksum=checksum,
            immutable_confirmed=True,
            metadata={
                "source_file": output_file,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "immutable_dir": immutable_dir,
            },
        )
        manifest_uris[table] = uri
    return manifest_uris


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Purge old audit logs and job history.")
    parser.add_argument("--audit-days", type=int, default=settings.audit_retention_days)
    parser.add_argument("--job-days", type=int, default=settings.job_retention_days)
    parser.add_argument("--execute", action="store_true", help="Apply deletions (default is dry-run).")
    parser.add_argument(
        "--export-dir",
        default=settings.retention_export_dir,
        help="Directory for retention exports before purge.",
    )
    parser.add_argument(
        "--immutable-dir",
        default=settings.retention_immutable_dir,
        help="Directory used as immutable retention sink for audit exports.",
    )
    args = parser.parse_args()

    dry_run = not args.execute
    audit_export_count, audit_export_file, audit_checksum = _export_rows(
        "audit_logs", args.audit_days, args.export_dir, dry_run=dry_run
    )
    audit_events_export_count, audit_events_export_file, audit_events_checksum = _export_rows(
        "audit_events", args.audit_days, args.export_dir, dry_run=dry_run
    )
    phi_export_count, phi_export_file, phi_checksum = _export_rows(
        "phi_egress_events", args.audit_days, args.export_dir, dry_run=dry_run
    )

    export_counts = {
        "audit_logs": audit_export_count,
        "audit_events": audit_events_export_count,
        "phi_egress_events": phi_export_count,
    }
    export_files = {
        "audit_logs": audit_export_file,
        "audit_events": audit_events_export_file,
        "phi_egress_events": phi_export_file,
    }
    checksums = {
        "audit_logs": audit_checksum,
        "audit_events": audit_events_checksum,
        "phi_egress_events": phi_checksum,
    }

    manifest_uris: dict[str, str] = {}
    if not dry_run:
        try:
            manifest_uris = _enforce_immutable_exports(
                immutable_dir=args.immutable_dir,
                audit_days=args.audit_days,
                export_counts=export_counts,
                export_files=export_files,
                checksums=checksums,
            )
        except Exception as exc:
            raise SystemExit(f"Purge aborted: immutable export confirmation failed: {exc}") from exc

    audit_count = purge_table("audit_logs", args.audit_days, dry_run=dry_run, time_column="created_at")
    audit_events_count = purge_table("audit_events", args.audit_days, dry_run=dry_run, time_column="event_time")
    phi_events_count = purge_table("phi_egress_events", args.audit_days, dry_run=dry_run, time_column="event_time")
    job_count = purge_jobs(args.job_days, dry_run=dry_run)
    mfa_count = purge_mfa_setup_tokens(dry_run=dry_run)
    session_count = purge_ui_sessions(dry_run=dry_run)

    mode = "DRY-RUN" if dry_run else "APPLIED"
    print(f"{mode}: audit_logs purge candidates: {audit_count}")
    print(f"{mode}: audit_events purge candidates: {audit_events_count}")
    print(f"{mode}: phi_egress_events purge candidates: {phi_events_count}")
    if dry_run:
        print(f"DRY-RUN: audit_logs export candidates: {audit_export_count}")
        print(f"DRY-RUN: audit_events export candidates: {audit_events_export_count}")
        print(f"DRY-RUN: phi_egress_events export candidates: {phi_export_count}")
    else:
        print(f"APPLIED: audit_logs export file: {audit_export_file or 'none'}")
        print(f"APPLIED: audit_events export file: {audit_events_export_file or 'none'}")
        print(f"APPLIED: phi_egress_events export file: {phi_export_file or 'none'}")
        print(f"APPLIED: audit_logs immutable URI: {manifest_uris.get('audit_logs', 'none')}")
        print(f"APPLIED: audit_events immutable URI: {manifest_uris.get('audit_events', 'none')}")
        print(f"APPLIED: phi_egress_events immutable URI: {manifest_uris.get('phi_egress_events', 'none')}")
    print(f"{mode}: jobs purge candidates: {job_count}")
    print(f"{mode}: mfa_setup_tokens purge candidates: {mfa_count}")
    print(f"{mode}: ui_sessions purge candidates: {session_count}")


if __name__ == "__main__":
    main()
