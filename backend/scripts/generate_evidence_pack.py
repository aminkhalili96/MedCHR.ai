from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.app.config import get_settings
from backend.app.db import get_conn
from backend.scripts.validate_controls import collect_control_validation
from backend.scripts.verify_audit_chain import verify_chain


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    row = conn.execute(
        """
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.columns
          WHERE table_schema = 'public'
            AND table_name = %s
            AND column_name = %s
        ) AS present
        """,
        (table_name, column_name),
    ).fetchone()
    return bool(row and row.get("present"))


def _load_latest_backup_restore_drill(drill_dir: Path) -> dict:
    if not drill_dir.exists():
        return {
            "status": "WARN",
            "message": f"No backup/restore drill directory found at {drill_dir}",
            "latest_report": None,
        }

    reports = sorted(drill_dir.glob("backup_restore_drill_*.json"), reverse=True)
    if not reports:
        return {
            "status": "WARN",
            "message": "No backup/restore drill report found.",
            "latest_report": None,
        }

    latest = reports[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "FAIL",
            "message": f"Failed to parse drill report {latest}: {exc}",
            "latest_report": str(latest),
        }
    status = "PASS" if payload.get("overall_status") == "PASS" else "FAIL"
    return {
        "status": status,
        "message": f"Latest drill report: {latest.name}",
        "latest_report": str(latest),
        "report": payload,
    }


def _retention_conformance(conn) -> dict:
    settings = get_settings()
    audit_days = int(settings.audit_retention_days)
    job_days = int(settings.job_retention_days)

    audit_overdue_row = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM audit_events
        WHERE created_at < NOW() - make_interval(days => %s)
        """,
        (audit_days,),
    ).fetchone()
    job_overdue_row = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM jobs
        WHERE created_at < NOW() - make_interval(days => %s)
        """,
        (job_days,),
    ).fetchone()
    manifest_rows = conn.execute(
        """
        SELECT table_name, MAX(created_at) AS latest_created_at,
               BOOL_OR(immutable_confirmed) AS immutable_confirmed
        FROM retention_manifests
        GROUP BY table_name
        ORDER BY table_name
        """
    ).fetchall()

    audit_overdue = int(audit_overdue_row.get("cnt", 0)) if audit_overdue_row else 0
    job_overdue = int(job_overdue_row.get("cnt", 0)) if job_overdue_row else 0
    overdue_total = audit_overdue + job_overdue
    status = "PASS" if overdue_total == 0 else "WARN"
    return {
        "status": status,
        "audit_retention_days": audit_days,
        "job_retention_days": job_days,
        "audit_overdue_rows": audit_overdue,
        "job_overdue_rows": job_overdue,
        "manifest_latest": [dict(row) for row in manifest_rows],
    }


def _collect_summary() -> dict:
    audit_ok, audit_errors = verify_chain()
    control_validation = collect_control_validation(require_db=True)
    with get_conn() as conn:
        manifest_rows = conn.execute(
            """
            SELECT table_name, retention_days, row_count, export_uri, checksum, immutable_confirmed, created_at
            FROM retention_manifests
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()
        user_rows = conn.execute(
            """
            SELECT tenant_id::text as tenant_id, role, COUNT(*) as count
            FROM users
            GROUP BY tenant_id, role
            ORDER BY tenant_id, role
            """
        ).fetchall()
        key_rows = conn.execute(
            """
            SELECT tenant_id::text as tenant_id, COUNT(*) as active_keys
            FROM api_keys
            WHERE revoked_at IS NULL
            GROUP BY tenant_id
            ORDER BY tenant_id
            """
        ).fetchall()
        stale_key_rows = conn.execute(
            """
            SELECT id::text AS key_id,
                   tenant_id::text AS tenant_id,
                   name,
                   created_at,
                   last_used_at
            FROM api_keys
            WHERE revoked_at IS NULL
              AND (
                last_used_at IS NULL
                OR last_used_at < NOW() - INTERVAL '90 days'
              )
            ORDER BY COALESCE(last_used_at, created_at) ASC
            LIMIT 200
            """
        ).fetchall()
        has_last_login = _column_exists(conn, "users", "last_login")
        if has_last_login:
            privileged_user_sql = """
                SELECT id::text AS user_id,
                       tenant_id::text AS tenant_id,
                       email,
                       role,
                       created_at,
                       last_login
                FROM users
                WHERE role = 'admin'
                ORDER BY tenant_id, email
            """
        else:
            privileged_user_sql = """
                SELECT id::text AS user_id,
                       tenant_id::text AS tenant_id,
                       email,
                       role,
                       created_at,
                       NULL::timestamptz AS last_login
                FROM users
                WHERE role = 'admin'
                ORDER BY tenant_id, email
            """
        privileged_user_rows = conn.execute(privileged_user_sql).fetchall()
        migration_rows = conn.execute(
            """
            SELECT version, checksum, applied_at
            FROM schema_migrations
            ORDER BY applied_at DESC
            """
        ).fetchall()
        retention_conformance = _retention_conformance(conn)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "control_validation": control_validation,
        "audit_chain": {"ok": audit_ok, "errors": audit_errors},
        "retention_manifests": [dict(row) for row in manifest_rows],
        "access_review": {
            "users_by_tenant_role": [dict(row) for row in user_rows],
            "active_api_keys_by_tenant": [dict(row) for row in key_rows],
            "stale_api_keys": [dict(row) for row in stale_key_rows],
            "privileged_users": [dict(row) for row in privileged_user_rows],
        },
        "retention_conformance": retention_conformance,
        "migration_integrity": [dict(row) for row in migration_rows],
        "backup_restore_drill": _load_latest_backup_restore_drill(
            Path("data/drills/backup_restore")
        ),
    }


def _render_markdown(summary: dict) -> str:
    lines = []
    lines.append("# MedCHR Evidence Pack")
    lines.append("")
    lines.append(f"- Generated at (UTC): {summary['generated_at']}")
    lines.append(f"- Control validation overall: {summary['control_validation']['overall_status']}")
    lines.append(
        "- Control checks (PASS/FAIL/WARN/SKIP): "
        f"{summary['control_validation']['summary']['PASS']}/"
        f"{summary['control_validation']['summary']['FAIL']}/"
        f"{summary['control_validation']['summary']['WARN']}/"
        f"{summary['control_validation']['summary']['SKIP']}"
    )
    lines.append(f"- Audit chain integrity: {'PASS' if summary['audit_chain']['ok'] else 'FAIL'}")
    if summary["audit_chain"]["errors"]:
        lines.append("- Audit chain errors:")
        for err in summary["audit_chain"]["errors"]:
            lines.append(f"  - {err}")
    lines.append("")
    lines.append("## Control Validation")
    failing_checks = [check for check in summary["control_validation"]["checks"] if check["status"] == "FAIL"]
    warning_checks = [check for check in summary["control_validation"]["checks"] if check["status"] == "WARN"]
    lines.append(f"- Failing checks: {len(failing_checks)}")
    lines.append(f"- Warning checks: {len(warning_checks)}")
    lines.append("")
    lines.append("## Retention Manifests (latest 100)")
    lines.append(f"- Count: {len(summary['retention_manifests'])}")
    lines.append("")
    lines.append("## Access Review Snapshot")
    lines.append(f"- User role rows: {len(summary['access_review']['users_by_tenant_role'])}")
    lines.append(f"- Active key rows: {len(summary['access_review']['active_api_keys_by_tenant'])}")
    lines.append(f"- Stale active keys (>90 days unused): {len(summary['access_review']['stale_api_keys'])}")
    lines.append(f"- Privileged users (admin role): {len(summary['access_review']['privileged_users'])}")
    lines.append("")
    lines.append("## Retention Conformance")
    lines.append(f"- Status: {summary['retention_conformance']['status']}")
    lines.append(f"- Overdue audit rows: {summary['retention_conformance']['audit_overdue_rows']}")
    lines.append(f"- Overdue job rows: {summary['retention_conformance']['job_overdue_rows']}")
    lines.append("")
    lines.append("## Backup/Restore Drill")
    lines.append(f"- Status: {summary['backup_restore_drill']['status']}")
    lines.append(f"- Note: {summary['backup_restore_drill']['message']}")
    lines.append("")
    lines.append("## Migration Integrity")
    lines.append(f"- Tracked migrations: {len(summary['migration_integrity'])}")
    return "\n".join(lines) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(pack_dir: Path, files: list[Path]) -> Path:
    settings = get_settings()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "name": file.name,
                "size_bytes": file.stat().st_size,
                "sha256": _sha256_file(file),
            }
            for file in files
        ],
    }
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(
        settings.app_secret_key.encode("utf-8"),
        canonical_payload,
        hashlib.sha256,
    ).hexdigest()
    manifest = {
        "payload": payload,
        "signature": {
            "algorithm": "HMAC-SHA256",
            "value": signature,
        },
    }
    manifest_path = pack_dir / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _prune_old_packs(output_root: Path, keep_latest: int) -> None:
    if keep_latest <= 0:
        return
    pack_dirs = sorted([item for item in output_root.iterdir() if item.is_dir()], reverse=True)
    for stale in pack_dirs[keep_latest:]:
        shutil.rmtree(stale, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compliance evidence pack artifacts.")
    parser.add_argument(
        "--output-dir",
        default="data/evidence_packs",
        help="Directory where evidence pack files are generated.",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=12,
        help="Number of latest evidence pack directories to keep in output-dir.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pack_dir = output_root / stamp
    pack_dir.mkdir(parents=True, exist_ok=True)

    summary = _collect_summary()
    json_path = pack_dir / "evidence_report.json"
    md_path = pack_dir / "evidence_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(summary), encoding="utf-8")
    manifest_path = _write_manifest(pack_dir, [json_path, md_path])
    _prune_old_packs(output_root, int(args.keep_latest))

    print(f"Evidence pack generated: {pack_dir}")
    print(f"- JSON: {json_path}")
    print(f"- Markdown: {md_path}")
    print(f"- Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
