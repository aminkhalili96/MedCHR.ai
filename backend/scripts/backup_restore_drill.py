from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.db import get_conn

CRITICAL_TABLES = (
    "tenants",
    "users",
    "patients",
    "documents",
    "extractions",
    "api_keys",
    "audit_events",
    "retention_manifests",
)


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) AS name", (f"public.{table_name}",)).fetchone()
    return bool(row and row.get("name"))


def _run_table_copy_drill(conn, table_name: str, sample_limit: int) -> dict[str, Any]:
    source_count = int(conn.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}").fetchone()["cnt"])
    temp_table = f"drill_{table_name}_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE TEMP TABLE {temp_table} AS TABLE {table_name} WITH NO DATA")
    conn.execute(
        f"INSERT INTO {temp_table} SELECT * FROM {table_name} LIMIT %s",
        (max(1, int(sample_limit)),),
    )
    restored_count = int(conn.execute(f"SELECT COUNT(*) AS cnt FROM {temp_table}").fetchone()["cnt"])
    expected = min(source_count, max(1, int(sample_limit)))
    return {
        "table": table_name,
        "source_rows": source_count,
        "restored_rows": restored_count,
        "expected_rows": expected,
        "status": "PASS" if restored_count == expected else "FAIL",
    }


def run_backup_restore_drill(*, sample_limit: int = 100) -> dict[str, Any]:
    started = time.perf_counter()
    table_results: list[dict[str, Any]] = []
    errors: list[str] = []
    available_tables = 0

    with get_conn() as conn:
        for table in CRITICAL_TABLES:
            if not _table_exists(conn, table):
                table_results.append(
                    {
                        "table": table,
                        "status": "SKIP",
                        "reason": "table_not_found",
                    }
                )
                continue
            available_tables += 1
            try:
                result = _run_table_copy_drill(conn, table, sample_limit=sample_limit)
                table_results.append(result)
            except Exception as exc:
                table_results.append({"table": table, "status": "FAIL", "reason": str(exc)})
                errors.append(f"{table}: {exc}")
        conn.commit()

    failed = [item for item in table_results if item.get("status") == "FAIL"]
    elapsed = round(time.perf_counter() - started, 3)
    return {
        "drill_type": "logical_backup_restore_table_copy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": elapsed,
        "sample_limit": int(sample_limit),
        "tables_checked": len(table_results),
        "tables_available": available_tables,
        "failed_tables": len(failed),
        "overall_status": "FAIL" if failed else "PASS",
        "errors": errors,
        "table_results": table_results,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Backup/Restore Drill Report",
        "",
        f"- Generated at (UTC): {report['generated_at']}",
        f"- Drill type: {report['drill_type']}",
        f"- Overall status: {report['overall_status']}",
        f"- Duration: {report['duration_seconds']}s",
        f"- Tables checked: {report['tables_checked']} (available={report['tables_available']})",
        f"- Failed tables: {report['failed_tables']}",
        "",
        "## Table Results",
    ]
    for item in report["table_results"]:
        if item["status"] == "PASS":
            lines.append(
                f"- [PASS] {item['table']} source={item['source_rows']} restored={item['restored_rows']}"
            )
        elif item["status"] == "SKIP":
            lines.append(f"- [SKIP] {item['table']} reason={item.get('reason')}")
        else:
            lines.append(f"- [FAIL] {item['table']} reason={item.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backup/restore drill and emit machine-readable artifact.")
    parser.add_argument("--output-dir", default="data/drills/backup_restore")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = run_backup_restore_drill(sample_limit=args.sample_limit)

    json_path = output_dir / f"backup_restore_drill_{stamp}.json"
    md_path = output_dir / f"backup_restore_drill_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Backup/restore drill report written: {json_path}")
    print(f"Backup/restore drill markdown: {md_path}")

    if args.fail_on_errors and report["overall_status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
