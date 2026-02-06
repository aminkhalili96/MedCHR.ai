from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.config import get_settings
from backend.app.db import get_conn
from backend.scripts.validate_llm_gateway_usage import find_llm_gateway_violations
from backend.scripts.validate_runtime_policy import validate_runtime_policy
from backend.scripts.verify_audit_chain import verify_chain

INSECURE_APP_SECRETS = {"", "dev-secret", "change-me", "changeme", "default"}
REQUIRED_RUNBOOKS = (
    "doc/runbooks/key_rotation.md",
    "doc/runbooks/access_review.md",
    "doc/runbooks/incident_response.md",
    "doc/runbooks/breach_notification.md",
)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _count_scoped_env_api_keys(raw_value: str | None) -> tuple[int, int]:
    scoped = 0
    unscoped = 0
    for token in _split_csv(raw_value):
        if ":" not in token:
            unscoped += 1
            continue
        tenant, key = token.split(":", 1)
        if tenant.strip() and key.strip():
            scoped += 1
        else:
            unscoped += 1
    return scoped, unscoped


def _add_check(
    checks: list[dict[str, Any]],
    *,
    control_id: str,
    family: str,
    severity: str,
    title: str,
    status: str,
    evidence: str,
    remediation: str,
) -> None:
    checks.append(
        {
            "id": control_id,
            "family": family,
            "severity": severity,
            "title": title,
            "status": status,
            "evidence": evidence,
            "remediation": remediation,
        }
    )


def _status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
    for check in checks:
        status = str(check.get("status", "SKIP")).upper()
        counts[status] = counts.get(status, 0) + 1
    return counts


def _db_ready() -> tuple[bool, str]:
    try:
        with get_conn() as conn:
            conn.execute("SELECT 1")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _table_exists(conn, table_name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) AS name", (f"public.{table_name}",)).fetchone()
    return bool(row and row.get("name"))


def collect_control_validation(*, require_db: bool = False) -> dict[str, Any]:
    settings = get_settings()
    checks: list[dict[str, Any]] = []
    prod_like = settings.app_env == "prod" or settings.hipaa_mode

    if settings.app_env == "prod" and not settings.hipaa_mode:
        _add_check(
            checks,
            control_id="C-APP-001",
            family="application",
            severity="high",
            title="HIPAA mode is enabled for production runtime",
            status="FAIL",
            evidence=f"APP_ENV={settings.app_env}, HIPAA_MODE={settings.hipaa_mode}",
            remediation="Set HIPAA_MODE=true for production environments.",
        )
    else:
        _add_check(
            checks,
            control_id="C-APP-001",
            family="application",
            severity="high",
            title="HIPAA mode is enabled for production runtime",
            status="PASS",
            evidence=f"APP_ENV={settings.app_env}, HIPAA_MODE={settings.hipaa_mode}",
            remediation="Keep HIPAA_MODE true in production.",
        )

    app_secret = (settings.app_secret_key or "").strip()
    insecure_secret = app_secret in INSECURE_APP_SECRETS
    app_secret_status = "FAIL" if insecure_secret and prod_like else ("WARN" if insecure_secret else "PASS")
    _add_check(
        checks,
        control_id="C-APP-002",
        family="application",
        severity="high",
        title="Application secret is non-default",
        status=app_secret_status,
        evidence="APP_SECRET_KEY is set." if app_secret and not insecure_secret else "APP_SECRET_KEY is missing or default.",
        remediation="Use a strong, rotated secret from managed secret storage.",
    )

    mfa_secret = (settings.mfa_secret_key or "").strip()
    mfa_secret_status = "FAIL" if not mfa_secret and prod_like else ("WARN" if not mfa_secret else "PASS")
    _add_check(
        checks,
        control_id="C-APP-003",
        family="application",
        severity="medium",
        title="Dedicated MFA encryption key is configured",
        status=mfa_secret_status,
        evidence="MFA_SECRET_KEY is configured." if mfa_secret else "MFA_SECRET_KEY is empty; APP_SECRET_KEY fallback is in use.",
        remediation="Configure MFA_SECRET_KEY via managed secrets and rotate periodically.",
    )

    if settings.trust_proxy_headers and not _split_csv(settings.trusted_proxy_ips):
        _add_check(
            checks,
            control_id="C-NET-001",
            family="network",
            severity="high",
            title="Proxy header trust is restricted to explicit proxy IPs/CIDRs",
            status="FAIL",
            evidence="TRUST_PROXY_HEADERS=true with empty TRUSTED_PROXY_IPS.",
            remediation="Set TRUSTED_PROXY_IPS to ingress/load-balancer source addresses.",
        )
    else:
        _add_check(
            checks,
            control_id="C-NET-001",
            family="network",
            severity="high",
            title="Proxy header trust is restricted to explicit proxy IPs/CIDRs",
            status="PASS",
            evidence=(
                "Proxy header trust disabled."
                if not settings.trust_proxy_headers
                else f"Trusted proxy entries: {len(_split_csv(settings.trusted_proxy_ips))}"
            ),
            remediation="Keep forwarding header trust boundaries explicit.",
        )

    processors = _split_csv(settings.phi_processors)
    if settings.hipaa_mode and not processors:
        _add_check(
            checks,
            control_id="C-PHI-001",
            family="phi_egress",
            severity="high",
            title="PHI processor allowlist is configured",
            status="FAIL",
            evidence="HIPAA_MODE=true and PHI_PROCESSORS is empty.",
            remediation="Set PHI_PROCESSORS to BAA-covered processors only.",
        )
    else:
        _add_check(
            checks,
            control_id="C-PHI-001",
            family="phi_egress",
            severity="high",
            title="PHI processor allowlist is configured",
            status="PASS",
            evidence=f"Configured processors: {processors}" if processors else "HIPAA mode is off in this runtime.",
            remediation="Keep processor allowlist synced with BAA-approved vendors.",
        )

    if prod_like and not settings.retention_immutable_dir:
        _add_check(
            checks,
            control_id="C-RET-001",
            family="retention",
            severity="high",
            title="Immutable retention export target is configured",
            status="FAIL",
            evidence="RETENTION_IMMUTABLE_DIR is empty in prod/HIPAA runtime.",
            remediation="Configure immutable retention export target before purge execution.",
        )
    else:
        _add_check(
            checks,
            control_id="C-RET-001",
            family="retention",
            severity="high",
            title="Immutable retention export target is configured",
            status="PASS",
            evidence=f"RETENTION_IMMUTABLE_DIR={settings.retention_immutable_dir or '(not required in current runtime)'}",
            remediation="Keep immutable exports enabled for retention workflows.",
        )

    missing_runbooks = [path for path in REQUIRED_RUNBOOKS if not Path(path).exists()]
    _add_check(
        checks,
        control_id="C-OPS-001",
        family="operations",
        severity="medium",
        title="Core compliance runbooks are present",
        status="FAIL" if missing_runbooks else "PASS",
        evidence="Missing runbooks: " + ", ".join(missing_runbooks) if missing_runbooks else "All required runbooks found.",
        remediation="Create and maintain incident, breach, key rotation, and access review runbooks.",
    )

    llm_import_violations = find_llm_gateway_violations(scan_root=Path("backend/app"))
    _add_check(
        checks,
        control_id="C-APP-004",
        family="application",
        severity="high",
        title="Outbound OpenAI usage is centralized in llm_gateway",
        status="FAIL" if llm_import_violations else "PASS",
        evidence="Violations: " + "; ".join(
            f"{item['file']} ({len(item['violations'])})" for item in llm_import_violations
        )
        if llm_import_violations
        else "No direct OpenAI imports outside llm_gateway.",
        remediation="Route all outbound model calls through backend/app/llm_gateway.py.",
    )

    runtime_policy = validate_runtime_policy(
        dockerfile_path=Path("Dockerfile"),
        requirements_path=Path("backend/requirements.txt"),
    )
    runtime_failures = [item["id"] for item in runtime_policy["checks"] if item["status"] == "FAIL"]
    _add_check(
        checks,
        control_id="C-RUNTIME-001",
        family="runtime",
        severity="high",
        title="Runtime policy checks pass",
        status="FAIL" if runtime_failures else "PASS",
        evidence=f"Runtime policy failures: {runtime_failures}" if runtime_failures else "Runtime policy checks passed.",
        remediation="Fix failing runtime policy controls before release.",
    )

    scoped_env_keys, unscoped_env_keys = _count_scoped_env_api_keys(
        getattr(settings, "api_keys", None)
    )

    db_ok, db_error = _db_ready()
    if not db_ok:
        db_status = "FAIL" if require_db else "WARN"
        _add_check(
            checks,
            control_id="C-DB-000",
            family="database",
            severity="high",
            title="Database connectivity for control verification",
            status=db_status,
            evidence=f"Database unavailable: {db_error}",
            remediation="Provide database connectivity for full control verification.",
        )

        key_status = "PASS" if scoped_env_keys > 0 else "WARN"
        if key_status == "PASS" and prod_like and unscoped_env_keys > 0:
            key_status = "WARN"
        _add_check(
            checks,
            control_id="C-IAM-001",
            family="identity",
            severity="medium",
            title="At least one active tenant-scoped API key source exists",
            status=key_status,
            evidence=(
                f"active_db_api_keys=unavailable, scoped_env_api_keys={scoped_env_keys}, "
                f"unscoped_env_api_keys={unscoped_env_keys}"
            ),
            remediation=(
                "Provision tenant-scoped API keys (DB-backed and/or scoped env keys). "
                "Avoid unscoped API keys in production/HIPAA mode."
            ),
        )
    else:
        _add_check(
            checks,
            control_id="C-DB-000",
            family="database",
            severity="high",
            title="Database connectivity for control verification",
            status="PASS",
            evidence="Database connection succeeded.",
            remediation="Keep runtime DB health checks and credentials managed.",
        )

        with get_conn() as conn:
            required_tables = (
                "api_keys",
                "audit_events",
                "mfa_lockouts",
                "retention_manifests",
                "schema_migrations",
                "ui_sessions",
            )
            missing_tables = [table for table in required_tables if not _table_exists(conn, table)]
            _add_check(
                checks,
                control_id="C-DB-001",
                family="database",
                severity="high",
                title="Required security/compliance tables are present",
                status="FAIL" if missing_tables else "PASS",
                evidence="Missing tables: " + ", ".join(missing_tables) if missing_tables else "All required tables are present.",
                remediation="Apply migrations and verify schema integrity in all environments.",
            )

            migration_row = conn.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  COUNT(*) FILTER (WHERE checksum IS NOT NULL) AS with_checksum
                FROM schema_migrations
                """
            ).fetchone()
            total = int(migration_row.get("total", 0)) if migration_row else 0
            with_checksum = int(migration_row.get("with_checksum", 0)) if migration_row else 0
            _add_check(
                checks,
                control_id="C-DB-002",
                family="database",
                severity="high",
                title="Applied migrations have integrity checksums",
                status="PASS" if total == with_checksum else "FAIL",
                evidence=f"migrations={total}, with_checksum={with_checksum}",
                remediation="Backfill or repair missing migration checksums before release.",
            )

            active_keys_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM api_keys WHERE revoked_at IS NULL"
            ).fetchone()
            active_keys = int(active_keys_row.get("cnt", 0)) if active_keys_row else 0
            configured_key_sources = active_keys + scoped_env_keys
            if prod_like and configured_key_sources == 0:
                key_status = "FAIL"
            elif configured_key_sources == 0:
                key_status = "WARN"
            elif prod_like and unscoped_env_keys > 0:
                key_status = "WARN"
            else:
                key_status = "PASS"
            _add_check(
                checks,
                control_id="C-IAM-001",
                family="identity",
                severity="medium",
                title="At least one active tenant-scoped API key source exists",
                status=key_status,
                evidence=(
                    f"active_db_api_keys={active_keys}, scoped_env_api_keys={scoped_env_keys}, "
                    f"unscoped_env_api_keys={unscoped_env_keys}"
                ),
                remediation=(
                    "Provision tenant-scoped API keys (DB-backed and/or scoped env keys). "
                    "Avoid unscoped API keys in production/HIPAA mode."
                ),
            )

        audit_ok, audit_errors = verify_chain()
        _add_check(
            checks,
            control_id="C-DB-003",
            family="audit",
            severity="high",
            title="Audit hash chain verifies successfully",
            status="PASS" if audit_ok else "FAIL",
            evidence="Audit chain verified." if audit_ok else "; ".join(audit_errors),
            remediation="Run verifier, identify chain divergence, and repair before go-live.",
        )

    counts = _status_counts(checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "FAIL" if counts.get("FAIL", 0) > 0 else "PASS",
        "summary": counts,
        "checks": checks,
    }
    return report


def render_control_validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cloud-Agnostic Healthcare Control Validation",
        "",
        f"- Generated at (UTC): {report.get('generated_at')}",
        f"- Overall status: {report.get('overall_status')}",
    ]
    summary = report.get("summary", {})
    lines.append(
        f"- Checks: PASS={summary.get('PASS', 0)}, FAIL={summary.get('FAIL', 0)}, WARN={summary.get('WARN', 0)}, SKIP={summary.get('SKIP', 0)}"
    )
    lines.append("")
    lines.append("## Control Results")
    for check in report.get("checks", []):
        lines.append(
            f"- [{check.get('status')}] {check.get('id')} ({check.get('severity')}) {check.get('title')}"
        )
        lines.append(f"  - Evidence: {check.get('evidence')}")
        lines.append(f"  - Remediation: {check.get('remediation')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate cloud-agnostic healthcare-grade controls.")
    parser.add_argument("--output-json", help="Optional output path for JSON report.")
    parser.add_argument("--output-markdown", help="Optional output path for Markdown report.")
    parser.add_argument(
        "--require-db",
        action="store_true",
        help="Treat missing DB connectivity as failure instead of warning.",
    )
    parser.add_argument(
        "--fail-on-failures",
        action="store_true",
        help="Exit non-zero when any controls fail.",
    )
    args = parser.parse_args()

    report = collect_control_validation(require_db=args.require_db)
    markdown = render_control_validation_markdown(report)
    json_text = json.dumps(report, indent=2, default=str)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json_text, encoding="utf-8")

    if args.output_markdown:
        output_md = Path(args.output_markdown)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")

    if not args.output_json and not args.output_markdown:
        print(markdown)
    else:
        if args.output_json:
            print(f"JSON report: {args.output_json}")
        if args.output_markdown:
            print(f"Markdown report: {args.output_markdown}")

    if args.fail_on_failures and report.get("overall_status") == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
