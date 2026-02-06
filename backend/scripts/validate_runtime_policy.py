from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _add_check(
    checks: list[dict[str, Any]],
    *,
    check_id: str,
    status: str,
    title: str,
    evidence: str,
    remediation: str,
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": status,
            "title": title,
            "evidence": evidence,
            "remediation": remediation,
        }
    )


def _pinned_requirements(requirements_path: Path) -> tuple[bool, str]:
    if not requirements_path.exists():
        return False, f"{requirements_path} does not exist"
    unpinned: list[str] = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if "==" not in item and not item.startswith("-r "):
            unpinned.append(item)
    if unpinned:
        return False, f"Unpinned packages: {', '.join(unpinned[:5])}"
    return True, "All requirements are pinned with exact versions."


def validate_runtime_policy(
    *,
    dockerfile_path: Path = Path("Dockerfile"),
    requirements_path: Path = Path("backend/requirements.txt"),
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    if not dockerfile_path.exists():
        _add_check(
            checks,
            check_id="RUNTIME-001",
            status="FAIL",
            title="Dockerfile exists",
            evidence=f"Missing file: {dockerfile_path}",
            remediation="Create and maintain production Dockerfile.",
        )
        return {"overall_status": "FAIL", "checks": checks}

    content = dockerfile_path.read_text(encoding="utf-8")
    normalized = content.lower()

    has_non_root_user = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("USER "):
            continue
        user_value = stripped.split(None, 1)[1].strip().lower()
        if user_value and user_value not in {"root", "0", "0:0", "root:root"}:
            has_non_root_user = True

    _add_check(
        checks,
        check_id="RUNTIME-002",
        status="PASS" if has_non_root_user else "FAIL",
        title="Container runs as non-root user",
        evidence="USER directive uses non-root account." if has_non_root_user else "No non-root USER directive found.",
        remediation="Set explicit non-root USER in Dockerfile.",
    )

    has_healthcheck = "healthcheck" in normalized
    _add_check(
        checks,
        check_id="RUNTIME-003",
        status="PASS" if has_healthcheck else "FAIL",
        title="Container defines healthcheck",
        evidence="HEALTHCHECK directive detected." if has_healthcheck else "HEALTHCHECK directive missing.",
        remediation="Add HEALTHCHECK for liveness probe compatibility.",
    )

    copies_env = any(
        token in normalized
        for token in (
            "copy .env",
            "copy .env.example",
            "add .env",
            "add .env.example",
        )
    )
    _add_check(
        checks,
        check_id="RUNTIME-004",
        status="FAIL" if copies_env else "PASS",
        title="Image does not bake environment secret files",
        evidence="Potential .env copy/add instruction found." if copies_env else "No .env file copy detected.",
        remediation="Inject runtime configuration via secret manager or environment variables.",
    )

    has_python_hardening_env = "pythondontwritebytecode=1" in normalized and "pythonunbuffered=1" in normalized
    _add_check(
        checks,
        check_id="RUNTIME-005",
        status="PASS" if has_python_hardening_env else "WARN",
        title="Python runtime hardening env flags are set",
        evidence="PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED are configured."
        if has_python_hardening_env
        else "One or more Python runtime hardening env flags are missing.",
        remediation="Set PYTHONDONTWRITEBYTECODE=1 and PYTHONUNBUFFERED=1 in Dockerfile.",
    )

    requirements_ok, requirements_evidence = _pinned_requirements(requirements_path)
    _add_check(
        checks,
        check_id="RUNTIME-006",
        status="PASS" if requirements_ok else "FAIL",
        title="Dependency lockfile is pinned",
        evidence=requirements_evidence,
        remediation="Pin dependencies (exact versions) before release.",
    )

    overall = "FAIL" if any(check["status"] == "FAIL" for check in checks) else "PASS"
    return {"overall_status": overall, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate provider-neutral runtime policy controls.")
    parser.add_argument("--dockerfile", default="Dockerfile")
    parser.add_argument("--requirements", default="backend/requirements.txt")
    parser.add_argument("--output-json", help="Optional JSON output path")
    args = parser.parse_args()

    report = validate_runtime_policy(
        dockerfile_path=Path(args.dockerfile),
        requirements_path=Path(args.requirements),
    )
    text = json.dumps(report, indent=2)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Runtime policy report written to {output_path}")
    else:
        print(text)

    if report["overall_status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
