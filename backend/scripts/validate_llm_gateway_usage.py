from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_SCAN_ROOT = Path("backend/app")
ALLOWED_OPENAI_FILES = {
    Path("backend/app/llm_gateway.py").resolve(),
}


def _openai_import_violations(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "openai" or alias.name.startswith("openai."):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "openai" or (node.module and node.module.startswith("openai.")):
                names = ", ".join(alias.name for alias in node.names)
                violations.append(f"line {node.lineno}: from {node.module} import {names}")
    return violations


def find_llm_gateway_violations(scan_root: Path = DEFAULT_SCAN_ROOT) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for path in sorted(scan_root.rglob("*.py")):
        resolved = path.resolve()
        file_violations = _openai_import_violations(path)
        if not file_violations:
            continue
        if resolved in ALLOWED_OPENAI_FILES:
            continue
        violations.append(
            {
                "file": str(path),
                "violations": file_violations,
            }
        )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure all outbound OpenAI usage is centralized in backend/app/llm_gateway.py."
    )
    parser.add_argument(
        "--scan-root",
        default=str(DEFAULT_SCAN_ROOT),
        help="Directory to scan for direct OpenAI imports.",
    )
    args = parser.parse_args()

    scan_root = Path(args.scan_root)
    violations = find_llm_gateway_violations(scan_root=scan_root)
    if not violations:
        print("PASS: OpenAI imports are centralized in llm_gateway.")
        return

    print("FAIL: Direct OpenAI imports found outside llm_gateway:")
    for item in violations:
        print(f"- {item['file']}")
        for violation in item["violations"]:
            print(f"  - {violation}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
