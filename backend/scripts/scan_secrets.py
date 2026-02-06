from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ALLOWLIST_PATH = Path(".github/secret-scan-allowlist.txt")


@dataclass(frozen=True)
class SecretPattern:
    code: str
    regex: re.Pattern[str]


PATTERNS = (
    SecretPattern("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    SecretPattern("GCP_API_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    SecretPattern("OPENAI_KEY", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    SecretPattern("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    SecretPattern("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----")),
)


def _git_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def _load_allowlist(path: Path = ALLOWLIST_PATH) -> list[str]:
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        patterns.append(token)
    return patterns


def _is_binary(path: Path) -> bool:
    try:
        data = path.read_bytes()[:1024]
    except Exception:
        return True
    return b"\x00" in data


def _is_allowed(path: Path, allowlist: list[str]) -> bool:
    posix = str(path.as_posix())
    return any(fnmatch.fnmatch(posix, pattern) for pattern in allowlist)


def scan_text_for_patterns(text: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for pattern in PATTERNS:
        for match in pattern.regex.finditer(text):
            findings.append((pattern.code, match.group(0)))
    return findings


def scan_repository(
    *,
    files: list[Path] | None = None,
    allowlist: list[str] | None = None,
) -> list[dict[str, object]]:
    files = files or _git_tracked_files()
    allowlist = allowlist or _load_allowlist()
    findings: list[dict[str, object]] = []

    for path in files:
        if _is_allowed(path, allowlist):
            continue
        if not path.exists() or path.is_dir() or _is_binary(path):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line_no, line in enumerate(lines, start=1):
            if "example" in str(path).lower() and "sk-" in line:
                # Allow placeholder examples in explicitly example files.
                continue
            for code, value in scan_text_for_patterns(line):
                findings.append(
                    {
                        "file": str(path),
                        "line": line_no,
                        "pattern": code,
                        "match": value[:16] + "...",
                    }
                )

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan tracked repository files for hardcoded secrets.")
    parser.add_argument(
        "--allowlist",
        default=str(ALLOWLIST_PATH),
        help="Optional allowlist file with glob patterns for files to ignore.",
    )
    args = parser.parse_args()

    findings = scan_repository(allowlist=_load_allowlist(Path(args.allowlist)))
    if not findings:
        print("PASS: no hardcoded secret patterns detected.")
        return

    print("FAIL: potential secrets detected:")
    for finding in findings:
        print(
            f"- {finding['file']}:{finding['line']} [{finding['pattern']}] {finding['match']}"
        )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
