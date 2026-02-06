from pathlib import Path

from backend.scripts.scan_secrets import scan_repository, scan_text_for_patterns


def _fake_openai_key() -> str:
    return "sk-" + ("A" * 40)


def test_scan_text_for_patterns_detects_openai_key():
    findings = scan_text_for_patterns(f"OPENAI_API_KEY={_fake_openai_key()}")
    assert any(code == "OPENAI_KEY" for code, _ in findings)


def test_scan_repository_reports_finding_for_non_allowlisted_file(tmp_path):
    file_path = tmp_path / "config.txt"
    file_path.write_text(f"token={_fake_openai_key()}\n", encoding="utf-8")
    findings = scan_repository(files=[file_path], allowlist=[])
    assert len(findings) == 1
    assert findings[0]["file"].endswith("config.txt")


def test_scan_repository_respects_allowlist(tmp_path):
    file_path = tmp_path / "config.txt"
    file_path.write_text(f"token={_fake_openai_key()}\n", encoding="utf-8")
    pattern = f"*{Path(file_path).name}"
    findings = scan_repository(files=[file_path], allowlist=[pattern])
    assert findings == []
