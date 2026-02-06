from pathlib import Path

from backend.scripts.validate_llm_gateway_usage import find_llm_gateway_violations


def test_project_has_no_direct_openai_imports_outside_gateway():
    violations = find_llm_gateway_violations(scan_root=Path("backend/app"))
    assert violations == []


def test_detects_direct_openai_import_violation(tmp_path):
    module = tmp_path / "bad_module.py"
    module.write_text("from openai import OpenAI\n", encoding="utf-8")
    violations = find_llm_gateway_violations(scan_root=tmp_path)
    assert len(violations) == 1
    assert violations[0]["file"].endswith("bad_module.py")
