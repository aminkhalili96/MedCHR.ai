from pathlib import Path

from backend.scripts.validate_runtime_policy import validate_runtime_policy


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_validate_runtime_policy_passes_for_hardened_inputs(tmp_path):
    dockerfile = _write(
        tmp_path / "Dockerfile",
        """
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
HEALTHCHECK CMD echo ok
USER appuser
""".strip(),
    )
    requirements = _write(tmp_path / "requirements.txt", "fastapi==0.115.0\npydantic==2.9.2\n")

    report = validate_runtime_policy(dockerfile_path=dockerfile, requirements_path=requirements)
    assert report["overall_status"] == "PASS"


def test_validate_runtime_policy_fails_for_root_user_and_unpinned_requirements(tmp_path):
    dockerfile = _write(
        tmp_path / "Dockerfile",
        """
FROM python:3.11-slim
USER root
""".strip(),
    )
    requirements = _write(tmp_path / "requirements.txt", "fastapi\n")

    report = validate_runtime_policy(dockerfile_path=dockerfile, requirements_path=requirements)
    assert report["overall_status"] == "FAIL"
