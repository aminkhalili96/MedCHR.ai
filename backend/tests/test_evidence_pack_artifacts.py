import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.scripts.generate_evidence_pack import _prune_old_packs, _write_manifest


def test_write_manifest_generates_hmac_signature(tmp_path):
    report = tmp_path / "evidence_report.json"
    markdown = tmp_path / "evidence_report.md"
    report.write_text('{"ok": true}', encoding="utf-8")
    markdown.write_text("# Evidence", encoding="utf-8")

    with patch(
        "backend.scripts.generate_evidence_pack.get_settings",
        return_value=SimpleNamespace(app_secret_key="test-secret"),
    ):
        manifest_path = _write_manifest(tmp_path, [report, markdown])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["signature"]["algorithm"] == "HMAC-SHA256"
    assert payload["signature"]["value"]
    assert len(payload["payload"]["files"]) == 2


def test_prune_old_packs_keeps_latest_directories(tmp_path):
    for name in ("20260101T000000Z", "20260201T000000Z", "20260301T000000Z"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)

    _prune_old_packs(tmp_path, keep_latest=2)

    remaining = sorted(path.name for path in tmp_path.iterdir() if path.is_dir())
    assert remaining == ["20260201T000000Z", "20260301T000000Z"]
