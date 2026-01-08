from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path
from typing import Iterable

from backend.app.config import get_settings
from backend.app.db import get_conn
from backend.app.main import _upload_document, _extract_document, _embed_document, _draft_chr, _log_action
from backend.app.storage import ensure_bucket


PATIENTS = {
    "patient_a": {
        "full_name": "Alex Parker",
        "dob": "1982-05-14",
        "notes": "Metabolic risk and dyslipidemia workup.",
    },
    "patient_b": {
        "full_name": "Bianca Flores",
        "dob": "1990-11-02",
        "notes": "Thyroid and hormone evaluation.",
    },
    "patient_c": {
        "full_name": "Chris Nordin",
        "dob": "1978-03-09",
        "notes": "GI symptoms with inflammatory markers.",
    },
    "patient_d": {
        "full_name": "Dana Lee",
        "dob": "1986-12-22",
        "notes": "Genetics and medication response.",
    },
}


class SimpleUpload:
    def __init__(self, path: Path, content_type: str) -> None:
        self.filename = path.name
        self.content_type = content_type
        self.file = path.open("rb")

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


def iter_files(folder: Path) -> Iterable[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file()])


def guess_content_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    if path.suffix.lower() in {".txt", ".md"}:
        return "text/plain"
    return "application/octet-stream"


def create_patient(full_name: str, dob: str | None, notes: str | None) -> str:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO patients (full_name, dob, notes)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (full_name, dob, notes),
        ).fetchone()
        patient_id = str(row["id"])
        _log_action(conn, patient_id, "patient.create", "seed", {"name": full_name})
        conn.commit()
    return patient_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Import mock data into MedCHR.")
    parser.add_argument("--data-dir", default="data", help="Path to mock data root.")
    parser.add_argument("--patient", action="append", help="Patient folder name (repeatable).")
    parser.add_argument("--skip-extract", action="store_true", help="Skip text extraction.")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embeddings.")
    parser.add_argument("--skip-draft", action="store_true", help="Skip CHR draft generation.")
    args = parser.parse_args()

    settings = get_settings()
    ensure_bucket(settings.storage_bucket)

    do_extract = not args.skip_extract
    do_embed = not args.skip_embed
    do_draft = not args.skip_draft

    if do_embed and not settings.openai_api_key:
        print("OPENAI_API_KEY not set; skipping embeddings.")
        do_embed = False
    if do_draft and not settings.openai_api_key:
        print("OPENAI_API_KEY not set; skipping draft.")
        do_draft = False

    data_root = Path(args.data_dir).resolve()
    if not data_root.exists():
        raise SystemExit(f"Data dir not found: {data_root}")

    targets = args.patient or sorted(PATIENTS.keys())
    for key in targets:
        patient_folder = data_root / key
        if not patient_folder.exists():
            print(f"Skipping missing folder: {patient_folder}")
            continue

        meta = PATIENTS.get(key)
        if not meta:
            print(f"Skipping unknown patient: {key}")
            continue

        patient_id = create_patient(meta["full_name"], meta["dob"], meta["notes"])
        print(f"Created patient {meta['full_name']} ({patient_id})")

        for file_path in iter_files(patient_folder):
            upload = SimpleUpload(file_path, guess_content_type(file_path))
            try:
                doc = _upload_document(patient_id, upload, actor="seed")
            finally:
                upload.close()
            print(f"  Uploaded: {file_path.name}")

            if do_extract:
                _extract_document(doc.id, actor="seed")
                print(f"    Extracted: {file_path.name}")
            if do_embed:
                _embed_document(doc.id, actor="seed")
                print(f"    Embedded: {file_path.name}")

        if do_draft:
            _draft_chr(patient_id, None, actor="seed")
            print("  Draft generated.")

    print("Import complete.")


if __name__ == "__main__":
    main()
