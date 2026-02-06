import mimetypes
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import HTTPException

from .config import get_settings


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def allowed_mime_types() -> set[str]:
    settings = get_settings()
    return _split_csv(settings.allowed_mime_types)


def sanitize_filename(filename: str) -> str:
    basename = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return cleaned or "upload.bin"


def resolve_content_type(filename: str, content_type: str | None) -> str:
    if content_type and content_type != "application/octet-stream":
        return content_type
    guess, _ = mimetypes.guess_type(filename)
    return guess or "application/octet-stream"


def read_upload_bytes(file) -> tuple[bytes, str, int]:
    settings = get_settings()
    filename = sanitize_filename(file.filename or "upload.bin")
    content_type = resolve_content_type(filename, getattr(file, "content_type", None))

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if hasattr(file, "file") and file.file:
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()
        file.file.seek(0)
    else:
        data = file.read()
        size = len(data)
        file = type("SimpleFile", (), {"file": None, "content_type": content_type, "filename": filename, "read": lambda: data})()

    if size <= 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.max_upload_mb}MB.",
        )

    if hasattr(file, "file") and file.file:
        data = file.file.read()
    else:
        data = file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    allowed = allowed_mime_types()
    if allowed and content_type.lower() not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

    _scan_bytes(data)

    return data, content_type, size


def _scan_bytes(data: bytes) -> None:
    settings = get_settings()
    if not settings.malware_scan_enabled:
        return
    if not shutil.which("clamdscan"):
        raise HTTPException(status_code=500, detail="Malware scanning enabled but clamdscan not available")
    with NamedTemporaryFile(delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        result = subprocess.run(
            ["clamdscan", "--no-summary", tmp.name],
            capture_output=True,
            text=True,
            timeout=settings.malware_scan_timeout_seconds,
        )
    if result.returncode == 1:
        raise HTTPException(status_code=400, detail="Malware detected in upload")
    if result.returncode not in (0, 1):
        raise HTTPException(status_code=502, detail="Malware scan failed")
