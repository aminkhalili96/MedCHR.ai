from pathlib import Path

import httpx
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_supabase_admin():
    settings = get_settings()
    supabase_url = settings.supabase_url
    if supabase_url and not supabase_url.endswith("/"):
        supabase_url = f"{supabase_url}/"
    return create_client(supabase_url, settings.supabase_service_role_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def upload_bytes(bucket: str, path: str, data: bytes, content_type: str | None = None) -> dict:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    options = {"upsert": "true"}
    if content_type:
        options["content-type"] = content_type
    return storage.upload(path, data, options)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def upload_bytes_via_signed_url(bucket: str, path: str, data: bytes, content_type: str | None = None) -> dict:
    signed = create_signed_upload_url(bucket, path)
    filename = Path(signed["path"]).name
    mime = content_type or "application/octet-stream"
    with httpx.Client(timeout=30.0) as client:
        response = client.put(
            signed["upload_url"],
            files={"file": (filename, data, mime)},
        )
    response.raise_for_status()
    payload = {}
    try:
        payload = response.json()
    except Exception:
        payload = {"status_code": response.status_code}
    return {"path": signed["path"], **payload}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def create_signed_upload_url(bucket: str, path: str) -> dict:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    result = storage.create_signed_upload_url(path)
    upload_url = getattr(result, "signed_url", None) or getattr(result, "signedUrl", None) or ""
    token = getattr(result, "token", None) or ""
    resolved_path = getattr(result, "path", None) or path
    if not upload_url or not token:
        raise RuntimeError("Failed to create signed upload URL")
    if upload_url.startswith("/") and not upload_url.startswith("//"):
        settings = get_settings()
        upload_url = f"{settings.supabase_url.rstrip('/')}{upload_url}"
    return {"upload_url": upload_url, "token": token, "path": resolved_path}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def create_signed_download_url(bucket: str, path: str, expires_in_seconds: int = 300) -> str:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    result = storage.create_signed_url(path, expires_in_seconds)
    signed = getattr(result, "signedURL", None) or getattr(result, "signedUrl", None)
    if not signed:
        raise RuntimeError("Failed to create signed download URL")
    if signed.startswith("http://") or signed.startswith("https://"):
        return signed
    settings = get_settings()
    base = settings.supabase_url.rstrip("/")
    if signed.startswith("/"):
        return f"{base}{signed}"
    return f"{base}/{signed}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def download_bytes(bucket: str, path: str) -> bytes:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    return storage.download(path)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def delete_bytes(bucket: str, paths: list[str]) -> None:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    storage.remove(paths)


def ensure_bucket(bucket: str) -> None:
    client = get_supabase_admin()
    buckets = client.storage.list_buckets()
    existing = set()
    if buckets:
        for b in buckets:
            if isinstance(b, dict):
                name = b.get("name")
            else:
                name = getattr(b, "name", None)
            if name:
                existing.add(name)
    settings = get_settings()
    options = {"public": False}
    allowed = _split_csv(settings.allowed_mime_types)
    if allowed:
        options["allowed_mime_types"] = allowed
    if settings.max_upload_mb:
        options["file_size_limit"] = settings.max_upload_mb * 1024 * 1024
    if bucket not in existing:
        client.storage.create_bucket(bucket, options)
    else:
        client.storage.update_bucket(bucket, options)


def storage_health(bucket: str) -> bool:
    client = get_supabase_admin()
    buckets = client.storage.list_buckets()
    return any(getattr(b, "name", None) == bucket or (isinstance(b, dict) and b.get("name") == bucket) for b in buckets or [])
