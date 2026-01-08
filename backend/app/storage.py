from supabase import create_client

from .config import get_settings


def get_supabase_admin():
    settings = get_settings()
    supabase_url = settings.supabase_url
    if supabase_url and not supabase_url.endswith("/"):
        supabase_url = f"{supabase_url}/"
    return create_client(supabase_url, settings.supabase_service_role_key)


def upload_bytes(bucket: str, path: str, data: bytes, content_type: str | None = None) -> dict:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    options = {"upsert": "true"}
    if content_type:
        options["content-type"] = content_type
    return storage.upload(path, data, options)


def download_bytes(bucket: str, path: str) -> bytes:
    client = get_supabase_admin()
    storage = client.storage.from_(bucket)
    return storage.download(path)


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
    if bucket not in existing:
        client.storage.create_bucket(bucket, {"public": False})
