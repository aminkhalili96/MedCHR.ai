from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    supabase_url: str
    supabase_anon_key: str | None = None
    supabase_service_role_key: str
    storage_bucket: str = "medchr-uploads"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5.2"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_timeout_seconds: int = 30

    app_secret_key: str = "dev-secret"
    app_username: str = "admin"
    app_password: str | None = None
    app_password_hash: str | None = None
    api_keys: str | None = None

    app_env: str = "dev"
    log_level: str = "info"
    app_base_url: str = "http://127.0.0.1:8000"
    api_docs_enabled: bool = True

    # Security + compliance
    hipaa_mode: bool = False
    phi_processors: str = ""
    phi_redaction_enabled: bool = False
    mfa_secret_key: str | None = None
    csrf_enabled: bool = True
    allowed_hosts: str = "localhost,127.0.0.1,0.0.0.0"
    cors_origins: str = ""
    referrer_policy: str = "no-referrer"
    trust_proxy_headers: bool = False
    trusted_proxy_ips: str = ""

    # Upload constraints
    max_upload_mb: int = 25
    allowed_mime_types: str = "application/pdf,image/png,image/jpeg,image/tiff,text/plain"
    malware_scan_enabled: bool = False
    malware_scan_timeout_seconds: int = 30

    # DB settings
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_statement_timeout_ms: int = 15000

    # Embedding chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200
    aggregate_notes_max_chars: int = 8000

    # Rate limiting
    rate_limit_default: str = "60/minute"

    # Session settings
    session_cookie_name: str = "medchr_session"
    session_max_age_minutes: int = 720
    step_up_window_minutes: int = 15

    # Enterprise SSO
    oidc_enabled: bool = False
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_discovery_url: str | None = None
    azure_ad_enabled: bool = False
    azure_client_id: str | None = None
    azure_client_secret: str | None = None
    azure_tenant_id: str | None = None
    google_workspace_enabled: bool = False
    google_client_id: str | None = None
    google_client_secret: str | None = None
    saml_idp_entity_id: str | None = None
    saml_idp_sso_url: str | None = None
    saml_idp_cert: str | None = None
    sso_allowed_domains: str = ""
    sso_allowed_providers: str = ""
    sso_require_verified_email: bool = True

    # Background jobs
    job_queue_enabled: bool = False
    job_poll_interval_seconds: int = 5
    job_max_attempts: int = 3
    redis_url: str | None = None
    redis_queue_name: str = "medchr:jobs"

    # Data retention
    audit_retention_days: int = 365
    job_retention_days: int = 30
    retention_export_dir: str = "data/retention_exports"
    retention_immutable_dir: str = ""

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
