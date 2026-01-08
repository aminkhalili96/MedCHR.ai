from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    storage_bucket: str = "medchr-uploads"

    openai_api_key: str
    openai_model: str = "gpt-5.2"
    openai_embedding_model: str = "text-embedding-3-large"

    app_secret_key: str = "dev-secret"
    app_username: str = "admin"
    app_password: str = "admin"

    app_env: str = "dev"
    log_level: str = "info"

    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
