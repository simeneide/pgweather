"""Centralised configuration loaded from environment variables via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — validated once at import time."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    supabase_db_url: str
    """Postgres connection string (required)."""

    # Forecast cache TTL
    data_ttl_seconds: int = 600
    """How long (seconds) to cache forecast queries before re-fetching."""

    # Yr / MET Norway cache TTL
    yr_ttl_seconds: int = 1800
    """How long (seconds) to cache Yr locationforecast responses."""


settings = Settings()  # type: ignore[call-arg]
