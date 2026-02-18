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

    # Database — accepts SUPABASE_DB_URL (local/.envrc, GitHub Actions)
    # or DATABASE_URL (Fly.io secret)
    supabase_db_url: str = ""  # type: ignore[assignment]
    """Postgres connection string (required). Set via SUPABASE_DB_URL or DATABASE_URL."""
    database_url: str = ""
    """Alternative env var name for the Postgres connection string (Fly.io)."""

    @property
    def db_url(self) -> str:
        """Return the database URL, preferring supabase_db_url over database_url."""
        url = self.supabase_db_url or self.database_url
        if not url:
            raise RuntimeError(
                "No database URL configured. Set SUPABASE_DB_URL or DATABASE_URL."
            )
        return url

    # Forecast cache TTL
    data_ttl_seconds: int = 600
    """How long (seconds) to cache forecast queries before re-fetching."""

    # Yr / MET Norway cache TTL
    yr_ttl_seconds: int = 3600
    """How long (seconds) to cache Yr locationforecast responses.

    Yr data updates hourly, so 3600s avoids redundant fetches."""

    # Multi-model support
    default_model_source: str = "meps"
    """Default weather model source used when none is specified."""


settings = Settings()  # type: ignore[call-arg]
