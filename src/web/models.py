"""Typed models for API responses, cache entries, and shared data structures."""

from __future__ import annotations

import datetime as dt
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


class TakeoffOption(BaseModel):
    """Dropdown option for a takeoff location."""

    label: str
    value: str


class TakeoffInfo(BaseModel):
    """Static metadata for a single takeoff / area location."""

    name: str
    latitude: float
    longitude: float
    point_type: str


# ---------------------------------------------------------------------------
# Forecast metadata (replaces the need to load full DataFrame for layout)
# ---------------------------------------------------------------------------


class ForecastMeta(BaseModel):
    """Lightweight metadata derived from the latest forecast — used for
    populating dropdowns, time sliders, and day selectors without loading
    the full forecast DataFrame."""

    latest_forecast_timestamp: dt.datetime
    available_times: list[dt.datetime]
    takeoff_options: list[TakeoffOption]
    takeoffs: list[TakeoffInfo]


# ---------------------------------------------------------------------------
# Yr weather entry
# ---------------------------------------------------------------------------


class YrEntry(BaseModel):
    """One hourly entry from the MET Norway locationforecast API."""

    time: dt.datetime
    symbol_code: str
    air_temperature: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    cloud_area_fraction: Optional[float] = None


class YrDisplayEntry(YrEntry):
    """Yr entry enriched with display-ready fields."""

    local_hour: int
    icon_url: str
    icon_png_url: str


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------


class CacheEntry(BaseModel, Generic[T]):
    """Generic TTL-aware cache wrapper."""

    model_config = {"arbitrary_types_allowed": True}

    loaded_at: dt.datetime
    data: T

    def is_fresh(self, ttl_seconds: int) -> bool:
        age = (dt.datetime.now(dt.timezone.utc) - self.loaded_at).total_seconds()
        return age < ttl_seconds


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    rows: Optional[int] = None
    latest_forecast_timestamp: Optional[str] = None
    forecast_age_hours: Optional[float] = None
    detail: Optional[str] = None


class MetaResponse(BaseModel):
    latest_forecast_timestamp: str
    available_times: list[str]
    takeoff_count: int
