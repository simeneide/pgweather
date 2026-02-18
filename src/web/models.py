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
# Wind sector suitability
# ---------------------------------------------------------------------------

COMPASS_SECTORS: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Degree midpoints for each sector (N=0, NE=45, …)
_SECTOR_MIDPOINTS: dict[str, float] = {
    s: i * 45.0 for i, s in enumerate(COMPASS_SECTORS)
}


class WindSectorData(BaseModel):
    """Wind-sector suitability ratings for a takeoff from ParaglidingEarth.

    Each sector value: 2 = suitable, 1 = marginal/crosswind, 0 = not suitable.
    """

    sectors: dict[str, int]  # e.g. {"N": 0, "NE": 2, "E": 1, …}

    @property
    def has_data(self) -> bool:
        """True if at least one sector is rated (non-zero)."""
        return any(v > 0 for v in self.sectors.values())

    def facing_label(self) -> str:
        """Human-readable label of suitable wind directions, e.g. 'S-W'."""
        suitable = [s for s in COMPASS_SECTORS if self.sectors.get(s, 0) >= 2]
        if not suitable:
            marginal = [s for s in COMPASS_SECTORS if self.sectors.get(s, 0) >= 1]
            if marginal:
                return "-".join(marginal)
            return "?"
        return "-".join(suitable)


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


class ModelSourceOption(BaseModel):
    label: str
    value: str


class FrontendDay(BaseModel):
    key: str
    label: str
    times: list[str]


class FrontendMetaResponse(BaseModel):
    latest_forecast_timestamp: str
    selected_model_source: str
    default_model_source: str
    model_source_options: list[ModelSourceOption]
    selected_day: str
    selected_time: str
    days: list[FrontendDay]
    location_options: list[TakeoffOption]


class MapCenter(BaseModel):
    lat: float
    lon: float
    zoom: int


class MapFeaturePoint(BaseModel):
    name: str
    latitude: float
    longitude: float
    thermal_top: float
    peak_thermal_velocity: float
    selected: bool = False
    suitability_color: str = "#9e9e9e"
    suitability_label: str = "No data"
    suitability_tooltip: str = ""
    wind_speed: float = 0.0
    wind_direction_compass: str = ""


class MapAreaFeature(BaseModel):
    type: str
    geometry: dict
    properties: dict[str, object]


class WindVector(BaseModel):
    latitude: float
    longitude: float
    tip_latitude: float
    tip_longitude: float
    wind_speed: float
    direction_degrees: float
    direction_compass: str


class MapPayloadResponse(BaseModel):
    selected_time: str
    selected_time_local_label: str
    selected_name: Optional[str] = None
    center: MapCenter
    points: list[MapFeaturePoint]
    area_features: list[MapAreaFeature]
    wind_altitude: Optional[float] = None
    wind_vectors: list[WindVector]


class AirgramWindSample(BaseModel):
    time_label: str
    altitude: float
    wind_speed: float
    wind_direction: float
    thermal_velocity: float


class AirgramThermalTop(BaseModel):
    time_label: str
    thermal_top: float


class AirgramYrPoint(BaseModel):
    time_label: str
    icon_png_url: str
    symbol_code: str
    air_temperature: Optional[float] = None
    precipitation: Optional[float] = None


class AirgramPayloadResponse(BaseModel):
    location: str
    date: str
    timezone: str
    elevation: float
    altitude_max: int
    selected_hour: Optional[int] = None
    snow_depth_cm: Optional[float] = None
    time_labels: list[str]
    altitudes: list[float]
    thermal_matrix: list[list[float]]
    thermal_tops: list[AirgramThermalTop]
    wind_samples: list[AirgramWindSample]
    yr: list[AirgramYrPoint]


class SummaryResponse(BaseModel):
    summary: str
    forecast_used_timestamp: str
    forecast_age_hours: float


class FrontendMetaRequest(BaseModel):
    model_source: Optional[str] = None
    forecast_timestamp: Optional[dt.datetime] = None


class MapPayloadRequest(BaseModel):
    selected_time: dt.datetime
    selected_name: Optional[str] = None
    zoom: int = 6
    forecast_timestamp: Optional[dt.datetime] = None
    wind_altitude: Optional[float] = None
    model_source: Optional[str] = None


class AirgramPayloadRequest(BaseModel):
    location: str
    selected_date: dt.date
    altitude_max: int = 3000
    selected_hour: Optional[int] = None
    forecast_timestamp: Optional[dt.datetime] = None
    model_source: Optional[str] = None


class SummaryRequest(BaseModel):
    selected_name: str
    selected_time: dt.datetime
    forecast_timestamp: Optional[dt.datetime] = None
    model_source: Optional[str] = None
