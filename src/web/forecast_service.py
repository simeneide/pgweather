"""Forecast data loading, caching, and figure building.

Data loading uses per-view queries instead of a monolithic bulk load:
- ``load_metadata()``   — lightweight query for layout init (dropdowns, times)
- ``load_map_data()``   — single time-step, all locations (map view)
- ``load_windgram_data()`` — single location + day, all altitudes (airgram)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go
import polars as pl
import requests

import db_utils

from .config import settings
from .models import (
    CacheEntry,
    ForecastMeta,
    TakeoffInfo,
    TakeoffOption,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared database helper
# ---------------------------------------------------------------------------
_db = db_utils.Database()

# ---------------------------------------------------------------------------
# Yr / MET Norway locationforecast
# ---------------------------------------------------------------------------
_YR_CACHE: dict[str, CacheEntry[list[dict]]] = {}
_YR_ICON_BASE = "https://raw.githubusercontent.com/metno/weathericons/main/weather/svg"
_YR_ICON_PNG_BASE = (
    "https://raw.githubusercontent.com/metno/weathericons/main/weather/png"
)
_YR_USER_AGENT = "pgweather/1.0 github.com/simeneide/pgweather"

# ---------------------------------------------------------------------------
# Cached latest forecast timestamp (short TTL, shared across queries)
# ---------------------------------------------------------------------------
_TS_CACHE: CacheEntry[dt.datetime] | None = None


def _get_latest_forecast_timestamp() -> dt.datetime:
    """Return the latest forecast_timestamp, cached with short TTL."""
    global _TS_CACHE
    if _TS_CACHE is not None and _TS_CACHE.is_fresh(settings.data_ttl_seconds):
        return _TS_CACHE.data

    query = "SELECT max(forecast_timestamp) AS ts FROM detailed_forecasts"
    row = _db.read(query)
    ts = row[0, "ts"]
    if ts is None:
        raise RuntimeError("No forecasts in detailed_forecasts table")

    # Ensure timezone-aware
    if isinstance(ts, dt.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        raise RuntimeError(f"Unexpected type for forecast_timestamp: {type(ts)}")

    _TS_CACHE = CacheEntry(loaded_at=dt.datetime.now(dt.timezone.utc), data=ts)
    return ts


def _ensure_tz_utc(ts: dt.datetime) -> dt.datetime:
    """Ensure a datetime is timezone-aware (UTC)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts


# ---------------------------------------------------------------------------
# Available forecast generations (for debug mode)
# ---------------------------------------------------------------------------
_FORECAST_TS_CACHE: CacheEntry[list[dt.datetime]] | None = None


def get_available_forecast_timestamps() -> list[dt.datetime]:
    """Return all distinct forecast_timestamps in the DB, newest first."""
    global _FORECAST_TS_CACHE
    if _FORECAST_TS_CACHE is not None and _FORECAST_TS_CACHE.is_fresh(
        settings.data_ttl_seconds
    ):
        return _FORECAST_TS_CACHE.data

    query = """
    SELECT DISTINCT forecast_timestamp
    FROM detailed_forecasts
    ORDER BY forecast_timestamp DESC
    """
    df = _db.read(query)
    timestamps = [
        _ensure_tz_utc(t) for t in df.get_column("forecast_timestamp").to_list()
    ]

    _FORECAST_TS_CACHE = CacheEntry(
        loaded_at=dt.datetime.now(dt.timezone.utc), data=timestamps
    )
    return timestamps


def _resolve_forecast_ts(forecast_ts: Optional[dt.datetime] = None) -> dt.datetime:
    """Return *forecast_ts* if given, otherwise the latest."""
    if forecast_ts is not None:
        return _ensure_tz_utc(forecast_ts)
    return _get_latest_forecast_timestamp()


# ---------------------------------------------------------------------------
# Metadata query (layout init — dropdowns, time slider, day selector)
# ---------------------------------------------------------------------------
_META_CACHE: dict[str, CacheEntry[ForecastMeta]] = {}


def load_metadata(
    forecast_ts: Optional[dt.datetime] = None,
) -> ForecastMeta:
    """Lightweight metadata for populating the UI — no full DataFrame load."""
    resolved_ts = _resolve_forecast_ts(forecast_ts)
    cache_key = resolved_ts.isoformat()

    cached = _META_CACHE.get(cache_key)
    if cached is not None and cached.is_fresh(settings.data_ttl_seconds):
        return cached.data

    latest_ts = resolved_ts

    # Available times
    times_query = f"""
    SELECT DISTINCT time
    FROM detailed_forecasts
    WHERE forecast_timestamp = '{latest_ts.isoformat()}'
    ORDER BY time
    """
    times_df = _db.read(times_query)
    available_times = [
        t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t
        for t in times_df.get_column("time").to_list()
    ]

    # Takeoff / area info — one row per name (areas may have multiple grid
    # points; DISTINCT ON picks one representative lat/lon per name).
    ref_time = max(available_times)
    info_query = f"""
    SELECT DISTINCT ON (name) name, latitude, longitude, point_type
    FROM detailed_forecasts
    WHERE forecast_timestamp = '{latest_ts.isoformat()}'
      AND time = '{ref_time.isoformat()}'
    ORDER BY name
    """
    info_df = _db.read(info_query)

    takeoffs_raw = [
        TakeoffInfo(**row)
        for row in info_df.filter(pl.col("point_type") != "area").iter_rows(named=True)
    ]
    areas_raw = [
        TakeoffInfo(**row)
        for row in info_df.filter(pl.col("point_type") == "area").iter_rows(named=True)
    ]

    # Build dropdown options with region labels
    takeoff_options = _build_takeoff_options(takeoffs_raw, areas_raw)

    meta = ForecastMeta(
        latest_forecast_timestamp=latest_ts,
        available_times=available_times,
        takeoff_options=takeoff_options,
        takeoffs=takeoffs_raw + areas_raw,
    )
    _META_CACHE[cache_key] = CacheEntry(
        loaded_at=dt.datetime.now(dt.timezone.utc), data=meta
    )
    return meta


def _build_takeoff_options(
    takeoffs: list[TakeoffInfo],
    areas: list[TakeoffInfo],
) -> list[TakeoffOption]:
    """Build dropdown options, matching each takeoff to its nearest area label."""
    if not takeoffs:
        return []

    sorted_takeoffs = sorted(takeoffs, key=lambda t: t.name)

    if not areas:
        return [TakeoffOption(label=t.name, value=t.name) for t in sorted_takeoffs]

    area_names = np.array([a.name for a in areas])
    area_lat = np.array([a.latitude for a in areas])
    area_lon = np.array([a.longitude for a in areas])
    max_dist2 = 0.25  # ~50 km

    options: list[TakeoffOption] = []
    for t in sorted_takeoffs:
        if "(" in t.name:
            options.append(TakeoffOption(label=t.name, value=t.name))
            continue
        dist2 = (area_lat - t.latitude) ** 2 + (area_lon - t.longitude) ** 2
        min_dist2 = float(np.min(dist2))
        if min_dist2 > max_dist2:
            options.append(TakeoffOption(label=t.name, value=t.name))
        else:
            idx = int(np.argmin(dist2))
            region = str(area_names[idx])
            options.append(TakeoffOption(label=f"{t.name} ({region})", value=t.name))
    return options


# ---------------------------------------------------------------------------
# Convenience accessors (used by dash_ui and main)
# ---------------------------------------------------------------------------


def get_latest_forecast_timestamp() -> dt.datetime:
    return _get_latest_forecast_timestamp()


def get_available_times() -> list[dt.datetime]:
    return load_metadata().available_times


def get_takeoff_options() -> list[dict[str, str]]:
    """Return takeoff options as plain dicts for Dash dropdown compatibility."""
    return [opt.model_dump() for opt in load_metadata().takeoff_options]


def get_takeoff_names() -> list[str]:
    return sorted(t.name for t in load_metadata().takeoffs if t.point_type != "area")


def get_default_selected_time() -> dt.datetime:
    times = get_available_times()
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    return min(times, key=lambda t: abs((t - now).total_seconds()))


# ---------------------------------------------------------------------------
# Gridded wind query — single time step + altitude, full spatial grid
# ---------------------------------------------------------------------------
_GRID_WIND_CACHE: dict[str, CacheEntry[pl.DataFrame]] = {}

# Available altitude levels in the gridded_forecasts table (metres AGL).
# Must match GRID_ALTITUDES in preprocess_forecast.py.
GRID_ALTITUDES = [0, 500, 1000, 1500, 2000, 3000]


def load_grid_wind_data(
    selected_time: dt.datetime,
    altitude: float = 1000,
    forecast_ts: Optional[dt.datetime] = None,
) -> pl.DataFrame:
    """Load gridded wind vectors for a specific time and altitude level.

    Returns a DataFrame with one row per grid point, containing lat/lon and
    wind components for rendering wind arrows on the map.
    """
    latest_ts = _resolve_forecast_ts(forecast_ts)
    cache_key = f"{latest_ts.isoformat()}|{selected_time.isoformat()}|{altitude}"

    cached = _GRID_WIND_CACHE.get(cache_key)
    if cached is not None and cached.is_fresh(settings.data_ttl_seconds):
        return cached.data

    query = f"""
    SELECT latitude, longitude, x_wind_ml, y_wind_ml, wind_speed,
           thermal_velocity, thermal_top, elevation
    FROM gridded_forecasts
    WHERE forecast_timestamp = '{latest_ts.isoformat()}'
      AND time = '{selected_time.isoformat()}'
      AND altitude = {altitude}
    """
    try:
        df = _db.read(query)
    except Exception:
        logger.warning("gridded_forecasts table not available yet")
        df = pl.DataFrame()

    # Evict stale entries
    ts_prefix = latest_ts.isoformat()
    stale_keys = [k for k in _GRID_WIND_CACHE if not k.startswith(ts_prefix)]
    for k in stale_keys:
        del _GRID_WIND_CACHE[k]

    _GRID_WIND_CACHE[cache_key] = CacheEntry(
        loaded_at=dt.datetime.now(dt.timezone.utc), data=df
    )
    return df


# ---------------------------------------------------------------------------
# Map view query — single time step, all locations
# ---------------------------------------------------------------------------
_MAP_CACHE: dict[str, CacheEntry[pl.DataFrame]] = {}


def load_map_data(
    selected_time: dt.datetime,
    forecast_ts: Optional[dt.datetime] = None,
) -> pl.DataFrame:
    """Load data for map view: one row per (name, point_type) at *selected_time*.

    Returns a small DataFrame (~520 rows) with columns needed for the map.
    """
    latest_ts = _resolve_forecast_ts(forecast_ts)
    cache_key = f"{latest_ts.isoformat()}|{selected_time.isoformat()}"

    cached = _MAP_CACHE.get(cache_key)
    if cached is not None and cached.is_fresh(settings.data_ttl_seconds):
        return cached.data

    # One row per name with surface wind (lowest altitude) and peak thermal
    # velocity (max across all altitudes).  Uses a window function to compute
    # the max before DISTINCT ON picks the lowest-altitude row.
    query = f"""
    SELECT DISTINCT ON (name)
           name, latitude, longitude, point_type,
           thermal_top, wind_speed,
           MAX(thermal_velocity) OVER (PARTITION BY name) AS peak_thermal_velocity
    FROM detailed_forecasts
    WHERE forecast_timestamp = '{latest_ts.isoformat()}'
      AND time = '{selected_time.isoformat()}'
    ORDER BY name, altitude
    """
    df = _db.read(query)

    # Evict old entries (keep only current forecast_timestamp)
    ts_prefix = latest_ts.isoformat()
    stale_keys = [k for k in _MAP_CACHE if not k.startswith(ts_prefix)]
    for k in stale_keys:
        del _MAP_CACHE[k]

    _MAP_CACHE[cache_key] = CacheEntry(
        loaded_at=dt.datetime.now(dt.timezone.utc), data=df
    )
    return df


# ---------------------------------------------------------------------------
# Windgram query — single location + day, all altitudes
# ---------------------------------------------------------------------------
_WINDGRAM_CACHE: dict[str, CacheEntry[pl.DataFrame]] = {}


def load_windgram_data(
    name: str,
    date: dt.date,
    forecast_ts: Optional[dt.datetime] = None,
) -> pl.DataFrame:
    """Load all altitude data for a single location on a given day.

    Returns a DataFrame with ~294 rows (14 hours × 21 altitudes).
    """
    latest_ts = _resolve_forecast_ts(forecast_ts)
    cache_key = f"{latest_ts.isoformat()}|{name}|{date.isoformat()}"

    cached = _WINDGRAM_CACHE.get(cache_key)
    if cached is not None and cached.is_fresh(settings.data_ttl_seconds):
        return cached.data

    # Date filtering: convert time to Europe/Oslo and filter by date
    # Since we can't do TZ conversion in the SQL query on all Postgres setups,
    # we compute the UTC range for the local date.
    local_tz = ZoneInfo("Europe/Oslo")
    day_start_local = dt.datetime.combine(date, dt.time.min, tzinfo=local_tz)
    day_end_local = dt.datetime.combine(
        date + dt.timedelta(days=1), dt.time.min, tzinfo=local_tz
    )
    day_start_utc = day_start_local.astimezone(dt.timezone.utc)
    day_end_utc = day_end_local.astimezone(dt.timezone.utc)

    query = f"""
    SELECT time, altitude, elevation,
           air_temperature_ml, x_wind_ml, y_wind_ml,
           wind_speed, thermal_velocity, thermal_top,
           thermal_height_above_ground, snow_depth
    FROM detailed_forecasts
    WHERE forecast_timestamp = '{latest_ts.isoformat()}'
      AND name = '{name}'
      AND point_type = 'takeoff'
      AND time >= '{day_start_utc.isoformat()}'
      AND time < '{day_end_utc.isoformat()}'
    ORDER BY time, altitude
    """
    df = _db.read(query).with_columns(
        pl.col("time").cast(pl.Datetime).dt.replace_time_zone("UTC"),
    )

    # Evict stale entries
    ts_prefix = latest_ts.isoformat()
    stale_keys = [k for k in _WINDGRAM_CACHE if not k.startswith(ts_prefix)]
    for k in stale_keys:
        del _WINDGRAM_CACHE[k]

    _WINDGRAM_CACHE[cache_key] = CacheEntry(
        loaded_at=dt.datetime.now(dt.timezone.utc), data=df
    )
    return df


# ---------------------------------------------------------------------------
# Yr / MET Norway locationforecast
# ---------------------------------------------------------------------------


def _fetch_yr_forecast(lat: float, lon: float) -> list[dict]:
    """Fetch compact locationforecast from MET Norway and return hourly entries."""
    cache_key = f"{lat:.4f},{lon:.4f}"
    cached = _YR_CACHE.get(cache_key)
    if cached is not None and cached.is_fresh(settings.yr_ttl_seconds):
        return cached.data

    url = (
        f"https://api.met.no/weatherapi/locationforecast/2.0/compact"
        f"?lat={lat:.4f}&lon={lon:.4f}"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": _YR_USER_AGENT}, timeout=10)
        resp.raise_for_status()
        timeseries = resp.json()["properties"]["timeseries"]
    except Exception:
        logger.exception("Failed to fetch Yr forecast for %s,%s", lat, lon)
        if cached:
            return cached.data
        return []

    entries: list[dict] = []
    for ts in timeseries:
        time_utc = dt.datetime.fromisoformat(ts["time"].replace("Z", "+00:00"))
        instant = ts["data"]["instant"]["details"]
        next1 = ts["data"].get("next_1_hours", {})
        symbol = next1.get("summary", {}).get("symbol_code", "")
        precip = next1.get("details", {}).get("precipitation_amount")
        entries.append(
            {
                "time": time_utc,
                "symbol_code": symbol,
                "air_temperature": instant.get("air_temperature"),
                "precipitation": precip,
                "wind_speed": instant.get("wind_speed"),
                "cloud_area_fraction": instant.get("cloud_area_fraction"),
            }
        )

    _YR_CACHE[cache_key] = CacheEntry(
        loaded_at=dt.datetime.now(dt.timezone.utc), data=entries
    )
    return entries


def get_forecast_hours_for_day(
    name: str,
    day: dt.date,
    forecast_ts: Optional[dt.datetime] = None,
) -> list[int]:
    """Return the local hours that have MEPS forecast data for *name* on *day*."""
    df = load_windgram_data(name, day, forecast_ts=forecast_ts)
    if len(df) == 0:
        return []
    hours = (
        df.with_columns(time=pl.col("time").dt.convert_time_zone("Europe/Oslo"))
        .select(pl.col("time").dt.hour().alias("hour"))
        .unique()
        .sort("hour")
        .get_column("hour")
        .to_list()
    )
    return hours


def get_yr_weather_for_day(
    name: str,
    day: dt.date,
    restrict_to_hours: Optional[list[int]] = None,
) -> list[dict]:
    """Return Yr hourly weather for *name* on *day* (local time)."""
    local_tz = ZoneInfo("Europe/Oslo")
    meta = load_metadata()

    # Look up lat/lon for this takeoff
    loc = [t for t in meta.takeoffs if t.name == name and t.point_type != "area"]
    if not loc:
        return []
    lat = loc[0].latitude
    lon = loc[0].longitude

    entries = _fetch_yr_forecast(lat, lon)
    if not entries:
        return []

    allowed_hours = set(restrict_to_hours) if restrict_to_hours else None

    result = []
    for e in entries:
        local_time = e["time"].astimezone(local_tz)
        if local_time.date() != day or not e["symbol_code"]:
            continue
        if allowed_hours is not None and local_time.hour not in allowed_hours:
            continue
        if allowed_hours is None and not (7 <= local_time.hour <= 21):
            continue
        result.append(
            {
                **e,
                "local_hour": local_time.hour,
                "icon_url": f"{_YR_ICON_BASE}/{e['symbol_code']}.svg",
                "icon_png_url": f"{_YR_ICON_PNG_BASE}/{e['symbol_code']}.png",
            }
        )
    return result


# ---------------------------------------------------------------------------
# GeoJSON — loaded once (immutable at runtime)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_geojson() -> dict:
    geojson_path = Path(__file__).resolve().parents[2] / "Kommuner-S.geojson"
    with geojson_path.open("r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Map figure builder
# ---------------------------------------------------------------------------


def build_map_figure(
    selected_time: dt.datetime,
    selected_name: Optional[str],
    zoom: int,
    forecast_ts: Optional[dt.datetime] = None,
    wind_altitude: Optional[float] = None,
) -> go.Figure:
    map_df = load_map_data(selected_time, forecast_ts=forecast_ts)

    # Climb rate colorscale: 0–5 m/s
    thermal_colorscale = [
        (0.0, "rgb(200,200,200)"),  # 0 m/s – grey (no thermals)
        (0.2, "rgb(255,255,150)"),  # 1 m/s – light yellow (weak)
        (0.4, "rgb(255,220,50)"),  # 2 m/s – golden (moderate)
        (0.6, "rgb(255,140,0)"),  # 3 m/s – orange (good)
        (0.8, "rgb(230,60,0)"),  # 4 m/s – red-orange (strong)
        (1.0, "rgb(180,0,0)"),  # 5 m/s – deep red (extreme)
    ]

    fig = go.Figure()
    geojson = _load_geojson()

    # Kommune choropleth — transparent colored polygons for area-level climb rate
    subset_area = map_df.filter(pl.col("point_type") == "area")
    if len(subset_area) > 0:
        area_names = subset_area.get_column("name").to_numpy()
        area_velocity = subset_area.get_column("peak_thermal_velocity").to_numpy()
        area_thermal_top = subset_area.get_column("thermal_top").to_numpy()
        fig.add_trace(
            go.Choroplethmap(
                geojson=geojson,
                zmin=0,
                zmax=5,
                featureidkey="properties.name",
                locations=area_names,
                ids=area_names,
                z=area_velocity,
                colorscale=thermal_colorscale,
                marker_opacity=0.15,
                showscale=False,
                showlegend=False,
                hoverinfo="text",
                hovertext=[
                    f"{name} | Climb: {vel:.1f} m/s | Top: {ht:.0f} m"
                    for vel, ht, name in zip(
                        area_velocity, area_thermal_top, area_names
                    )
                ],
            )
        )

    center = {"lat": 61.2, "lon": 8.0}
    subset_points = map_df.filter(pl.col("point_type") != "area")
    if len(subset_points) > 0:
        lat = subset_points.get_column("latitude").to_numpy()
        lon = subset_points.get_column("longitude").to_numpy()
        peak_velocity = subset_points.get_column("peak_thermal_velocity").to_numpy()
        thermal_top = subset_points.get_column("thermal_top").to_numpy().round()
        names = subset_points.get_column("name").to_numpy()
        if selected_name is not None:
            selected = names == selected_name
        else:
            selected = np.zeros_like(names, dtype=bool)
        marker_size = np.where(selected, 18, 11)
        if selected.any():
            selected_idx = int(np.argmax(selected))
            center = {
                "lat": float(lat[selected_idx]),
                "lon": float(lon[selected_idx]),
            }
        else:
            center = {"lat": float(lat.mean()), "lon": float(lon.mean())}

        # Outline layer — dark ring behind each marker for contrast
        outline_size = marker_size + 5
        fig.add_trace(
            go.Scattermap(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=go.scattermap.Marker(
                    size=outline_size,
                    color="rgba(30,41,59,0.5)",
                    opacity=1,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Main colored markers — colored by peak climb rate
        fig.add_trace(
            go.Scattermap(
                lat=lat,
                lon=lon,
                mode="markers",
                marker=go.scattermap.Marker(
                    size=marker_size,
                    cmin=0,
                    cmax=5,
                    color=peak_velocity,
                    colorscale=thermal_colorscale,
                    opacity=1,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Climb rate (m/s)",
                            side="right",
                            font=dict(size=10),
                        ),
                        orientation="h",
                        y=-0.02,
                        yanchor="top",
                        thickness=10,
                        len=0.6,
                        x=0.5,
                        xanchor="center",
                        tickfont=dict(size=10),
                    ),
                ),
                ids=names,
                customdata=names,
                text=[
                    f"{name} | {vel:.1f} m/s | top: {ht} m"
                    for vel, ht, name in zip(peak_velocity, thermal_top, names)
                ],
                hoverinfo="text",
                showlegend=False,
            )
        )

    # --- Wind overlay (gridded) ---
    # Two traces only: (1) thin direction lines, (2) colored speed markers.
    # At lower zoom levels we subsample to avoid clutter; at higher zoom
    # we show the full grid.
    if wind_altitude is not None:
        grid_df = load_grid_wind_data(
            selected_time, altitude=wind_altitude, forecast_ts=forecast_ts
        )
        if len(grid_df) > 0:
            g_lat = grid_df.get_column("latitude").to_numpy()
            g_lon = grid_df.get_column("longitude").to_numpy()
            g_u = grid_df.get_column("x_wind_ml").to_numpy()
            g_v = grid_df.get_column("y_wind_ml").to_numpy()
            g_spd = grid_df.get_column("wind_speed").to_numpy()

            # Drop NaN / near-zero
            valid = np.isfinite(g_spd) & (g_spd > 0.1)
            g_lat, g_lon = g_lat[valid], g_lon[valid]
            g_u, g_v, g_spd = g_u[valid], g_v[valid], g_spd[valid]

            # Adaptive thinning: the grid DB is already stride-3 (~7.5 km).
            # At zoom ≤6 show every 2nd point (~15 km), zoom ≥7 show all.
            if zoom <= 6:
                thin = 2
            else:
                thin = 1
            if thin > 1:
                # Subsample based on spatial index (approx. regular grid)
                keep = np.zeros(len(g_lat), dtype=bool)
                dlat = 0.075 * thin  # degrees — base spacing ~0.075°
                qi = (g_lat / dlat).astype(int)
                qj = (g_lon / dlat).astype(int)
                seen: set[tuple[int, int]] = set()
                for idx in range(len(g_lat)):
                    key = (qi[idx], qj[idx])
                    if key not in seen:
                        seen.add(key)
                        keep[idx] = True
                g_lat, g_lon = g_lat[keep], g_lon[keep]
                g_u, g_v, g_spd = g_u[keep], g_v[keep], g_spd[keep]

            # Arrow length adapts to zoom so arrows look proportional.
            arrow_len_deg = 0.04 * (2 ** (zoom - 6))  # ~0.04° at z6, ~0.16° at z8
            arrow_len_deg = min(arrow_len_deg, 0.15)
            cos_lat = np.cos(np.radians(g_lat))
            mag = np.maximum(g_spd, 0.01)
            du = g_u / mag
            dv = g_v / mag
            tip_lat = g_lat + dv * arrow_len_deg
            tip_lon = g_lon + du * arrow_len_deg / cos_lat

            # --- Trace 1: direction lines (single trace, subtle) ---
            n = len(g_lat)
            seg_lat = np.empty(n * 3, dtype=object)
            seg_lon = np.empty(n * 3, dtype=object)
            seg_lat[0::3] = g_lat
            seg_lat[1::3] = tip_lat
            seg_lat[2::3] = None
            seg_lon[0::3] = g_lon
            seg_lon[1::3] = tip_lon
            seg_lon[2::3] = None

            fig.add_trace(
                go.Scattermap(
                    lat=seg_lat.tolist(),
                    lon=seg_lon.tolist(),
                    mode="lines",
                    line=dict(width=1.5, color="rgba(40,40,60,0.4)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            # --- Trace 2: coloured speed dots with text labels ---
            alt_label = "sfc" if wind_altitude == 0 else f"{int(wind_altitude)}m"
            # Compass direction (meteorological: where wind comes FROM)
            wind_dir_deg = (np.degrees(np.arctan2(-g_u, -g_v)) + 360) % 360
            compass = [_deg_to_compass(d) for d in wind_dir_deg]

            fig.add_trace(
                go.Scattermap(
                    lat=g_lat.tolist(),
                    lon=g_lon.tolist(),
                    mode="markers",
                    marker=go.scattermap.Marker(
                        size=6,
                        color=[_wind_arrow_color(float(s)) for s in g_spd],
                        opacity=0.85,
                        symbol="circle",
                        allowoverlap=True,
                    ),
                    hoverinfo="text",
                    hovertext=[
                        f"{spd:.0f} m/s from {c} ({alt_label})"
                        for spd, c in zip(g_spd, compass)
                    ],
                    showlegend=False,
                )
            )

    fig.update_layout(
        map_style="open-street-map",
        map=dict(center=center, zoom=zoom),
        margin={"r": 0, "t": 0, "l": 0, "b": 34},
        uirevision="map-keep-view",
        autosize=True,
        showlegend=False,
    )

    # Time badge — top-right corner
    local_tz = ZoneInfo("Europe/Oslo")
    local_time = selected_time.astimezone(local_tz)
    time_label = local_time.strftime("%a %d %b %H:%M")
    fig.add_annotation(
        text=f"<b>{time_label}</b>",
        x=1,
        y=1,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font=dict(size=13, color="#1e293b"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(200,200,200,0.5)",
        borderwidth=1,
        borderpad=5,
    )

    # Selected takeoff name label — top-left corner
    if selected_name:
        fig.add_annotation(
            text=f"<b>{selected_name}</b>",
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=13, color="#1e293b"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1,
            borderpad=5,
        )

    if len(fig.data) == 0:
        fig.add_annotation(
            text="No map data for selected time.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    return fig


# ---------------------------------------------------------------------------
# Wind arrow color
# ---------------------------------------------------------------------------


def _wind_arrow_color(
    wind_speed: float,
    thresholds: tuple[float, ...] = (2, 4, 6, 8, 12),
    colors: tuple[str, ...] = (
        "#b0b0b0",  # calm – grey
        "#4caf50",  # light – green
        "#ffeb3b",  # moderate – yellow
        "#ff9800",  # fresh – orange
        "#f44336",  # strong – red
        "#4a148c",  # very strong – dark purple
    ),
) -> str:
    """Map wind speed to an arrow colour (discrete buckets)."""
    for i, thr in enumerate(thresholds):
        if wind_speed < thr:
            return colors[i]
    return colors[-1]


_COMPASS_LABELS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]


def _deg_to_compass(deg: float) -> str:
    """Convert meteorological wind direction (degrees) to compass label."""
    return _COMPASS_LABELS[int((deg + 11.25) % 360 / 22.5)]


# Thermal velocity colorscale: 0–5 m/s climb rate
_THERMAL_COLORSCALE = [
    [0.0, "rgb(255,255,255)"],  # 0 m/s – white (no thermals)
    [0.01, "rgb(255,255,255)"],  # tiny buffer to keep near-zero white
    [0.02, "rgb(255,255,210)"],  # barely above threshold – visible tint
    [0.10, "rgb(255,255,150)"],  # ~0.5 m/s – weak, light yellow
    [0.20, "rgb(255,245,80)"],  # ~1.0 m/s – moderate, yellow
    [0.40, "rgb(255,220,50)"],  # ~2.0 m/s – good, golden
    [0.60, "rgb(255,180,30)"],  # ~3.0 m/s – strong, orange-yellow
    [0.80, "rgb(255,140,0)"],  # ~4.0 m/s – very strong, orange
    [1.0, "rgb(255,80,0)"],  # ~5.0 m/s – extreme, deep orange
]


# ---------------------------------------------------------------------------
# Airgram figure builder
# ---------------------------------------------------------------------------


def build_airgram_figure(
    target_name: str,
    selected_date: dt.date,
    altitude_max: int,
    yr_entries: Optional[list[dict]] = None,
    selected_hour: Optional[int] = None,
    forecast_ts: Optional[dt.datetime] = None,
) -> go.Figure:
    location_data = load_windgram_data(
        target_name, selected_date, forecast_ts=forecast_ts
    )

    display_start_hour = 8
    display_end_hour = 21
    location_data = location_data.with_columns(
        time=pl.col("time").dt.convert_time_zone("Europe/Oslo")
    ).filter(
        (pl.col("time").dt.hour().is_between(display_start_hour, display_end_hour)),
    )

    if len(location_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No airgram data for this location/day.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=500)
        return fig

    # Determine ground elevation for this location
    elevation = float(location_data["elevation"].min())

    new_timestamps = location_data.select("time").to_series().unique().sort().to_list()
    # Use 200m steps for denser grid (like meteo-parapente)
    altitudes = np.arange(0.0, float(altitude_max) + 200.0, 200)

    output_frame = (
        pl.DataFrame({"time": [new_timestamps], "altitude": [altitudes]})
        .explode("time")
        .explode("altitude")
        .sort("altitude")
    )

    plot_frame = (
        output_frame.sort("time", "altitude")
        .with_columns(pl.col("altitude").set_sorted())
        .join_asof(
            location_data.sort("time", "altitude").with_columns(
                pl.col("altitude").set_sorted()
            ),
            on="altitude",
            by="time",
            strategy="nearest",
        )
        .with_columns(
            wind_direction=-pl.arctan2("y_wind_ml", "x_wind_ml").degrees() + 90
        )
        .sort("time")
    )

    # Format time labels as "HHh"
    time_labels = []
    for t in new_timestamps:
        if hasattr(t, "strftime"):
            time_labels.append(t.strftime("%Hh"))
        else:
            time_labels.append(str(t))

    # Add a formatted time_label column for consistent x-axis values
    ts_to_label = {t: lbl for t, lbl in zip(new_timestamps, time_labels)}
    plot_frame = plot_frame.with_columns(
        pl.col("time")
        .map_elements(lambda t: ts_to_label.get(t, str(t)), return_dtype=pl.Utf8)
        .alias("time_label")
    )

    fig = go.Figure()

    # --- 1) Thermal heatmap (background) ---
    # Zero out thermal_velocity above the computed thermal_top so the
    # heatmap boundary matches the thermal top line exactly.
    plot_frame = plot_frame.with_columns(
        pl.when(pl.col("altitude") > pl.col("thermal_top"))
        .then(0.0)
        .otherwise(pl.col("thermal_velocity"))
        .alias("thermal_velocity")
    )

    # Pivot thermal data into a 2D grid for the heatmap
    thermal_pivot = plot_frame.pivot(
        on="time_label", index="altitude", values="thermal_velocity"
    ).sort("altitude")
    z_altitudes = thermal_pivot["altitude"].to_numpy()
    # Columns are the time labels; select them in the right order
    z_cols = [c for c in time_labels if c in thermal_pivot.columns]
    z_matrix = thermal_pivot.select(z_cols).to_numpy()

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=z_cols,
            y=z_altitudes,
            colorscale=_THERMAL_COLORSCALE,
            zmin=0,
            zmax=5,
            zsmooth=False,
            showscale=False,
            hovertemplate=(
                "Alt: %{y:.0f}m<br>Time: %{x}<br>Climb: %{z:.1f} m/s<extra></extra>"
            ),
        )
    )

    # --- 1b) Thermal top line ---
    # Show computed thermal top as a dashed line so boundary is clear
    thermal_tops_per_time = (
        plot_frame.group_by("time_label")
        .agg(pl.col("thermal_top").first())
        .sort(
            pl.col("time_label").map_elements(
                lambda lbl: time_labels.index(lbl) if lbl in time_labels else 999,
                return_dtype=pl.Int64,
            )
        )
    )
    tt_labels = thermal_tops_per_time["time_label"].to_list()
    tt_vals = thermal_tops_per_time["thermal_top"].to_numpy()
    # Only show line where thermal top is above ground
    tt_y = [float(v) if v > elevation + 50 else None for v in tt_vals]
    fig.add_trace(
        go.Scatter(
            x=tt_labels,
            y=tt_y,
            mode="lines",
            line=dict(color="rgba(180,80,0,0.7)", width=2, dash="dot"),
            hovertemplate="Thermal top: %{y:.0f}m<extra></extra>",
            showlegend=False,
        )
    )

    # --- 2) Ground shading (grey area below terrain) ---
    fig.add_shape(
        type="rect",
        x0=time_labels[0],
        x1=time_labels[-1],
        y0=0,
        y1=elevation,
        fillcolor="rgba(180,180,180,0.7)",
        line=dict(width=0),
        layer="above",
    )

    # --- 3) Wind arrows with speed labels (overlaid) ---
    # Pick altitudes from the actual data grid closest to every 500m
    available_alts = np.array(sorted(plot_frame["altitude"].unique().to_list()))
    above_ground = available_alts[available_alts >= elevation]
    if len(above_ground) > 0:
        target_alts = np.arange(above_ground[0], float(altitude_max) + 1, 250)
        # Snap each target to the nearest altitude in the data grid
        wind_altitudes = np.unique(
            [above_ground[np.argmin(np.abs(above_ground - t))] for t in target_alts]
        )
    else:
        wind_altitudes = np.array([])
    plot_frame_wind = plot_frame.sort("time", "altitude").filter(
        pl.col("altitude").is_in(wind_altitudes.tolist())
    )

    if len(plot_frame_wind) > 0:
        wind_time_labels = plot_frame_wind["time_label"].to_list()
        wind_alts = plot_frame_wind["altitude"].to_numpy()
        wind_dirs = plot_frame_wind["wind_direction"].to_numpy()
        wind_spds = plot_frame_wind["wind_speed"].to_numpy()
        thermal_vels = plot_frame_wind["thermal_velocity"].to_numpy()

        # Arrow markers with speed labels
        fig.add_trace(
            go.Scatter(
                x=wind_time_labels,
                y=wind_alts,
                mode="markers+text",
                marker=dict(
                    symbol="arrow",
                    size=12,
                    angle=wind_dirs,
                    color=[_wind_arrow_color(float(s)) for s in wind_spds],
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                ),
                text=[f"{spd:.0f}" for spd in wind_spds],
                textposition="middle right",
                textfont=dict(size=9, color="#555", family="Arial"),
                hoverinfo="text",
                hovertext=[
                    f"Alt: {alt:.0f}m | Wind: {spd:.0f} m/s | {angle:.0f}° | Climb: {tv:.1f} m/s"
                    for alt, spd, angle, tv in zip(
                        wind_alts, wind_spds, wind_dirs, thermal_vels
                    )
                ],
                showlegend=False,
                cliponaxis=False,
            )
        )

    # --- 4) Yr weather icons & temp above the chart ---
    yr_by_label: dict[str, dict] = {}
    if yr_entries:
        for e in yr_entries:
            label = f"{e['local_hour']:02d}h"
            yr_by_label[label] = e

    has_yr = bool(yr_by_label)
    yr_margin_t = 100 if has_yr else 30

    images = []
    annotations = []
    if has_yr:
        n_cols = len(time_labels)
        for i, label in enumerate(time_labels):
            e = yr_by_label.get(label)
            if not e:
                continue
            x_frac = (i + 0.5) / n_cols

            # Weather icon above the hour labels
            icon_size = 1.0 / n_cols  # scale to column width
            icon_size = min(icon_size, 0.07)  # cap on wide screens
            images.append(
                dict(
                    source=e["icon_png_url"],
                    x=x_frac,
                    y=1.06,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="bottom",
                    sizex=icon_size,
                    sizey=icon_size,
                    sizing="contain",
                    layer="above",
                )
            )

            # Temperature text above the icon
            temp = e.get("air_temperature")
            temp_str = f"{temp:.0f}°" if temp is not None else ""
            annotations.append(
                dict(
                    x=x_frac,
                    y=1.16,
                    xref="paper",
                    yref="paper",
                    text=f"<b>{temp_str}</b>",
                    showarrow=False,
                    font=dict(size=10, color="#333"),
                )
            )

    # --- 5) Highlighted column for selected hour ---
    selected_label = f"{selected_hour:02d}h" if selected_hour is not None else None
    if selected_label and selected_label in time_labels:
        idx = time_labels.index(selected_label)
        # For a category axis, shape x coords use category values directly.
        # To span the full column width we go from idx-0.5 to idx+0.5
        fig.add_shape(
            type="rect",
            x0=idx - 0.5,
            x1=idx + 0.5,
            y0=0,
            y1=altitude_max,
            xref="x",
            yref="y",
            fillcolor="rgba(59,130,246,0.10)",
            line=dict(width=1.5, color="rgba(59,130,246,0.4)"),
            layer="above",
        )

    # --- 6) Snow depth annotation at bottom of chart ---
    if "snow_depth" in location_data.columns:
        snow_vals = location_data.select("snow_depth").to_series().drop_nulls()
        if len(snow_vals) > 0:
            median_snow_m = float(snow_vals.median())
            if median_snow_m > 0.01:  # >1 cm
                snow_cm = median_snow_m * 100
                snow_text = (
                    f"\u2744\ufe0f {snow_cm:.0f} cm snow"
                    if snow_cm >= 1
                    else "\u2744\ufe0f <1 cm snow"
                )
                annotations.append(
                    dict(
                        x=0.01,
                        y=0.01,
                        xref="paper",
                        yref="paper",
                        text=f"<b>{snow_text}</b>",
                        showarrow=False,
                        font=dict(size=12, color="#4a90d9"),
                        bgcolor="rgba(255,255,255,0.8)",
                        borderpad=3,
                        xanchor="left",
                        yanchor="bottom",
                    )
                )

    # --- Layout ---
    fig.update_layout(
        height=450,
        images=images,
        annotations=annotations,
        xaxis=dict(
            title="",
            tickangle=0,
            type="category",
            categoryorder="array",
            categoryarray=time_labels,
            gridcolor="rgba(200,200,200,0.4)",
            showgrid=True,
            side="top",
            fixedrange=True,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="",
            range=[0, altitude_max],
            dtick=500,
            gridcolor="rgba(200,200,200,0.4)",
            showgrid=True,
            fixedrange=True,
            tickfont=dict(size=10),
            ticksuffix="m",
        ),
        plot_bgcolor="white",
        autosize=True,
        margin=dict(l=46, r=10, t=yr_margin_t, b=6),
        showlegend=False,
        dragmode=False,
    )

    return fig


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def get_summary(
    selected_name: str,
    selected_time: dt.datetime,
    forecast_ts: Optional[dt.datetime] = None,
) -> str:
    map_df = load_map_data(selected_time, forecast_ts=forecast_ts)
    row = map_df.filter(
        (pl.col("point_type") != "area") & (pl.col("name") == selected_name)
    )
    if len(row) == 0:
        return "No detailed point data for current selection."
    return (
        f"Selected: {row[0, 'name']} | Time: {selected_time} UTC | "
        f"Thermal top: {row[0, 'thermal_top']:.0f} m | "
        f"Wind: {row[0, 'wind_speed']:.1f} m/s"
    )
