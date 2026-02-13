from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl
import requests


import db_utils

logger = logging.getLogger(__name__)

DATA_TTL_SECONDS = 600
_CACHE: dict[str, object] = {"loaded_at": None, "df": None}

# ---------------------------------------------------------------------------
# Yr / MET Norway locationforecast
# ---------------------------------------------------------------------------
_YR_CACHE: dict[str, object] = {}  # key=(lat,lon) -> {"fetched_at": dt, "data": [...]}
_YR_TTL_SECONDS = 1800  # 30 min cache per location
_YR_ICON_BASE = "https://raw.githubusercontent.com/metno/weathericons/main/weather/svg"
_YR_USER_AGENT = "pgweather/1.0 github.com/simeneide/pgweather"


def _fetch_yr_forecast(lat: float, lon: float) -> list[dict]:
    """Fetch compact locationforecast from MET Norway and return hourly entries."""
    cache_key = f"{lat:.4f},{lon:.4f}"
    now = dt.datetime.now(dt.timezone.utc)
    cached = _YR_CACHE.get(cache_key)
    if (
        cached
        and isinstance(cached.get("fetched_at"), dt.datetime)
        and (now - cached["fetched_at"]).total_seconds() < _YR_TTL_SECONDS
    ):
        return cached["data"]

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
            return cached["data"]
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

    _YR_CACHE[cache_key] = {"fetched_at": now, "data": entries}
    return entries


def get_yr_weather_for_day(
    name: str, day: dt.date, df: Optional[pl.DataFrame] = None
) -> list[dict]:
    """Return Yr hourly weather for *name* on *day* (local time), 07-21h."""
    from zoneinfo import ZoneInfo

    local_tz = ZoneInfo("Europe/Oslo")
    frame = df if df is not None else load_forecast_data()

    # Look up lat/lon for this takeoff
    loc = (
        frame.filter((pl.col("point_type") != "area") & (pl.col("name") == name))
        .select("latitude", "longitude")
        .unique()
        .head(1)
    )
    if len(loc) == 0:
        return []
    lat = float(loc[0, "latitude"])
    lon = float(loc[0, "longitude"])

    entries = _fetch_yr_forecast(lat, lon)
    if not entries:
        return []

    # Filter to the requested local day, 07-21h
    result = []
    for e in entries:
        local_time = e["time"].astimezone(local_tz)
        if local_time.date() == day and 7 <= local_time.hour <= 21 and e["symbol_code"]:
            result.append(
                {
                    **e,
                    "local_hour": local_time.hour,
                    "icon_url": f"{_YR_ICON_BASE}/{e['symbol_code']}.svg",
                }
            )
    return result


def _interpolate_color(
    wind_speed: float,
    thresholds: list[float] = [2, 4, 5, 14],
    colors: list[str] = ["grey", "green", "orange", "red", "black"],
) -> str:
    norm_thresholds = [t / max(thresholds) for t in thresholds]
    norm_thresholds = [0] + norm_thresholds + [1]
    extended_colors = [colors[0]] + colors + [colors[-1]]
    cmap = LinearSegmentedColormap.from_list(
        "wind_speed_cmap", list(zip(norm_thresholds, extended_colors)), N=256
    )
    norm_wind_speed = wind_speed / max(thresholds)
    return to_hex(cmap(np.clip(norm_wind_speed, 0, 1)))


def _load_geojson() -> dict:
    geojson_path = Path(__file__).resolve().parents[2] / "Kommuner-S.geojson"
    with geojson_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_forecast_data(force_refresh: bool = False) -> pl.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    loaded_at = _CACHE["loaded_at"]
    cached_df = _CACHE["df"]

    if (
        (not force_refresh)
        and isinstance(loaded_at, dt.datetime)
        and isinstance(cached_df, pl.DataFrame)
        and (now - loaded_at).total_seconds() < DATA_TTL_SECONDS
    ):
        return cached_df

    db = db_utils.Database()
    query = """
    select *
    from detailed_forecasts
    where forecast_timestamp = (
      select max(forecast_timestamp) from detailed_forecasts
    )
    """
    df = db.read(query).with_columns(
        [
            pl.col("forecast_timestamp").cast(pl.Datetime).dt.replace_time_zone("UTC"),
            pl.col("time").cast(pl.Datetime).dt.replace_time_zone("UTC"),
        ]
    )
    _CACHE["loaded_at"] = now
    _CACHE["df"] = df
    return df


def get_latest_forecast_timestamp(df: Optional[pl.DataFrame] = None) -> dt.datetime:
    frame = df if df is not None else load_forecast_data()
    return frame.get_column("forecast_timestamp").max()


def get_available_times(df: Optional[pl.DataFrame] = None) -> list[dt.datetime]:
    frame = df if df is not None else load_forecast_data()
    return frame.get_column("time").unique().sort().to_list()


def get_takeoff_names(df: Optional[pl.DataFrame] = None) -> list[str]:
    frame = df if df is not None else load_forecast_data()
    return (
        frame.filter(pl.col("point_type") != "area")
        .get_column("name")
        .unique()
        .sort()
        .to_list()
    )


def get_takeoff_options(df: Optional[pl.DataFrame] = None) -> list[dict[str, str]]:
    frame = df if df is not None else load_forecast_data()
    reference_time = frame.get_column("time").max()

    takeoffs = (
        frame.filter(
            (pl.col("point_type") != "area") & (pl.col("time") == reference_time)
        )
        .select("name", "latitude", "longitude")
        .unique("name")
        .sort("name")
    )
    areas = (
        frame.filter(
            (pl.col("point_type") == "area") & (pl.col("time") == reference_time)
        )
        .select("name", "latitude", "longitude")
        .unique("name")
    )

    if len(takeoffs) == 0:
        return []
    if len(areas) == 0:
        return [
            {"label": name, "value": name}
            for name in takeoffs.get_column("name").to_list()
        ]

    area_names = areas.get_column("name").to_numpy()
    area_lat = areas.get_column("latitude").to_numpy()
    area_lon = areas.get_column("longitude").to_numpy()

    # Max squared distance (~0.5 deg ≈ 50 km) beyond which we skip region label
    max_dist2 = 0.25

    options: list[dict[str, str]] = []
    for row in takeoffs.iter_rows(named=True):
        name = str(row["name"])
        # Skip region if the name already contains parentheses (has its own qualifier)
        if "(" in name:
            options.append({"label": name, "value": name})
            continue
        dist2 = (area_lat - row["latitude"]) ** 2 + (area_lon - row["longitude"]) ** 2
        min_dist2 = float(np.min(dist2))
        if min_dist2 > max_dist2:
            # Too far from any area — don't add a misleading region
            options.append({"label": name, "value": name})
        else:
            idx = int(np.argmin(dist2))
            region = str(area_names[idx])
            options.append({"label": f"{name} ({region})", "value": name})
    return options


def get_default_selected_time(df: Optional[pl.DataFrame] = None) -> dt.datetime:
    frame = df if df is not None else load_forecast_data()
    times = get_available_times(frame)
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    return min(times, key=lambda t: abs((t - now).total_seconds()))


def build_map_figure(
    selected_time: dt.datetime,
    selected_name: Optional[str],
    map_layer: str,
    zoom: int,
    df: Optional[pl.DataFrame] = None,
) -> go.Figure:
    frame = df if df is not None else load_forecast_data()
    show_area = map_layer in ["both", "areas"]
    show_takeoffs = map_layer in ["both", "takeoffs"]
    geojson = _load_geojson()

    thermal_colorscale = [
        (0.0, "grey"),
        (0.2, "yellow"),
        (0.4, "red"),
        (0.6, "violet"),
        (1.0, "black"),
    ]

    fig = go.Figure()

    if show_area:
        subset_area = frame.filter(
            (pl.col("time") == selected_time) & (pl.col("point_type") == "area")
        )
        if len(subset_area) > 0:
            names = subset_area.get_column("name").to_numpy()
            thermal_top = subset_area.get_column("thermal_top").to_numpy().round()
            fig.add_trace(
                go.Choroplethmap(
                    geojson=geojson,
                    zmin=0,
                    zmax=5000,
                    featureidkey="properties.name",
                    locations=names,
                    ids=names,
                    z=thermal_top,
                    colorscale=thermal_colorscale,
                    marker_opacity=0.35,
                    showscale=True,
                    colorbar=dict(title="Thermal top (m)"),
                    hovertext=[
                        f"{name} | median thermal top: {ht} m"
                        for ht, name in zip(thermal_top, names)
                    ],
                )
            )

    center = {"lat": 61.2, "lon": 8.0}
    if show_takeoffs:
        subset_points = frame.filter(
            (pl.col("time") == selected_time) & (pl.col("point_type") != "area")
        )
        if len(subset_points) > 0:
            lat = subset_points.get_column("latitude").to_numpy()
            lon = subset_points.get_column("longitude").to_numpy()
            thermal_top = subset_points.get_column("thermal_top").to_numpy().round()
            names = subset_points.get_column("name").to_numpy()
            if selected_name is not None:
                selected = names == selected_name
            else:
                selected = np.zeros_like(names, dtype=bool)
            marker_size = np.where(selected, 20, 9)
            if selected.any():
                selected_idx = int(np.argmax(selected))
                center = {
                    "lat": float(lat[selected_idx]),
                    "lon": float(lon[selected_idx]),
                }
            else:
                center = {"lat": float(lat.mean()), "lon": float(lon.mean())}
            fig.add_trace(
                go.Scattermap(
                    lat=lat,
                    lon=lon,
                    mode="markers",
                    marker=go.scattermap.Marker(
                        size=marker_size,
                        cmin=0,
                        cmax=5000,
                        color=thermal_top,
                        colorscale=thermal_colorscale,
                        opacity=1,
                        showscale=False,
                    ),
                    ids=names,
                    customdata=names,
                    text=[
                        f"{name} | thermal top: {ht} m"
                        for ht, name in zip(thermal_top, names)
                    ],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        map_style="open-street-map",
        map=dict(center=center, zoom=zoom),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        uirevision="map-keep-view",
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


# Thermal colorscale: grey -> yellow -> orange (mimics meteo-parapente)
_THERMAL_COLORSCALE = [
    [0.0, "rgba(220,220,220,0.0)"],  # zero diff – transparent
    [0.05, "rgba(255,255,200,0.3)"],  # very slight – faint yellow
    [0.15, "rgba(255,255,100,0.6)"],  # weak thermal – yellow
    [0.30, "rgba(255,220,50,0.8)"],  # moderate – golden
    [0.50, "rgba(255,180,30,0.9)"],  # good thermal – orange-yellow
    [0.70, "rgba(255,140,0,0.95)"],  # strong – orange
    [1.0, "rgba(255,80,0,1.0)"],  # very strong – deep orange
]


def build_airgram_figure(
    target_name: str,
    selected_date: dt.date,
    altitude_max: int,
    df: Optional[pl.DataFrame] = None,
) -> go.Figure:
    frame = df if df is not None else load_forecast_data()
    display_start_hour = 8
    display_end_hour = 21
    location_data = frame.with_columns(
        time=pl.col("time").dt.convert_time_zone("Europe/Oslo")
    ).filter(
        (pl.col("time").dt.date() == selected_date),
        (pl.col("time").dt.hour().is_between(display_start_hour, display_end_hour)),
        pl.col("name") == target_name,
        pl.col("point_type") != "area",
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
        output_frame.join_asof(
            location_data.sort("altitude"), on="altitude", by="time", strategy="nearest"
        )
        .with_columns(
            wind_direction=-pl.arctan2("y_wind_ml", "x_wind_ml").degrees() + 90
        )
        .sort("time")
    )

    fig = go.Figure()

    # --- 1) Thermal heatmap (background) ---
    # Pivot thermal data into a 2D grid for the heatmap
    thermal_pivot = plot_frame.pivot(
        on="time", index="altitude", values="thermal_temp_diff"
    ).sort("altitude")
    z_altitudes = thermal_pivot["altitude"].to_numpy()
    z_times = [c for c in thermal_pivot.columns if c != "altitude"]
    z_matrix = thermal_pivot.select(z_times).to_numpy()

    # Format time labels as "HHh"
    time_labels = []
    for t in new_timestamps:
        if hasattr(t, "strftime"):
            time_labels.append(t.strftime("%Hh"))
        else:
            time_labels.append(str(t))

    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=time_labels,
            y=z_altitudes,
            colorscale=_THERMAL_COLORSCALE,
            zmin=0,
            zmax=6,
            showscale=False,
            hovertemplate=(
                "Alt: %{y:.0f}m<br>Time: %{x}<br>"
                "Thermal diff: %{z:.1f}°C<extra></extra>"
            ),
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
    # Use every 250m for wind arrows to get a dense grid like the reference
    wind_step = 250
    wind_altitudes = np.arange(
        max(wind_step, np.ceil(elevation / wind_step) * wind_step),
        float(altitude_max) + wind_step,
        wind_step,
    )
    plot_frame_wind = plot_frame.sort("time", "altitude").filter(
        pl.col("altitude").is_in(wind_altitudes)
    )

    if len(plot_frame_wind) > 0:
        wind_times_raw = plot_frame_wind["time"].to_list()
        wind_time_labels = []
        for t in wind_times_raw:
            if hasattr(t, "strftime"):
                wind_time_labels.append(t.strftime("%Hh"))
            else:
                wind_time_labels.append(str(t))
        wind_alts = plot_frame_wind["altitude"].to_numpy()
        wind_dirs = plot_frame_wind["wind_direction"].to_numpy()
        wind_spds = plot_frame_wind["wind_speed"].to_numpy()

        # Arrow markers
        fig.add_trace(
            go.Scatter(
                x=wind_time_labels,
                y=wind_alts,
                mode="markers",
                marker=dict(
                    symbol="arrow",
                    size=14,
                    angle=wind_dirs,
                    color=[_wind_arrow_color(float(s)) for s in wind_spds],
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                ),
                hoverinfo="text",
                text=[
                    f"Alt: {alt:.0f}m | {spd:.0f} m/s | {angle:.0f}°"
                    for alt, spd, angle in zip(wind_alts, wind_spds, wind_dirs)
                ],
                showlegend=False,
                cliponaxis=False,
            )
        )

        # Wind speed text labels next to arrows
        fig.add_trace(
            go.Scatter(
                x=wind_time_labels,
                y=wind_alts,
                mode="text",
                text=[f"{spd:.0f}" for spd in wind_spds],
                textposition="middle right",
                textfont=dict(size=10, color="#333", family="Arial"),
                showlegend=False,
                hoverinfo="skip",
                cliponaxis=False,
            )
        )

    # --- Layout ---
    fig.update_layout(
        height=700,
        title=dict(
            text=(
                f"Windgram – {target_name}<br>"
                f"<span style='font-size:13px;color:#666'>"
                f"{selected_date.strftime('%A %d %B %Y')}</span>"
            ),
            font=dict(size=16),
        ),
        xaxis=dict(
            title="",
            tickangle=0,
            type="category",
            categoryorder="array",
            categoryarray=time_labels,
            gridcolor="rgba(200,200,200,0.4)",
            showgrid=True,
        ),
        yaxis=dict(
            title="Altitude (m)",
            range=[0, altitude_max],
            dtick=500,
            gridcolor="rgba(200,200,200,0.4)",
            showgrid=True,
        ),
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=70, b=40),
        showlegend=False,
    )

    return fig


def get_summary(
    selected_name: str,
    selected_time: dt.datetime,
    df: Optional[pl.DataFrame] = None,
) -> str:
    frame = df if df is not None else load_forecast_data()
    selected = (
        frame.filter(
            (pl.col("point_type") != "area")
            & (pl.col("name") == selected_name)
            & (pl.col("time") == selected_time)
            & (pl.col("altitude") <= 1000)
        )
        .sort("altitude", descending=True)
        .head(1)
    )
    if len(selected) == 0:
        return "No detailed point data for current selection."
    return (
        f"Selected: {selected[0, 'name']} | Time: {selected_time} UTC | "
        f"Thermal top: {selected[0, 'thermal_top']:.0f} m | "
        f"Wind: {selected[0, 'wind_speed']:.1f} m/s"
    )
