from __future__ import annotations

import datetime as dt
import logging
from typing import Optional
from zoneinfo import ZoneInfo

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
_YR_ICON_PNG_BASE = (
    "https://raw.githubusercontent.com/metno/weathericons/main/weather/png"
)
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


def get_forecast_hours_for_day(
    name: str, day: dt.date, df: Optional[pl.DataFrame] = None
) -> list[int]:
    """Return the local hours that have MEPS forecast data for *name* on *day*."""
    frame = df if df is not None else load_forecast_data()
    hours = (
        frame.with_columns(time=pl.col("time").dt.convert_time_zone("Europe/Oslo"))
        .filter(
            (pl.col("time").dt.date() == day),
            pl.col("name") == name,
            pl.col("point_type") != "area",
        )
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
    df: Optional[pl.DataFrame] = None,
    restrict_to_hours: Optional[list[int]] = None,
) -> list[dict]:
    """Return Yr hourly weather for *name* on *day* (local time).

    If *restrict_to_hours* is given, only return entries whose local hour
    is in that list (used to sync Yr strip with MEPS forecast hours).
    """
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
    zoom: int,
    df: Optional[pl.DataFrame] = None,
) -> go.Figure:
    frame = df if df is not None else load_forecast_data()

    thermal_colorscale = [
        (0.0, "grey"),
        (0.2, "yellow"),
        (0.4, "red"),
        (0.6, "violet"),
        (1.0, "black"),
    ]

    fig = go.Figure()

    center = {"lat": 61.2, "lon": 8.0}
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

        # Main colored markers
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
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Thermal top (m)",
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
                    f"{name} | thermal top: {ht} m"
                    for ht, name in zip(thermal_top, names)
                ],
                hoverinfo="text",
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
    [0.0, "rgb(255,255,255)"],  # zero diff – white (matches background)
    [0.01, "rgb(255,255,255)"],  # tiny buffer to keep near-zero white
    [0.02, "rgb(255,255,210)"],  # barely above threshold – visible tint
    [0.08, "rgb(255,255,150)"],  # weak thermal (~0.5°C) – light yellow
    [0.17, "rgb(255,245,80)"],  # moderate (~1°C) – yellow
    [0.33, "rgb(255,220,50)"],  # good (~2°C) – golden
    [0.50, "rgb(255,180,30)"],  # strong (~3°C) – orange-yellow
    [0.70, "rgb(255,140,0)"],  # very strong (~4°C) – orange
    [1.0, "rgb(255,80,0)"],  # extreme (~6°C) – deep orange
]


def build_airgram_figure(
    target_name: str,
    selected_date: dt.date,
    altitude_max: int,
    yr_entries: Optional[list[dict]] = None,
    selected_hour: Optional[int] = None,
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
    # Zero out thermal_temp_diff above the computed thermal_top so the
    # heatmap boundary matches the thermal top line exactly.
    plot_frame = plot_frame.with_columns(
        pl.when(pl.col("altitude") > pl.col("thermal_top"))
        .then(0.0)
        .otherwise(pl.col("thermal_temp_diff"))
        .alias("thermal_temp_diff")
    )

    # Pivot thermal data into a 2D grid for the heatmap
    thermal_pivot = plot_frame.pivot(
        on="time_label", index="altitude", values="thermal_temp_diff"
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
            zmax=6,
            zsmooth=False,
            showscale=False,
            hovertemplate=(
                "Alt: %{y:.0f}m<br>Time: %{x}<br>"
                "Thermal diff: %{z:.1f}°C<extra></extra>"
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
        thermal_diffs = plot_frame_wind["thermal_temp_diff"].to_numpy()

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
                    f"Alt: {alt:.0f}m | {spd:.0f} m/s | {angle:.0f}° | Thermal diff: {td:.1f}°C"
                    for alt, spd, angle, td in zip(
                        wind_alts, wind_spds, wind_dirs, thermal_diffs
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
