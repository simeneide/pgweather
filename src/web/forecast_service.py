from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, to_hex
from plotly.subplots import make_subplots

import db_utils


DATA_TTL_SECONDS = 600
_CACHE: dict[str, object] = {"loaded_at": None, "df": None}


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

    new_timestamps = location_data.select("time").to_series().unique().sort().to_list()
    altitudes = np.arange(0.0, float(altitude_max) + 200.0, 200)
    altitudes = altitudes[altitudes >= float(location_data["altitude"].min())]

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

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        subplot_titles=(
            "Wind Speed and Direction [m/s]",
            "Thermal Temperature Difference [deg C]",
        ),
    )

    wind_altitudes = np.arange(200.0, float(altitude_max) + 200.0, 400)
    plot_frame_wind = plot_frame.sort("time", "altitude").filter(
        pl.col("altitude").is_in(wind_altitudes)
    )

    fig.add_trace(
        go.Scatter(
            x=plot_frame_wind.select("time").to_series().to_list(),
            y=plot_frame_wind.select("altitude").to_numpy().squeeze(),
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=18,
                angle=plot_frame_wind.select("wind_direction").to_numpy().squeeze(),
                color=[
                    _interpolate_color(float(s))
                    for s in plot_frame_wind.select("wind_speed").to_numpy().squeeze()
                ],
                showscale=False,
            ),
            hoverinfo="text",
            text=[
                f"Alt: {alt:.0f} m, Speed: {spd:.1f} m/s, Direction: {angle:.0f} deg"
                for alt, spd, angle in zip(
                    plot_frame_wind.select("altitude").to_numpy().squeeze(),
                    plot_frame_wind.select("wind_speed").to_numpy().squeeze(),
                    plot_frame_wind.select("wind_direction").to_numpy().squeeze(),
                )
            ],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=plot_frame.select("thermal_temp_diff").to_numpy().squeeze(),
            x=plot_frame.select("time").to_series().to_list(),
            y=plot_frame.select("altitude").to_numpy().squeeze(),
            colorscale="YlGn",
            showscale=False,
            zmin=0,
            zmax=8,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=760,
        title=(
            f"Airgram for {target_name} on "
            f"{selected_date.strftime('%A')} {selected_date.strftime('%Y-%m-%d')}"
        ),
        yaxis=dict(title="Altitude (m)"),
        xaxis2=dict(title="Time", tickangle=-45),
        yaxis2=dict(title="Altitude (m)", range=[0, altitude_max]),
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
