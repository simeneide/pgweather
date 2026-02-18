"""Shared preprocessing logic for all weather models.

This module contains model-independent calculations:
- Solar elevation and scaling
- Thermal buoyancy and velocity
- Thermal top detection
- Altitude interpolation
- Spatial aggregation (takeoff matching, kommune matching)
- Gridded output subsampling
- Database writing

Each model-specific module (meps.py, icon.py) produces a standardized
xarray.Dataset, then calls functions here for thermal calculation and
DB output.

Standardized Dataset interface
------------------------------
Each model loader must produce an ``xr.Dataset`` with:

Variables:
    air_temperature_ml : (time, altitude, y, x) — Kelvin
    air_temperature_0m : (time, y, x) — Kelvin, surface parcel temp
    x_wind_ml          : (time, altitude, y, x) — m/s, geographic east
    y_wind_ml          : (time, altitude, y, x) — m/s, geographic north
    wind_speed         : (time, altitude, y, x) — m/s
    height_agl         : (altitude, y, x) — metres above ground level
    elevation          : (y, x) — metres ASL
    snow_depth         : (time, y, x) — metres (or NaN)

Coordinates:
    latitude  : (y, x) — WGS84
    longitude : (y, x) — WGS84
    altitude  : primary vertical dim (metres AGL, domain-averaged)
    time      : forecast valid times (datetime64)

Attributes:
    forecast_timestamp : str — model init time, e.g. "2026-02-17T06:00Z"
    model_source       : str — e.g. "meps", "icon-eu", "icon-global"
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import polars as pl
import xarray as xr
from shapely.geometry import Point

import db_utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Empirical entrainment/drag factor for thermal velocity calculation.
ENTRAINMENT_FACTOR = 0.4

# Solar elevation thresholds for thermal scaling (degrees).
SOLAR_ELEV_MIN_DEG = 5.0
SOLAR_ELEV_FULL_DEG = 15.0

# Threshold for usable thermal top (deg C excess over environment).
THERMAL_TOP_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Solar calculations
# ---------------------------------------------------------------------------


def solar_elevation(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    utc_time: datetime.datetime | list[datetime.datetime] | np.ndarray,
) -> np.ndarray:
    """Compute solar elevation angle (degrees) for arrays of lat/lon and time.

    Uses the standard astronomical approximation (accurate to ~0.5 deg) with
    no external dependencies.
    """
    if isinstance(utc_time, (list, np.ndarray)):
        doy = np.array([t.timetuple().tm_yday for t in utc_time], dtype=float)
        hour_utc = np.array(
            [t.hour + t.minute / 60.0 + t.second / 3600.0 for t in utc_time],
            dtype=float,
        )
    else:
        doy = float(utc_time.timetuple().tm_yday)
        hour_utc = utc_time.hour + utc_time.minute / 60.0 + utc_time.second / 3600.0

    gamma = 2.0 * np.pi * (doy - 1) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.04089 * np.sin(2 * gamma)
    )

    lon_rad = np.radians(np.asarray(lon_deg, dtype=float))
    solar_time = hour_utc * 60.0 + eqtime + np.degrees(lon_rad) * 4.0
    hour_angle = np.radians((solar_time / 4.0) - 180.0)

    lat_rad = np.radians(np.asarray(lat_deg, dtype=float))
    sin_elev = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(
        hour_angle
    )
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


def solar_scaling_factor(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    utc_times: list[datetime.datetime],
) -> np.ndarray:
    """Return a scaling array (0-1) for each (time, y, x) based on solar elevation."""
    n_times = len(utc_times)
    ny, nx = np.asarray(lat_deg).shape
    scale = np.empty((n_times, ny, nx), dtype=float)
    for i, t in enumerate(utc_times):
        elev = solar_elevation(lat_deg, lon_deg, t)
        s = (elev - SOLAR_ELEV_MIN_DEG) / (SOLAR_ELEV_FULL_DEG - SOLAR_ELEV_MIN_DEG)
        scale[i] = np.clip(s, 0.0, 1.0)
    return scale


# ---------------------------------------------------------------------------
# Thermal calculations
# ---------------------------------------------------------------------------


def compute_thermal_temp_difference(subset: xr.Dataset) -> xr.DataArray:
    """Compute buoyancy excess of a dry-adiabatic parcel rising from surface."""
    lapse_rate = 0.0098  # dry adiabatic lapse rate, K/m
    ground_temp = subset.air_temperature_0m - 273.15
    air_temp = subset["air_temperature_ml"] - 273.15
    height_above_ground = subset["height_agl"]
    temp_decrease = lapse_rate * height_above_ground
    ground_parcel_temp = ground_temp - temp_decrease
    thermal_temp_diff = (ground_parcel_temp - air_temp).clip(min=0)
    return thermal_temp_diff


def compute_thermal_velocity(
    subset: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute thermal updraft speed (m/s) at each altitude level.

    Returns (thermal_velocity, thermal_temp_diff).
    """
    g = 9.80665

    thermal_temp_diff = compute_thermal_temp_difference(subset)
    T_env_K = subset["air_temperature_ml"]
    buoyancy = g * thermal_temp_diff / T_env_K

    height_agl = subset["height_agl"]
    original_alt_order = buoyancy.coords["altitude"].values.copy()

    buoyancy = buoyancy.sortby("altitude")
    height_agl = height_agl.sortby("altitude")

    buoyancy_mid = 0.5 * (buoyancy + buoyancy.shift(altitude=1))
    dz = height_agl.diff("altitude")

    b_trimmed = buoyancy_mid.isel(altitude=slice(1, None))
    dz_aligned = dz.assign_coords(altitude=b_trimmed.coords["altitude"])
    integrand_da = b_trimmed * dz_aligned
    alt_axis = list(integrand_da.dims).index("altitude")

    cumulative_work = np.cumsum(integrand_da.values, axis=alt_axis)

    zero_shape = list(cumulative_work.shape)
    zero_shape[alt_axis] = 1
    cumulative_work = np.concatenate(
        [np.zeros(zero_shape), cumulative_work], axis=alt_axis
    )

    thermal_velocity = ENTRAINMENT_FACTOR * np.sqrt(
        2.0 * np.clip(cumulative_work, 0, None)
    )

    thermal_velocity_da = xr.DataArray(
        thermal_velocity,
        dims=buoyancy.dims,
        coords=buoyancy.coords,
    )

    thermal_velocity_da = thermal_velocity_da.reindex(altitude=original_alt_order)
    thermal_temp_diff = thermal_temp_diff.reindex(altitude=original_alt_order)

    return thermal_velocity_da, thermal_temp_diff


def compute_thermal_top(
    subset: xr.Dataset,
    thermal_temp_diff: xr.DataArray,
) -> xr.DataArray:
    """Find thermal top: highest altitude where thermal excess exceeds threshold."""
    usable_diff = thermal_temp_diff.where(
        thermal_temp_diff >= THERMAL_TOP_THRESHOLD, 0.0
    )
    usable_diff = usable_diff.where(
        (usable_diff.sum("altitude") > 0)
        | (subset["altitude"] != subset.altitude.min()),
        usable_diff + 1e-6,
    )
    indices = (usable_diff > 0).argmax(dim="altitude")
    thermal_top = subset["height_agl"].isel(altitude=indices)
    return thermal_top


def apply_solar_scaling(
    subset: xr.Dataset,
    thermal_velocity_da: xr.DataArray,
    thermal_temp_diff_da: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply solar elevation scaling to suppress thermals at night."""
    import pandas as pd

    utc_times = pd.to_datetime(subset.time.values).to_pydatetime().tolist()
    lat_vals = subset["latitude"].values
    lon_vals = subset["longitude"].values
    sun_scale = solar_scaling_factor(lat_vals, lon_vals, utc_times)

    sun_scale_da = xr.DataArray(
        sun_scale,
        dims=("time", "y", "x"),
        coords={"time": subset.time},
    )
    thermal_velocity_da = thermal_velocity_da * sun_scale_da
    thermal_temp_diff_da = thermal_temp_diff_da * sun_scale_da

    return thermal_velocity_da, thermal_temp_diff_da


def add_thermals_to_dataset(subset: xr.Dataset) -> xr.Dataset:
    """Compute thermals, apply solar scaling, and add to dataset.

    This is the main entry point for thermal calculation. It modifies
    the dataset in-place by adding thermal_velocity, thermal_top, and
    setting latitude/longitude as coordinates.

    Parameters
    ----------
    subset : xr.Dataset
        Must conform to the standardized interface (see module docstring).

    Returns
    -------
    xr.Dataset
        The input dataset with thermal_velocity and thermal_top added.
    """
    thermal_velocity_da, thermal_temp_diff_da = compute_thermal_velocity(subset)

    thermal_velocity_da, thermal_temp_diff_da = apply_solar_scaling(
        subset, thermal_velocity_da, thermal_temp_diff_da
    )

    subset["thermal_velocity"] = thermal_velocity_da

    thermal_top = compute_thermal_top(subset, thermal_temp_diff_da)
    subset = subset.assign(thermal_top=(("time", "y", "x"), thermal_top.data))
    subset = subset.set_coords(["latitude", "longitude"])
    return subset


# ---------------------------------------------------------------------------
# Spatial aggregation
# ---------------------------------------------------------------------------


def subsample_lat_lon(
    dataset: xr.Dataset,
    lat_stride: int = 2,
    lon_stride: int = 2,
) -> xr.Dataset:
    """Subsample the y/x dimensions of a dataset."""
    if "y" not in dataset.dims or "x" not in dataset.dims:
        raise ValueError("Dataset does not contain 'y' and 'x' dimensions.")
    return dataset.isel(
        y=slice(None, None, lat_stride), x=slice(None, None, lon_stride)
    )


def match_takeoffs_to_grid(
    df: pl.DataFrame,
    takeoffs_gdf: gpd.GeoDataFrame,
    areas_gdf: Optional[gpd.GeoDataFrame] = None,
) -> pl.DataFrame:
    """Match forecast grid points to takeoffs and optionally to areas.

    Parameters
    ----------
    df : pl.DataFrame
        Flat forecast DataFrame with longitude, latitude columns.
    takeoffs_gdf : gpd.GeoDataFrame
        Takeoff locations from ParaglidingEarth or similar.
    areas_gdf : gpd.GeoDataFrame, optional
        Polygon GeoDataFrame with 'name' column for area aggregation
        (e.g., Norwegian municipalities). If None, area forecasts are skipped.

    Returns
    -------
    pl.DataFrame
        Combined takeoff + area forecasts with 'name' and 'point_type' columns.
    """
    unique_lat_lon = df.select("longitude", "latitude").unique().to_pandas()

    points_forecast = gpd.GeoDataFrame(
        unique_lat_lon,
        geometry=[
            Point(xy)
            for xy in zip(unique_lat_lon["longitude"], unique_lat_lon["latitude"])
        ],
    )

    # Determine CRS from areas if available, otherwise use WGS84
    crs = areas_gdf.crs if areas_gdf is not None else "EPSG:4326"
    points_forecast.set_crs(crs, inplace=True)

    frames: list[pl.DataFrame] = []

    # --- Area forecasts (optional) ---
    if areas_gdf is not None:
        named_lat_lon = gpd.sjoin(
            points_forecast, areas_gdf, how="left", predicate="within"
        )
        df_names = pl.DataFrame(
            named_lat_lon[["longitude", "latitude", "name"]]
        ).drop_nulls()

        area_forecasts = (
            df.join(df_names, on=["longitude", "latitude"], how="inner")
            .group_by("forecast_timestamp", "time", "name", "altitude")
            .median()
            .with_columns(point_type=pl.lit("area"))
        )
        frames.append(area_forecasts)

    # --- Takeoff forecasts ---
    takeoffs_gdf = takeoffs_gdf.copy()
    takeoffs_gdf["latitude_takeoff"] = takeoffs_gdf.geometry.y
    takeoffs_gdf["longitude_takeoff"] = takeoffs_gdf.geometry.x
    takeoffs_gdf.set_crs(crs, inplace=True)

    takeoffs = gpd.sjoin_nearest(
        takeoffs_gdf, points_forecast, how="left", max_distance=10000
    )[["name", "longitude_takeoff", "latitude_takeoff", "longitude", "latitude"]]
    takeoffs = takeoffs.drop_duplicates(subset="name")
    df_takeoffs = pl.DataFrame(takeoffs)

    takeoff_forecasts = (
        df.join(df_takeoffs, on=["longitude", "latitude"], how="inner")
        .select(pl.exclude("longitude", "latitude"))
        .rename({"longitude_takeoff": "longitude", "latitude_takeoff": "latitude"})
        .with_columns(point_type=pl.lit("takeoff"))
    )
    frames.append(takeoff_forecasts)

    # Combine — ensure same columns across all frames
    result_cols = takeoff_forecasts.columns
    combined = pl.concat(
        [f.select(result_cols) for f in frames],
        how="vertical_relaxed",
    )
    return combined


# ---------------------------------------------------------------------------
# Altitude interpolation
# ---------------------------------------------------------------------------


def interpolate_altitudes(
    subset: xr.Dataset,
    altitude_max: float = 4000.0,
) -> xr.Dataset:
    """Interpolate dataset to standard altitude bins.

    Uses 100m bins below 600m and 200m bins above.
    """
    below_600 = np.arange(0, 600, 100)
    above_600 = np.arange(
        600, min(subset.altitude.max().item(), altitude_max) + 200, 200
    )
    altitude_intervals = np.concatenate([below_600, above_600])
    return subset.interp(altitude=altitude_intervals, method="linear")


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


def dataset_to_flat_dataframe(
    subset: xr.Dataset,
    forecast_timestamp: str,
) -> pl.DataFrame:
    """Convert an altitude-interpolated xarray Dataset to a flat Polars DataFrame.

    Filters out rows where elevation > altitude and adds
    thermal_height_above_ground.
    """
    df = (
        pl.DataFrame(subset.to_dataframe().reset_index())
        .with_columns(forecast_timestamp=pl.lit(forecast_timestamp).cast(pl.Datetime))
        .filter(pl.col("elevation") <= pl.col("altitude"))
        .with_columns(
            thermal_height_above_ground=pl.col("altitude") - pl.col("elevation")
        )
        .select(
            "forecast_timestamp",
            "time",
            "elevation",
            "altitude",
            "air_temperature_ml",
            "x_wind_ml",
            "y_wind_ml",
            "longitude",
            "latitude",
            "wind_speed",
            "thermal_velocity",
            "thermal_top",
            "thermal_height_above_ground",
            "snow_depth",
        )
    )
    return df


# ---------------------------------------------------------------------------
# Gridded output
# ---------------------------------------------------------------------------


def build_gridded_dataframe(
    subset: xr.Dataset,
    forecast_timestamp: str,
    model_source: str,
    grid_stride: int = 3,
    grid_altitudes: Optional[list[int]] = None,
) -> pl.DataFrame:
    """Build the gridded forecast DataFrame from a full-resolution dataset.

    Parameters
    ----------
    subset : xr.Dataset
        Full-resolution dataset with thermals already computed.
    forecast_timestamp : str
        Model init time string.
    model_source : str
        Model identifier (e.g. "meps", "icon-eu").
    grid_stride : int
        Subsampling stride (every Nth grid point).
    grid_altitudes : list[int], optional
        Altitude levels for grid output. Defaults to [0, 500, 1000, 1500, 2000, 3000].

    Returns
    -------
    pl.DataFrame
    """
    if grid_altitudes is None:
        grid_altitudes = [0, 500, 1000, 1500, 2000, 3000]

    gridded_subset = subsample_lat_lon(
        subset, lat_stride=grid_stride, lon_stride=grid_stride
    )
    gridded_interp = gridded_subset.interp(altitude=grid_altitudes, method="linear")

    # Back-fill altitude=0 with lowest model level values
    for var in ["x_wind_ml", "y_wind_ml", "wind_speed"]:
        if var in gridded_interp:
            gridded_interp[var] = gridded_interp[var].bfill(dim="altitude")

    grid_df = (
        pl.DataFrame(gridded_interp.to_dataframe().reset_index())
        .with_columns(
            forecast_timestamp=pl.lit(forecast_timestamp).cast(pl.Datetime),
            model_source=pl.lit(model_source),
        )
        .select(
            "forecast_timestamp",
            "time",
            "latitude",
            "longitude",
            "altitude",
            "elevation",
            "x_wind_ml",
            "y_wind_ml",
            "wind_speed",
            "thermal_velocity",
            "thermal_top",
            "snow_depth",
            "model_source",
        )
    )
    return grid_df


# ---------------------------------------------------------------------------
# Database writing
# ---------------------------------------------------------------------------


def save_forecasts_to_db(
    point_forecasts: pl.DataFrame,
    grid_df: pl.DataFrame,
    forecast_timestamp: str,
    model_source: str,
    db: Optional[db_utils.Database] = None,
) -> None:
    """Write point and gridded forecasts to the database.

    Parameters
    ----------
    point_forecasts : pl.DataFrame
        Combined takeoff + area forecasts with model_source column.
    grid_df : pl.DataFrame
        Gridded forecast DataFrame with model_source column.
    forecast_timestamp : str
        Model init time string for replace_forecast().
    model_source : str
        Model identifier.
    db : Database, optional
        Database instance. Creates one if not provided.
    """
    if db is None:
        db = db_utils.Database()

    # Ensure all float columns are Float64 for Postgres double precision compat
    # (GRIB sources like ICON produce float32 which ADBC can't COPY to float8)
    float_cols = [
        c
        for c in point_forecasts.columns
        if point_forecasts[c].dtype in (pl.Float32, pl.Float64)
    ]
    if float_cols:
        point_forecasts = point_forecasts.with_columns(
            [pl.col(c).cast(pl.Float64) for c in float_cols]
        )

    logger.info("Saving %d point forecasts to db...", len(point_forecasts))
    db.replace_forecast(
        point_forecasts,
        "detailed_forecasts",
        forecast_timestamp,
        model_source=model_source,
    )
    logger.info("Saved %d point forecasts.", len(point_forecasts))

    # Same float64 cast for gridded data
    grid_float_cols = [
        c for c in grid_df.columns if grid_df[c].dtype in (pl.Float32, pl.Float64)
    ]
    if grid_float_cols:
        grid_df = grid_df.with_columns(
            [pl.col(c).cast(pl.Float64) for c in grid_float_cols]
        )

    logger.info("Saving %d gridded forecast rows to db...", len(grid_df))
    db.replace_data(grid_df, "gridded_forecasts", model_source=model_source)

    # Ensure indexes exist
    try:
        db.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_detailed_model_forecast "
            "ON detailed_forecasts (model_source, forecast_timestamp DESC)"
        )
        db.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_detailed_model_forecast_time "
            "ON detailed_forecasts (model_source, forecast_timestamp, time)"
        )
        db.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_detailed_map_lookup "
            "ON detailed_forecasts (model_source, forecast_timestamp, time, name, altitude)"
        )
        db.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_detailed_windgram_lookup "
            "ON detailed_forecasts (model_source, forecast_timestamp, name, point_type, time, altitude)"
        )
        db.execute_query(
            "CREATE INDEX IF NOT EXISTS idx_gridded_time_alt "
            "ON gridded_forecasts (model_source, forecast_timestamp, time, altitude)"
        )
    except Exception:
        logger.exception("Could not ensure forecast indexes")
    logger.info("Gridded forecast saved.")


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_post_loading_pipeline(
    subset: xr.Dataset,
    forecast_timestamp: str,
    model_source: str,
    takeoffs_gdf: gpd.GeoDataFrame,
    areas_gdf: Optional[gpd.GeoDataFrame] = None,
    db: Optional[db_utils.Database] = None,
) -> None:
    """Run the full post-loading pipeline: thermals, aggregation, DB write.

    This is the main orchestrator that model-specific loaders call after
    producing a standardized dataset.

    Parameters
    ----------
    subset : xr.Dataset
        Standardized dataset from a model loader.
    forecast_timestamp : str
        Model init time, e.g. "2026-02-17T06:00Z".
    model_source : str
        E.g. "meps", "icon-eu", "icon-global".
    takeoffs_gdf : gpd.GeoDataFrame
        Takeoff locations.
    areas_gdf : gpd.GeoDataFrame, optional
        Area polygons for spatial aggregation (e.g. Norwegian municipalities).
    db : Database, optional
        Database instance.
    """
    if db is None:
        db = db_utils.Database()

    # 1. Compute thermals
    logger.info("Computing thermals for %s...", model_source)
    subset = add_thermals_to_dataset(subset)

    # 2. Interpolate to standard altitude bins
    altitude_interpolated = interpolate_altitudes(subset)

    # 3. Convert to flat DataFrame
    df = dataset_to_flat_dataframe(altitude_interpolated, forecast_timestamp)

    # 4. Match to takeoffs (and optionally areas)
    logger.info("Matching to takeoffs...")
    point_forecasts = match_takeoffs_to_grid(df, takeoffs_gdf, areas_gdf=areas_gdf)
    point_forecasts = point_forecasts.with_columns(model_source=pl.lit(model_source))

    # 5. Build gridded output
    logger.info("Building gridded output...")
    grid_df = build_gridded_dataframe(subset, forecast_timestamp, model_source)

    # 6. Save to database
    save_forecasts_to_db(
        point_forecasts, grid_df, forecast_timestamp, model_source, db=db
    )

    logger.info("Pipeline complete for %s / %s", model_source, forecast_timestamp)
