"""MEPS-specific data loading and preprocessing.

MEPS (MetCoOp Ensemble Prediction System) is a 2.5 km Nordic NWP model
served via THREDDS/OPeNDAP from thredds.met.no.

This module handles:
- THREDDS catalog discovery for latest available model run
- OPeNDAP subsetting and loading
- Hybrid sigma-pressure -> height conversion (hypsometric equation)
- Lambert conformal wind rotation to geographic east/north
- Extraction of forecast timestamp from filename
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Optional

import numpy as np
import xarray as xr
from siphon.catalog import TDSCatalog

logger = logging.getLogger(__name__)


def extract_timestamp(filename: str) -> Optional[str]:
    """Extract ISO timestamp string from a MEPS filename.

    Examples
    --------
    >>> extract_timestamp("meps_det_ml_20260217T06Z.ncml")
    '2026-02-17T06:00Z'
    """
    pattern = r"(\d{4})(\d{2})(\d{2})T(\d{2})Z"
    match = re.search(pattern, filename)
    if match:
        year, month, day, hour = match.groups()
        return f"{year}-{month}-{day}T{hour}:00Z"
    return None


def find_latest_meps_file(
    date: Optional[datetime.date] = None,
    run: Optional[int | str] = None,
) -> str:
    """Discover the latest MEPS model-level file on THREDDS.

    Parameters
    ----------
    date : date, optional
        Date to search. Defaults to today.
    run : int or str, optional
        Specific model run hour (e.g. 6 or "06"). Defaults to latest available.

    Returns
    -------
    str
        OPeNDAP URL for the dataset.
    """
    today = date or datetime.datetime.today()
    catalog_url = (
        f"https://thredds.met.no/thredds/catalog/meps25epsarchive/"
        f"{today.year}/{today.month:02d}/{today.day:02d}/catalog.xml"
    )
    file_url_base = (
        f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/"
        f"{today.year}/{today.month:02d}/{today.day:02d}"
    )
    catalog = TDSCatalog(catalog_url)
    datasets = sorted([s for s in catalog.datasets if "meps_det_ml" in s])

    if run is not None:
        target = f"T{run:02d}Z" if isinstance(run, int) else f"T{run}Z"
        matches = [d for d in datasets if target in d]
        if not matches:
            raise ValueError(
                f"No MEPS file matching run={run} on {today}. Available: {datasets}"
            )
        file_path = f"{file_url_base}/{matches[-1]}"
    else:
        file_path = f"{file_url_base}/{datasets[-1]}"
    return file_path


def _compute_grid_rotation_angle(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> np.ndarray:
    """Compute local grid rotation angle (radians) for a Lambert grid.

    Returns alpha_rad with shape (y, x). To convert grid-relative winds
    to geographic east/north::

        u_east  = x_wind * cos(alpha) - y_wind * sin(alpha)
        v_north = x_wind * sin(alpha) + y_wind * cos(alpha)
    """
    lat = latitude.values
    lon = longitude.values

    dlat_dy = np.gradient(lat, axis=0)
    dlon_dy = np.gradient(lon, axis=0)
    cos_lat = np.cos(np.radians(lat))
    dlon_dy_km = dlon_dy * cos_lat
    dlat_dy_km = dlat_dy

    alpha_rad = np.arctan2(dlat_dy_km, dlon_dy_km) - np.pi / 2
    return alpha_rad


def load_meps_data(
    file_path: Optional[str] = None,
    altitude_min: float = 0,
    altitude_max: float = 4000,
) -> xr.Dataset:
    """Load MEPS model-level data and produce a standardized dataset.

    The returned dataset conforms to the standardized interface defined
    in ``src.preprocessing.base`` — with altitude as the primary vertical
    dimension and all winds rotated to geographic east/north.

    Parameters
    ----------
    file_path : str, optional
        OPeNDAP URL. Discovered automatically if not provided.
    altitude_min, altitude_max : float
        Height AGL range to keep (metres).

    Returns
    -------
    xr.Dataset
        Standardized dataset ready for thermal calculations.
    """
    if file_path is None:
        file_path = find_latest_meps_file()

    logger.info("Loading MEPS data from %s", file_path)

    # OPeNDAP subsetting ranges
    x_range = "[220:1:350]"
    y_range = "[350:1:500]"
    time_range = "[0:1:66]"
    hybrid_range = "[25:1:64]"
    height_range = "[0:1:0]"

    params = {
        "x": x_range,
        "y": y_range,
        "time": time_range,
        "hybrid": hybrid_range,
        "height": height_range,
        "longitude": f"{y_range}{x_range}",
        "latitude": f"{y_range}{x_range}",
        "air_temperature_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
        "ap": f"{hybrid_range}",
        "b": f"{hybrid_range}",
        "surface_air_pressure": f"{time_range}{height_range}{y_range}{x_range}",
        "x_wind_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
        "y_wind_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
    }

    path = f"{file_path}?{','.join(f'{k}{v}' for k, v in params.items())}"
    subset = xr.open_dataset(path, cache=True)
    subset.load()

    # --- Surface parameters ---
    time_range_sfc = "[0:1:0]"
    surf_params = {
        "x": x_range,
        "y": y_range,
        "time": f"{time_range}",
        "surface_geopotential": f"{time_range_sfc}[0:1:0]{y_range}{x_range}",
        "air_temperature_0m": f"{time_range}[0:1:0]{y_range}{x_range}",
        "SFX_DSN_T_ISBA": f"{time_range}{y_range}{x_range}",
    }
    file_path_surf = (
        f"{file_path.replace('meps_det_ml', 'meps_det_sfc')}?"
        f"{','.join(f'{k}{v}' for k, v in surf_params.items())}"
    )

    surf = xr.open_dataset(file_path_surf, cache=True)
    elevation = (surf.surface_geopotential / 9.80665).squeeze()
    subset["elevation"] = elevation
    subset["air_temperature_0m"] = surf.air_temperature_0m.squeeze()
    subset["snow_depth"] = surf["SFX_DSN_T_ISBA"]

    # --- Hybrid sigma-pressure -> height (hypsometric equation) ---
    def hybrid_to_height(ds: xr.Dataset) -> xr.DataArray:
        R = 287.05  # Gas constant for dry air
        g = 9.80665
        p = ds["ap"] + ds["b"] * ds["surface_air_pressure"]
        T = ds["air_temperature_ml"]
        dT_mean = 0.5 * (T + T.isel(hybrid=-1))
        dz = (R * dT_mean / g) * np.log(ds["surface_air_pressure"] / p)
        return dz

    height_agl_3d = hybrid_to_height(subset).mean("time").squeeze()
    subset["height_agl"] = height_agl_3d

    # Domain-averaged altitude as 1D coordinate
    altitude_1d = height_agl_3d.mean("x").mean("y")
    subset = subset.assign_coords(altitude=("hybrid", altitude_1d.data))
    subset = subset.swap_dims({"hybrid": "altitude"})

    # Filter altitude range
    subset = subset.where(
        (subset.altitude >= altitude_min) & (subset.altitude <= altitude_max),
        drop=True,
    ).squeeze()

    # --- Lambert wind rotation ---
    alpha_rad = _compute_grid_rotation_angle(subset["latitude"], subset["longitude"])
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    x_wind_geo = subset["x_wind_ml"] * cos_alpha - subset["y_wind_ml"] * sin_alpha
    y_wind_geo = subset["x_wind_ml"] * sin_alpha + subset["y_wind_ml"] * cos_alpha
    subset["x_wind_ml"] = x_wind_geo
    subset["y_wind_ml"] = y_wind_geo

    # Wind speed
    wind_speed = np.sqrt(subset["x_wind_ml"] ** 2 + subset["y_wind_ml"] ** 2)
    subset = subset.assign(wind_speed=(("time", "altitude", "y", "x"), wind_speed.data))

    logger.info(
        "MEPS data loaded: %d times, %d altitudes, %dx%d grid",
        len(subset.time),
        len(subset.altitude),
        len(subset.y),
        len(subset.x),
    )

    return subset
