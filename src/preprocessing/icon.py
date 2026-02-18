"""ICON model data loading (ICON-EU and ICON Global).

Downloads GRIB2 model-level data from DWD opendata, converts to the
standardized xarray.Dataset interface for thermal calculations.

ICON-EU: 0.0625 deg (~7 km), regular lat/lon, 74 model levels, Europe
ICON Global: ~13 km, icosahedral (regular-lat-lon output available), 90 levels

Data source: https://opendata.dwd.de/weather/nwp/

File naming:
  model-level: icon-eu_europe_regular-lat-lon_model-level_{YYYYMMDDHH}_{FFF}_{LEVEL}_{PARAM}.grib2.bz2
  single-level: icon-eu_europe_regular-lat-lon_single-level_{YYYYMMDDHH}_{FFF}_{PARAM}.grib2.bz2
  time-invariant: icon-eu_europe_regular-lat-lon_time-invariant_{YYYYMMDDHH}_{LEVEL}_HHL.grib2.bz2

ICON-EU runs: 00, 03, 06, 09, 12, 15, 18, 21 UTC
ICON Global runs: 00, 06, 12, 18 UTC
Timesteps: hourly to 78h, then 3-hourly to 120h
"""

from __future__ import annotations

import bz2
import datetime
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cfgrib
import numpy as np
import requests
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DWD opendata base URLs
# ---------------------------------------------------------------------------
DWD_BASE_ICON_EU = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
DWD_BASE_ICON_GLOBAL = "https://opendata.dwd.de/weather/nwp/icon/grib"

# ICON-EU has 74 model levels + 75 half-levels (HHL).
# Bottom levels (~50-74) cover 0-4000m AGL. We download from level 40
# to be safe for high-elevation terrain.
ICON_EU_BOTTOM_LEVELS = list(range(40, 75))  # levels 40-74
ICON_EU_HHL_LEVELS = list(range(40, 76))  # HHL has one more level (half-levels)

# Download settings
MAX_WORKERS = 8
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Discovery: find latest available model run
# ---------------------------------------------------------------------------


def find_latest_icon_eu_run(
    date: Optional[datetime.date] = None,
    run: Optional[int] = None,
) -> tuple[str, datetime.datetime]:
    """Find the latest available ICON-EU run on DWD opendata.

    Parameters
    ----------
    date : date, optional
        Date to check. Defaults to today UTC.
    run : int, optional
        Specific run hour (0, 3, 6, 9, 12, 15, 18, 21).

    Returns
    -------
    (run_id, init_time) : tuple[str, datetime.datetime]
        run_id is e.g. "2026021806", init_time is a UTC datetime.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    if date is None:
        date = now.date()

    if run is not None:
        runs_to_try = [run]
    else:
        # Try runs in reverse order, starting from the most recent possible
        all_runs = [0, 3, 6, 9, 12, 15, 18, 21]
        # Allow ~3 hours for data to become available
        latest_possible = now.hour - 3
        runs_to_try = [r for r in reversed(all_runs) if r <= latest_possible]
        if not runs_to_try:
            # Try yesterday's late runs
            date = date - datetime.timedelta(days=1)
            runs_to_try = list(reversed(all_runs))

    for r in runs_to_try:
        run_id = f"{date:%Y%m%d}{r:02d}"
        # Check if the run directory has data by probing one file
        url = (
            f"{DWD_BASE_ICON_EU}/{r:02d}/t/"
            f"icon-eu_europe_regular-lat-lon_model-level_{run_id}_000_74_T.grib2.bz2"
        )
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                init_time = datetime.datetime(
                    date.year,
                    date.month,
                    date.day,
                    r,
                    tzinfo=datetime.timezone.utc,
                )
                logger.info("Found ICON-EU run: %s", run_id)
                return run_id, init_time
        except requests.RequestException:
            continue

    raise RuntimeError(
        f"No ICON-EU run found for date={date}, tried runs={runs_to_try}"
    )


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_and_decompress(url: str, allow_missing: bool = False) -> bytes | None:
    """Download a bz2-compressed GRIB2 file and return decompressed bytes.

    If *allow_missing* is True, returns None for 404 responses instead of raising.
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404 and allow_missing:
                return None
            resp.raise_for_status()
            return bz2.decompress(resp.content)
        except requests.exceptions.HTTPError as exc:
            if (
                allow_missing
                and exc.response is not None
                and exc.response.status_code == 404
            ):
                return None
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning("Retry %d for %s", attempt + 1, url.split("/")[-1])
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning("Retry %d for %s", attempt + 1, url.split("/")[-1])
    return b""  # unreachable


def _download_grib_files(
    urls: list[str],
    max_workers: int = MAX_WORKERS,
    allow_missing: bool = False,
) -> list[bytes]:
    """Download multiple GRIB2 files in parallel, return decompressed bytes.

    If *allow_missing* is True, 404 responses are silently skipped and
    only successfully downloaded files are returned.
    """
    results: dict[int, bytes | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_and_decompress, url, allow_missing): i
            for i, url in enumerate(urls)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                if allow_missing:
                    logger.warning(
                        "Skipping failed download: %s", urls[idx].split("/")[-1]
                    )
                    results[idx] = None
                else:
                    logger.exception("Failed to download %s", urls[idx].split("/")[-1])
                    raise

    # Filter out None (missing files) and return in order
    out: list[bytes] = []
    for i in range(len(urls)):
        val = results.get(i)
        if val is not None:
            out.append(val)
    return out


def _grib_bytes_to_dataset(grib_bytes_list: list[bytes]) -> xr.Dataset:
    """Convert a list of GRIB2 byte blobs into a single xarray Dataset.

    Writes to a temporary file because cfgrib requires a file path.
    Each GRIB file may contain data for a single level/timestep; cfgrib
    will assign scalar coordinates for those dimensions.  We expand them
    to real dimensions before merging so that xr.merge stacks them into
    a multi-dimensional dataset.
    """
    datasets = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, data in enumerate(grib_bytes_list):
            fpath = Path(tmpdir) / f"data_{i:04d}.grib2"
            fpath.write_bytes(data)
            try:
                ds_list = cfgrib.open_datasets(str(fpath))
                for ds in ds_list:
                    # Promote scalar coordinates to dimensions so merging
                    # stacks along them instead of overwriting.
                    expand_dims: list[str] = []
                    for coord_name in [
                        "generalVerticalLayer",
                        "generalVertical",
                        "level",
                        "hybrid",
                        "step",
                    ]:
                        if coord_name in ds.coords and coord_name not in ds.dims:
                            expand_dims.append(coord_name)
                    if expand_dims:
                        ds = ds.expand_dims(expand_dims)
                    datasets.append(ds)
            except Exception:
                logger.warning("Failed to parse GRIB file %d", i)
                continue

    if not datasets:
        raise RuntimeError("No GRIB data could be parsed")

    # Merge all datasets — cfgrib may split by level type
    merged = xr.combine_by_coords(datasets, combine_attrs="override")
    return merged


# ---------------------------------------------------------------------------
# ICON-EU data loading
# ---------------------------------------------------------------------------


def _build_icon_eu_urls(
    run_id: str,
    run_hour: int,
    param: str,
    levels: list[int],
    timesteps: list[int],
    level_type: str = "model-level",
) -> list[str]:
    """Build download URLs for ICON-EU model-level or single-level data."""
    urls = []
    base = f"{DWD_BASE_ICON_EU}/{run_hour:02d}/{param.lower()}"

    if level_type == "model-level":
        for step in timesteps:
            for level in levels:
                fname = (
                    f"icon-eu_europe_regular-lat-lon_model-level_"
                    f"{run_id}_{step:03d}_{level}_{param}.grib2.bz2"
                )
                urls.append(f"{base}/{fname}")
    elif level_type == "single-level":
        for step in timesteps:
            fname = (
                f"icon-eu_europe_regular-lat-lon_single-level_"
                f"{run_id}_{step:03d}_{param}.grib2.bz2"
            )
            urls.append(f"{base}/{fname}")
    elif level_type == "time-invariant":
        for level in levels:
            fname = (
                f"icon-eu_europe_regular-lat-lon_time-invariant_"
                f"{run_id}_{level}_{param}.grib2.bz2"
            )
            urls.append(f"{base}/{fname}")

    return urls


def load_icon_eu_data(
    run_id: Optional[str] = None,
    init_time: Optional[datetime.datetime] = None,
    altitude_min: float = 0,
    altitude_max: float = 4000,
    max_forecast_hours: int = 48,
    lat_bounds: Optional[tuple[float, float]] = None,
    lon_bounds: Optional[tuple[float, float]] = None,
) -> xr.Dataset:
    """Load ICON-EU model-level data and produce a standardized dataset.

    Parameters
    ----------
    run_id : str, optional
        Run identifier like "2026021806". Auto-discovered if not provided.
    init_time : datetime, optional
        Model initialization time. Auto-discovered if not provided.
    altitude_min, altitude_max : float
        Height AGL range to keep (metres).
    max_forecast_hours : int
        Maximum lead time in hours to download.
    lat_bounds : tuple, optional
        (lat_min, lat_max) for spatial subsetting. Defaults to Norway coverage.
    lon_bounds : tuple, optional
        (lon_min, lon_max) for spatial subsetting. Defaults to Norway coverage.

    Returns
    -------
    xr.Dataset
        Standardized dataset ready for thermal calculations.
    """
    if run_id is None or init_time is None:
        run_id, init_time = find_latest_icon_eu_run()

    run_hour = init_time.hour

    # Default bounds: Norway + surrounding area
    if lat_bounds is None:
        lat_bounds = (57.0, 72.0)
    if lon_bounds is None:
        lon_bounds = (3.0, 32.0)

    # Timesteps: hourly to min(78, max_forecast_hours), then 3-hourly
    hourly_steps = list(range(0, min(79, max_forecast_hours + 1)))
    three_hourly_steps = list(range(81, min(121, max_forecast_hours + 1), 3))
    timesteps = hourly_steps + three_hourly_steps
    # Only keep up to max_forecast_hours
    timesteps = [t for t in timesteps if t <= max_forecast_hours]

    logger.info(
        "Loading ICON-EU run=%s, %d timesteps, levels %d-%d",
        run_id,
        len(timesteps),
        ICON_EU_BOTTOM_LEVELS[0],
        ICON_EU_BOTTOM_LEVELS[-1],
    )

    # --- Download HHL (time-invariant height of half-levels) ---
    logger.info("Downloading HHL (height of half-levels)...")
    hhl_urls = _build_icon_eu_urls(
        run_id, run_hour, "HHL", ICON_EU_HHL_LEVELS, [], "time-invariant"
    )
    hhl_bytes = _download_grib_files(hhl_urls, allow_missing=True)
    if not hhl_bytes:
        raise RuntimeError("No HHL data could be downloaded — cannot proceed")
    hhl_ds = _grib_bytes_to_dataset(hhl_bytes)

    # --- Download T (temperature) at model levels ---
    logger.info("Downloading T (temperature) at model levels...")
    t_urls = _build_icon_eu_urls(
        run_id, run_hour, "T", ICON_EU_BOTTOM_LEVELS, timesteps
    )
    t_bytes = _download_grib_files(t_urls, allow_missing=True)
    if not t_bytes:
        raise RuntimeError("No temperature data could be downloaded")
    n_downloaded = len(t_bytes)
    n_requested = len(t_urls)
    if n_downloaded < n_requested:
        logger.warning(
            "T download: got %d/%d files (some timesteps unavailable on DWD server)",
            n_downloaded,
            n_requested,
        )
    t_ds = _grib_bytes_to_dataset(t_bytes)

    # --- Download U, V (wind components) at model levels ---
    logger.info("Downloading U, V (wind) at model levels...")
    u_urls = _build_icon_eu_urls(
        run_id, run_hour, "U", ICON_EU_BOTTOM_LEVELS, timesteps
    )
    v_urls = _build_icon_eu_urls(
        run_id, run_hour, "V", ICON_EU_BOTTOM_LEVELS, timesteps
    )
    u_bytes = _download_grib_files(u_urls, allow_missing=True)
    v_bytes = _download_grib_files(v_urls, allow_missing=True)
    if not u_bytes or not v_bytes:
        raise RuntimeError("No wind data could be downloaded")
    u_ds = _grib_bytes_to_dataset(u_bytes)
    v_ds = _grib_bytes_to_dataset(v_bytes)

    # --- Download T_2M (2m temperature, single-level) ---
    logger.info("Downloading T_2M (2m temperature)...")
    t2m_urls = _build_icon_eu_urls(
        run_id, run_hour, "T_2M", [], timesteps, "single-level"
    )
    t2m_bytes = _download_grib_files(t2m_urls, allow_missing=True)
    if not t2m_bytes:
        raise RuntimeError("No T_2M data could be downloaded")
    t2m_ds = _grib_bytes_to_dataset(t2m_bytes)

    # --- Build standardized dataset ---
    logger.info("Building standardized dataset...")
    subset = _build_standardized_dataset_icon_eu(
        hhl_ds=hhl_ds,
        t_ds=t_ds,
        u_ds=u_ds,
        v_ds=v_ds,
        t2m_ds=t2m_ds,
        init_time=init_time,
        altitude_min=altitude_min,
        altitude_max=altitude_max,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )

    logger.info(
        "ICON-EU data ready: %d times, %d altitudes, %dx%d grid",
        len(subset.time),
        len(subset.altitude),
        len(subset.y),
        len(subset.x),
    )

    return subset


def _build_standardized_dataset_icon_eu(
    hhl_ds: xr.Dataset,
    t_ds: xr.Dataset,
    u_ds: xr.Dataset,
    v_ds: xr.Dataset,
    t2m_ds: xr.Dataset,
    init_time: datetime.datetime,
    altitude_min: float,
    altitude_max: float,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.Dataset:
    """Convert raw ICON-EU GRIB data to the standardized interface.

    ICON GRIB data uses 'generalVerticalLayer' for model levels and
    'latitude'/'longitude' for the regular grid.
    """
    # --- Extract variables ---
    # cfgrib variable names depend on GRIB shortName:
    # T -> 't', U -> 'u', V -> 'v', T_2M -> 't2m', HHL -> 'z' or 'hhl'
    # The exact key may vary; try common options.
    t_var = _find_var(t_ds, ["t", "T", "air_temperature"])
    u_var = _find_var(u_ds, ["u", "U", "10u", "eastward_wind"])
    v_var = _find_var(v_ds, ["v", "V", "10v", "northward_wind"])
    t2m_var = _find_var(t2m_ds, ["t2m", "T_2M", "2t"])
    hhl_var = _find_var(hhl_ds, ["z", "hhl", "HHL", "h"])

    # Identify dimension names from cfgrib output
    # Typical dims: (step, generalVerticalLayer, latitude, longitude) or
    # (valid_time, generalVerticalLayer, latitude, longitude)
    t_data = t_ds[t_var]
    logger.info("T data dims: %s, shape: %s", t_data.dims, t_data.shape)

    # --- Spatial subset ---
    lat_dim = _find_dim(t_data, ["latitude", "lat"])
    lon_dim = _find_dim(t_data, ["longitude", "lon"])

    # Apply spatial subsetting
    lat_vals = t_data[lat_dim].values
    lon_vals = t_data[lon_dim].values

    lat_mask = (lat_vals >= lat_bounds[0]) & (lat_vals <= lat_bounds[1])
    lon_mask = (lon_vals >= lon_bounds[0]) & (lon_vals <= lon_bounds[1])

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise RuntimeError(
            f"No grid points in bounds lat={lat_bounds}, lon={lon_bounds}"
        )

    # Slice all datasets
    lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)

    t_sub = t_ds.isel({lat_dim: lat_slice, lon_dim: lon_slice})
    u_sub = u_ds.isel({lat_dim: lat_slice, lon_dim: lon_slice})
    v_sub = v_ds.isel({lat_dim: lat_slice, lon_dim: lon_slice})
    t2m_sub = t2m_ds.isel({lat_dim: lat_slice, lon_dim: lon_slice})
    hhl_sub = hhl_ds.isel({lat_dim: lat_slice, lon_dim: lon_slice})

    # --- Compute height AGL from HHL ---
    hhl_data = hhl_sub[hhl_var]
    level_dim = _find_dim(
        hhl_data, ["generalVertical", "generalVerticalLayer", "level", "hybrid"]
    )

    # HHL: height of half-levels. Full level height = mean of adjacent half-levels.
    # If HHL has N+1 half-levels, full levels have N values.
    hhl_sorted = hhl_data.sortby(level_dim)
    hhl_vals = hhl_sorted.values
    # Average adjacent half-levels
    if len(hhl_vals.shape) > 1:
        level_axis = list(hhl_sorted.dims).index(level_dim)
        slices_lower = [slice(None)] * len(hhl_vals.shape)
        slices_upper = [slice(None)] * len(hhl_vals.shape)
        slices_lower[level_axis] = slice(0, -1)
        slices_upper[level_axis] = slice(1, None)
        full_level_height = 0.5 * (
            hhl_vals[tuple(slices_lower)] + hhl_vals[tuple(slices_upper)]
        )
    else:
        full_level_height = 0.5 * (hhl_vals[:-1] + hhl_vals[1:])

    # Surface elevation = lowest half-level height (level index = max)
    surface_idx = [slice(None)] * len(hhl_vals.shape)
    level_axis = list(hhl_sorted.dims).index(level_dim)
    surface_idx[level_axis] = -1
    elevation = hhl_vals[tuple(surface_idx)]

    # Height AGL = full level height - surface elevation
    elev_expanded = np.expand_dims(elevation, axis=level_axis)
    height_agl = full_level_height - elev_expanded
    height_agl = np.maximum(height_agl, 0)  # ensure non-negative

    # --- Build time coordinate ---
    t_sub_data = t_sub[t_var]
    time_dim = _find_dim(t_sub_data, ["step", "time", "valid_time", "forecast_time"])

    # Convert step to valid_time
    if "step" in t_sub_data.dims:
        steps = t_sub_data.step.values
        valid_times = np.array(
            [np.datetime64(init_time.replace(tzinfo=None)) + step for step in steps]
        )
    elif "valid_time" in t_sub_data.dims:
        valid_times = t_sub_data.valid_time.values
    else:
        valid_times = t_sub_data.time.values

    # --- Determine level coordinate for T/U/V ---
    t_level_dim = _find_dim(t_sub_data, ["generalVerticalLayer", "level", "hybrid"])
    n_levels = t_sub_data.sizes[t_level_dim]

    # Domain-averaged altitude for 1D coordinate (same approach as MEPS)
    # height_agl shape: (level, lat, lon)
    spatial_dims = [d for d in range(len(height_agl.shape)) if d != level_axis]
    altitude_1d = np.mean(height_agl, axis=tuple(spatial_dims))

    # Filter by altitude range
    alt_mask = (altitude_1d >= altitude_min) & (altitude_1d <= altitude_max)
    if not np.any(alt_mask):
        raise RuntimeError(
            f"No altitude levels in range {altitude_min}-{altitude_max}m"
        )

    # Get lat/lon 2D arrays
    lat_1d = t_sub[lat_dim].values
    lon_1d = t_sub[lon_dim].values
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    # --- Build output dataset ---
    # Select valid altitude levels
    valid_levels = np.where(alt_mask)[0]
    altitude_vals = altitude_1d[valid_levels]

    # Extract data arrays and reindex to common dims: (time, altitude, y, x)
    t_arr = t_sub_data.isel({t_level_dim: valid_levels}).values
    u_arr = u_sub[u_var].isel({t_level_dim: valid_levels}).values
    v_arr = v_sub[v_var].isel({t_level_dim: valid_levels}).values
    t2m_arr = t2m_sub[t2m_var].values
    height_agl_sub = height_agl[valid_levels]

    # Wind speed
    wind_speed = np.sqrt(u_arr**2 + v_arr**2)

    # Build dataset
    n_time = len(valid_times)
    n_alt = len(altitude_vals)
    n_y = len(lat_1d)
    n_x = len(lon_1d)

    # Snow depth: not available from ICON model-level download; fill with NaN
    snow_depth = np.full((n_time, n_y, n_x), np.nan)

    ds = xr.Dataset(
        {
            "air_temperature_ml": (
                ("time", "altitude", "y", "x"),
                t_arr.reshape(n_time, n_alt, n_y, n_x),
            ),
            "x_wind_ml": (
                ("time", "altitude", "y", "x"),
                u_arr.reshape(n_time, n_alt, n_y, n_x),
            ),
            "y_wind_ml": (
                ("time", "altitude", "y", "x"),
                v_arr.reshape(n_time, n_alt, n_y, n_x),
            ),
            "wind_speed": (
                ("time", "altitude", "y", "x"),
                wind_speed.reshape(n_time, n_alt, n_y, n_x),
            ),
            "air_temperature_0m": (
                ("time", "y", "x"),
                t2m_arr.reshape(n_time, n_y, n_x),
            ),
            "height_agl": (
                ("altitude", "y", "x"),
                height_agl_sub.reshape(n_alt, n_y, n_x),
            ),
            "elevation": (
                ("y", "x"),
                elevation.reshape(n_y, n_x),
            ),
            "snow_depth": (
                ("time", "y", "x"),
                snow_depth,
            ),
        },
        coords={
            "time": valid_times,
            "altitude": altitude_vals,
            "latitude": (("y", "x"), lat_2d),
            "longitude": (("y", "x"), lon_2d),
        },
    )

    return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_var(ds: xr.Dataset, candidates: list[str]) -> str:
    """Find the first matching variable name in a dataset."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    available = list(ds.data_vars)
    raise KeyError(f"None of {candidates} found in dataset. Available: {available}")


def _find_dim(da: xr.DataArray | xr.Dataset, candidates: list[str]) -> str:
    """Find the first matching dimension name."""
    dims = list(da.dims)
    for name in candidates:
        if name in dims:
            return name
    raise KeyError(f"None of {candidates} found in dims {dims}")


def format_icon_timestamp(init_time: datetime.datetime) -> str:
    """Format an ICON init time as an ISO timestamp string."""
    return f"{init_time:%Y-%m-%dT%H}:00Z"
