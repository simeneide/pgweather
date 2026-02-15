import xarray as xr
from siphon.catalog import TDSCatalog
import numpy as np
import datetime
import re
import os
from dotenv import load_dotenv
import geopandas as gpd
from shapely.geometry import Point

load_dotenv()
import db_utils
import polars as pl
import takeoff_utils


# Empirical entrainment/drag factor for thermal velocity calculation.
# Real thermals lose energy to mixing with environmental air.  Raw parcel
# theory over-predicts by ~2-3×; k ≈ 0.4 is a reasonable starting point
# from literature (Lenschow 1980, Allen 2006).  Tune against pilot vario data.
ENTRAINMENT_FACTOR = 0.4


# %%
def compute_thermal_temp_difference(subset):
    """Compute the buoyancy excess of a dry-adiabatic parcel rising from the surface.

    Uses per-gridpoint height_agl (height above local ground) instead of the
    domain-averaged altitude coordinate, giving physically correct results in
    mountainous terrain.
    """
    lapse_rate = 0.0098  # dry adiabatic lapse rate, K/m
    ground_temp = subset.air_temperature_0m - 273.15
    air_temp = subset["air_temperature_ml"] - 273.15

    # height_agl is per-gridpoint height above local surface (altitude, y, x).
    # It is always >= 0 by construction (hypsometric equation from surface up).
    height_above_ground = subset["height_agl"]
    temp_decrease = lapse_rate * height_above_ground
    ground_parcel_temp = ground_temp - temp_decrease
    thermal_temp_diff = (ground_parcel_temp - air_temp).clip(min=0)
    return thermal_temp_diff


def compute_thermal_velocity(subset):
    """Compute thermal updraft speed (m/s) at each altitude level.

    Uses buoyancy integration: a dry-adiabatic parcel accelerates upward
    due to temperature excess over the environment.  The cumulative
    velocity is derived from the work-energy theorem::

        B(z) = g × ΔT(z) / T_env(z)           # buoyancy acceleration [m/s²]
        w(z) = k × sqrt(2 × ∫₀ᶻ B(z') dz')    # integrated from ground up

    where k = ENTRAINMENT_FACTOR accounts for drag and entrainment losses
    in real thermals.

    Returns both ``thermal_velocity`` and the intermediate
    ``thermal_temp_diff`` (needed for thermal_top calculation).
    """
    g = 9.80665

    # Step 1: dry-adiabatic parcel buoyancy excess (°C, clipped ≥ 0)
    thermal_temp_diff = compute_thermal_temp_difference(subset)

    # Step 2: buoyancy acceleration at each level  [m/s²]
    # ΔT is in °C (== K difference), T_env is in K → ratio is dimensionless
    T_env_K = subset["air_temperature_ml"]  # Kelvin
    buoyancy = g * thermal_temp_diff / T_env_K

    # Step 3: cumulative upward integral of buoyancy × dz  [m²/s²]
    # Use xarray named-dimension operations so broadcasting is automatic
    # regardless of dimension ordering (which can be (time, y, x, altitude)).
    height_agl = subset["height_agl"]  # time-averaged, (altitude, y, x)

    # Trapezoidal rule: average buoyancy between adjacent levels × dz.
    # shift(altitude=1) shifts values toward higher altitudes, filling the
    # lowest level with NaN.
    buoyancy_mid = 0.5 * (buoyancy + buoyancy.shift(altitude=1))

    # dz between consecutive altitude levels (xarray handles named dims)
    dz = height_agl.diff("altitude")

    # Both buoyancy_mid.isel(altitude=slice(1,None)) and dz have N-1 altitude
    # levels but with mismatched altitude coordinates.  Overwrite dz's altitude
    # coordinate to match buoyancy_mid's so xarray multiplies element-wise.
    b_trimmed = buoyancy_mid.isel(altitude=slice(1, None))
    dz_aligned = dz.assign_coords(altitude=b_trimmed.coords["altitude"])
    integrand_da = b_trimmed * dz_aligned  # xarray handles broadcasting by dim name
    alt_axis = list(integrand_da.dims).index("altitude")

    cumulative_work = np.cumsum(integrand_da.values, axis=alt_axis)

    # Prepend a zero slice for the lowest altitude level (w=0 at ground)
    zero_shape = list(cumulative_work.shape)
    zero_shape[alt_axis] = 1
    cumulative_work = np.concatenate(
        [np.zeros(zero_shape), cumulative_work], axis=alt_axis
    )

    # Step 4: velocity from work-energy theorem  [m/s]
    thermal_velocity = ENTRAINMENT_FACTOR * np.sqrt(
        2.0 * np.clip(cumulative_work, 0, None)
    )

    # Wrap back into an xarray DataArray with original coords
    thermal_velocity_da = xr.DataArray(
        thermal_velocity,
        dims=buoyancy.dims,
        coords=buoyancy.coords,
    )

    return thermal_velocity_da, thermal_temp_diff


def extract_timestamp(filename):
    # Define a regex pattern to capture the timestamp
    pattern = r"(\d{4})(\d{2})(\d{2})T(\d{2})Z"
    match = re.search(pattern, filename)

    if match:
        year, month, day, hour = match.groups()
        return f"{year}-{month}-{day}T{hour}:00Z"
    else:
        return None


def find_latest_meps_file():
    # The MEPS dataset: https://github.com/metno/NWPdocs/wiki/MEPS-dataset
    today = datetime.datetime.today()
    catalog_url = f"https://thredds.met.no/thredds/catalog/meps25epsarchive/{today.year}/{today.month:02d}/{today.day:02d}/catalog.xml"
    file_url_base = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{today.year}/{today.month:02d}/{today.day:02d}"
    # Get the datasets from the catalog
    catalog = TDSCatalog(catalog_url)
    datasets = [s for s in catalog.datasets if "meps_det_ml" in s]
    file_path = f"{file_url_base}/{sorted(datasets)[-1]}"
    return file_path


def _compute_grid_rotation_angle(latitude, longitude):
    """Compute the local grid rotation angle *alpha* (radians) for a Lambert grid.

    Uses numpy finite differences on the 2-D lat/lon fields to determine how
    the grid y-axis is rotated relative to true north at each point.  This is
    the vectorised equivalent of the loop-based approach shown in the MET
    Norway NWPdocs examples.

    Parameters
    ----------
    latitude, longitude : xarray.DataArray
        2-D arrays with dims (y, x).

    Returns
    -------
    alpha_rad : numpy.ndarray
        Rotation angle in **radians** with shape (y, x).  To convert grid-
        relative (x_wind, y_wind) to geographic (u_east, v_north)::

            u_east  = x_wind * cos(alpha) - y_wind * sin(alpha)
            v_north = x_wind * sin(alpha) + y_wind * cos(alpha)
    """
    lat = latitude.values
    lon = longitude.values

    # Finite differences along the y-axis (grid "northward" direction).
    # Use central differences for interior points and forward/backward at edges.
    dlat_dy = np.gradient(lat, axis=0)  # change in latitude per y-step
    dlon_dy = np.gradient(lon, axis=0)  # change in longitude per y-step

    # Convert longitude difference to km-equivalent, accounting for latitude.
    # At latitude phi, 1 degree of longitude ≈ cos(phi) × 1 degree of latitude
    # in terms of great-circle distance, so we weight dlon accordingly.
    cos_lat = np.cos(np.radians(lat))
    dlon_dy_km = dlon_dy * cos_lat  # proportional to east-west km per y-step
    dlat_dy_km = dlat_dy  # proportional to north-south km per y-step

    # alpha = atan2(dlat_dy_km, dlon_dy_km) - 90°   (in radians)
    # This gives the angle between the grid y-axis and true north.
    alpha_rad = np.arctan2(dlat_dy_km, dlon_dy_km) - np.pi / 2

    return alpha_rad


def load_meps_for_location(file_path=None, altitude_min=0, altitude_max=4000):
    """
    file_path=None
    altitude_min=0
    altitude_max=3000
    """

    if file_path is None:
        file_path = find_latest_meps_file()

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

    # get geopotential
    time_range_sfc = "[0:1:0]"
    surf_params = {
        "x": x_range,
        "y": y_range,
        "time": f"{time_range}",
        "surface_geopotential": f"{time_range_sfc}[0:1:0]{y_range}{x_range}",
        "air_temperature_0m": f"{time_range}[0:1:0]{y_range}{x_range}",
    }
    file_path_surf = f"{file_path.replace('meps_det_ml', 'meps_det_sfc')}?{','.join(f'{k}{v}' for k, v in surf_params.items())}"

    # Load surface parameters and merge into the main dataset
    surf = xr.open_dataset(file_path_surf, cache=True)
    # Convert the surface geopotential to elevation
    elevation = (surf.surface_geopotential / 9.80665).squeeze()
    # elevation.plot()
    subset["elevation"] = elevation
    air_temperature_0m = surf.air_temperature_0m.squeeze()
    subset["air_temperature_0m"] = air_temperature_0m

    # subset.elevation.plot()
    def hybrid_to_height(ds):
        """
        ds = subset
        """
        # Constants
        R = 287.05  # Gas constant for dry air
        g = 9.80665  # Gravitational acceleration

        # Calculate the pressure at each level
        p = ds["ap"] + ds["b"] * ds["surface_air_pressure"]  # .mean("ensemble_member")

        # Get the temperature at each level
        T = ds["air_temperature_ml"]  # .mean("ensemble_member")

        # Mean temperature between each level and the lowest (surface-adjacent) level
        dT_mean = 0.5 * (T + T.isel(hybrid=-1))

        # Calculate the height using the hypsometric equation
        dz = (R * dT_mean / g) * np.log(ds["surface_air_pressure"] / p)

        return dz

    # Compute height above ground (AGL) per gridpoint: (hybrid, y, x)
    # This is the physically correct height for thermal calculations.
    height_agl_3d = hybrid_to_height(subset).mean("time").squeeze()

    # Store per-gridpoint AGL as a data variable (used in thermal calcs).
    # Assign BEFORE swap_dims so it inherits the dimension rename.
    subset["height_agl"] = height_agl_3d

    # Domain-averaged altitude used as 1D dimension coordinate for
    # interpolation and binning — an acceptable approximation.
    altitude_1d = height_agl_3d.mean("x").mean("y")
    subset = subset.assign_coords(altitude=("hybrid", altitude_1d.data))
    subset = subset.swap_dims({"hybrid": "altitude"})

    # filter subset on altitude ranges
    subset = subset.where(
        (subset.altitude >= altitude_min) & (subset.altitude <= altitude_max), drop=True
    ).squeeze()

    # --- Grid rotation correction ---
    # x_wind_ml and y_wind_ml are grid-relative (Lambert conformal conic),
    # not geographic east/north.  Rotate them to true east/north using the
    # local grid rotation angle alpha, computed from the lat/lon fields.
    # Reference: https://github.com/metno/NWPdocs/wiki/FAQ#wind-direction-obtained-from-x-y-wind
    alpha_rad = _compute_grid_rotation_angle(subset["latitude"], subset["longitude"])
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    x_wind_geo = subset["x_wind_ml"] * cos_alpha - subset["y_wind_ml"] * sin_alpha
    y_wind_geo = subset["x_wind_ml"] * sin_alpha + subset["y_wind_ml"] * cos_alpha
    subset["x_wind_ml"] = x_wind_geo
    subset["y_wind_ml"] = y_wind_geo

    wind_speed = np.sqrt(subset["x_wind_ml"] ** 2 + subset["y_wind_ml"] ** 2)
    subset = subset.assign(wind_speed=(("time", "altitude", "y", "x"), wind_speed.data))

    thermal_velocity_da, thermal_temp_diff_da = compute_thermal_velocity(subset)
    subset["thermal_velocity"] = thermal_velocity_da

    # Find thermal top: highest altitude where thermal_temp_diff exceeds a
    # usable threshold.  0.5 °C excess ≈ minimum for soarable thermals.
    THERMAL_TOP_THRESHOLD = 0.5  # °C

    thermal_temp_diff = thermal_temp_diff_da
    # Zero out values below the threshold so argmax finds the first altitude
    # (from the top, since MEPS altitudes are descending) where thermals are
    # still usable.
    usable_diff = thermal_temp_diff.where(
        thermal_temp_diff >= THERMAL_TOP_THRESHOLD, 0.0
    )
    # Add tiny value at the lowest altitude to avoid ground being reported as
    # thermal top when no altitude exceeds the threshold.
    usable_diff = usable_diff.where(
        (usable_diff.sum("altitude") > 0)
        | (subset["altitude"] != subset.altitude.min()),
        usable_diff + 1e-6,
    )
    indices = (usable_diff > 0).argmax(dim="altitude")
    # Use per-gridpoint height_agl for thermal top so the value reflects the
    # actual height above local ground, not the domain-averaged altitude label.
    thermal_top = subset["height_agl"].isel(altitude=indices)
    subset = subset.assign(thermal_top=(("time", "y", "x"), thermal_top.data))
    subset = subset.set_coords(["latitude", "longitude"])
    return subset


def subsample_lat_lon(dataset, lat_stride=2, lon_stride=2):
    """
    Subsample the latitude and longitude points from the dataset.

    Parameters:
    - dataset: xarray.Dataset, the dataset to subsample.
    - lat_stride: int, stride value for latitude subsampling.
    - lon_stride: int, stride value for longitude subsampling.

    Returns:
    - xarray.Dataset, the subsampled dataset.
    """
    # Check if latitude and longitude dimensions are present
    if "y" not in dataset.dims or "x" not in dataset.dims:
        raise ValueError(
            "Dataset does not contain 'y' and 'x' dimensions for latitude and longitude."
        )

    # Subsample latitude and longitude
    subsampled_dataset = dataset.isel(
        y=slice(None, None, lat_stride), x=slice(None, None, lon_stride)
    )

    return subsampled_dataset


if __name__ == "__main__":
    dataset_file_path = find_latest_meps_file()
    forecast_timestamp_str = extract_timestamp(dataset_file_path.split("/")[-1])

    from dateutil import parser

    forecast_timestamp_datetime = parser.isoparse(forecast_timestamp_str)

    # Check in db if forecast already exists
    db = db_utils.Database()
    # Find max forecast timestamp from the detailed output table.
    last_executed_forecast_timestamp = db.read(
        "select max(forecast_timestamp) as max_forecast_timestamp from detailed_forecasts"
    )
    max_forecast_timestamp = last_executed_forecast_timestamp[0, 0]
    no_new_forecast_exists = (max_forecast_timestamp is not None) and (
        max_forecast_timestamp >= forecast_timestamp_datetime
    )

    if no_new_forecast_exists and (os.getenv("TRIGGER_SOURCE") != "push"):
        print(
            f"Forecast timestamp: \n {forecast_timestamp_datetime} \nLast executed forecast timestamp: \n {last_executed_forecast_timestamp[0, 0]} \n Trigger source: \n {os.getenv('TRIGGER_SOURCE')}"
        )
        print("Same or newer forecast already exists in db. Exiting.")
    else:
        subset = load_meps_for_location(dataset_file_path)

        # %% Interpolate altitude and subsample lat lon
        below_600_intervals = np.arange(0, 600, 100)
        above_600_intervals = np.arange(600, subset.altitude.max() + 200, 200)
        altitude_intervals = np.concatenate([below_600_intervals, above_600_intervals])
        altitude_interpolated_subset = subset.interp(
            altitude=altitude_intervals, method="linear"
        )

        # %% Convert to dataframe
        df = (
            pl.DataFrame(altitude_interpolated_subset.to_dataframe().reset_index())
            .with_columns(
                forecast_timestamp=pl.lit(forecast_timestamp_str).cast(pl.Datetime)
            )
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
            )
        )
        # %% categorize all points into a kommunenavn
        # Step 1: Read the GeoJSON file
        # hentet fra https://github.com/robhop/fylker-og-kommuner/blob/main/Kommuner-S.geojson
        geojson_path = "Kommuner-S.geojson"
        areas_gdf = gpd.read_file(geojson_path)[["geometry", "name"]]
        unique_lat_lon = df.select("longitude", "latitude").unique().to_pandas()

        points_forecast = gpd.GeoDataFrame(
            unique_lat_lon,
            geometry=[
                Point(xy)
                for xy in zip(unique_lat_lon["longitude"], unique_lat_lon["latitude"])
            ],
        )
        points_forecast.set_crs(areas_gdf.crs, inplace=True)
        named_lat_lon = gpd.sjoin(
            points_forecast, areas_gdf, how="left", predicate="within"
        )
        df_names = pl.DataFrame(
            named_lat_lon[["longitude", "latitude", "name"]]
        ).drop_nulls()

        # Group by name, time and altitude and calculate the mean of the other columns
        area_forecasts = (
            df.join(df_names, on=["longitude", "latitude"], how="inner")
            .group_by("forecast_timestamp", "time", "name", "altitude")
            .median()
            .with_columns(point_type=pl.lit("area"))
        )

        # get takeoffs and find nearest forecast point
        geojson_takeoffs = takeoff_utils.fetch_takeoffs_norway()
        geojson_takeoffs["latitude_takeoff"] = geojson_takeoffs.geometry.y
        geojson_takeoffs["longitude_takeoff"] = geojson_takeoffs.geometry.x
        geojson_takeoffs.set_crs(areas_gdf.crs, inplace=True)
        takeoffs = gpd.sjoin_nearest(
            geojson_takeoffs, points_forecast, how="left", max_distance=10000
        )[
            ["name", "longitude_takeoff", "latitude_takeoff", "longitude", "latitude"]
        ]  # ,'pge_link'
        df_takeoffs = pl.DataFrame(takeoffs)

        takeoff_forecasts = (
            df.join(df_takeoffs, on=["longitude", "latitude"], how="inner")
            # deselect lat lon
            .select(pl.exclude("longitude", "latitude"))
            .rename({"longitude_takeoff": "longitude", "latitude_takeoff": "latitude"})
            .with_columns(point_type=pl.lit("takeoff"))
        )

        point_forecasts = pl.concat(
            [takeoff_forecasts, area_forecasts.select(takeoff_forecasts.columns)],
            how="vertical_relaxed",
        )
        # %%

        # Save to db — use TRUNCATE+append to preserve indexes and avoid
        # DROP TABLE timeouts on large indexed tables.
        print("Save area forecast to db..")
        db.replace_data(area_forecasts, "area_forecasts")
        print("saving detailed forecast to db...")
        db.replace_data(point_forecasts, "detailed_forecasts")
        print(
            f"saved {len(point_forecasts)} point forecasts and {len(area_forecasts)} area forecasts to db."
        )

        # create_index_query = "CREATE INDEX idx_time_name ON weather_forecasts (time, longitude, latitude);"
        # res = db.execute_query(create_index_query)
# %%
