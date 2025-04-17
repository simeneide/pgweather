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


# %%
def compute_thermal_temp_difference(subset):
    lapse_rate = 0.0098
    ground_temp = subset.air_temperature_0m - 273.3
    air_temp = subset["air_temperature_ml"] - 273.3  # .ffill(dim='altitude')

    # dimensions
    # 'air_temperature_ml'  altitude: 4 y: 3, x: 3
    # 'elevation'                       y: 3  x: 3
    # 'altitude'            altitude: 4

    # broadcast ground temperature to all altitudes, but let it decrease by lapse rate
    altitude_diff = subset.altitude - subset.elevation
    altitude_diff = altitude_diff.where(altitude_diff >= 0, 0)
    temp_decrease = lapse_rate * altitude_diff
    ground_parcel_temp = ground_temp - temp_decrease
    thermal_temp_diff = (ground_parcel_temp - air_temp).clip(min=0)
    return thermal_temp_diff


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


def load_meps_for_location(file_path=None, altitude_min=0, altitude_max=3000):
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

        # Calculate the height difference between each level and the surface
        dp = ds["surface_air_pressure"] - p  # Pressure difference
        dT = T - T.isel(hybrid=-1)  # Temperature difference relative to the surface
        dT_mean = 0.5 * (T + T.isel(hybrid=-1))  # Mean temperature

        # Calculate the height using the hypsometric equation
        dz = (R * dT_mean / g) * np.log(ds["surface_air_pressure"] / p)

        return dz

    altitude = hybrid_to_height(subset).mean("time").squeeze().mean("x").mean("y")
    subset = subset.assign_coords(altitude=("hybrid", altitude.data))
    subset = subset.swap_dims({"hybrid": "altitude"})

    # filter subset on altitude ranges
    subset = subset.where(
        (subset.altitude >= altitude_min) & (subset.altitude <= altitude_max), drop=True
    ).squeeze()

    wind_speed = np.sqrt(subset["x_wind_ml"] ** 2 + subset["y_wind_ml"] ** 2)
    subset = subset.assign(wind_speed=(("time", "altitude", "y", "x"), wind_speed.data))

    subset["thermal_temp_diff"] = compute_thermal_temp_difference(subset)
    # subset = subset.assign(thermal_temp_diff=(('time', 'altitude','y','x'), thermal_temp_diff.data))

    # Find the indices where the thermal temperature difference is zero or negative
    # Create tiny value at ground level to avoid finding the ground as the thermal top
    thermal_temp_diff = subset["thermal_temp_diff"]
    thermal_temp_diff = thermal_temp_diff.where(
        (thermal_temp_diff.sum("altitude") > 0)
        | (subset["altitude"] != subset.altitude.min()),
        thermal_temp_diff + 1e-6,
    )
    indices = (thermal_temp_diff > 0).argmax(dim="altitude")
    # Get the altitudes corresponding to these indices
    thermal_top = subset.altitude[indices]
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
    # Find max forecast timestamp:
    last_executed_forecast_timestamp = db.read(
        f"select max(forecast_timestamp) as max_forecast_timestamp from weather_forecasts"
    )
    no_new_forecast_exists = (
        last_executed_forecast_timestamp[0, 0] >= forecast_timestamp_datetime
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
                "thermal_temp_diff",
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

        point_forecasts = (
            df.join(df_takeoffs, on=["longitude", "latitude"], how="inner")
            # deselect lat lon
            .select(pl.exclude("longitude", "latitude"))
            .rename({"longitude_takeoff": "longitude", "latitude_takeoff": "latitude"})
            .with_columns(point_type=pl.lit("takeoff"))
        )

        point_forecasts = pl.concat(
            [point_forecasts, area_forecasts.select(point_forecasts.columns)],
            how="vertical_relaxed",
        )

        # Save to aiven db
        print("Save area forecast to db..")
        db.write(area_forecasts, "area_forecasts", if_table_exists="replace")
        print("saving detailed forecast to db...")
        db.write(point_forecasts, "detailed_forecasts", if_table_exists="replace")
        print(
            f"saved {len(point_forecasts)} point forecasts and {len(area_forecasts)} area forecasts to db."
        )

        # create_index_query = "CREATE INDEX idx_time_name ON weather_forecasts (time, longitude, latitude);"
        # res = db.execute_query(create_index_query)
# %%
