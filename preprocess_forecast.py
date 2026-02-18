"""MEPS forecast preprocessing pipeline.

This is the entry point for the MEPS data pipeline, typically run via
GitHub Actions cron job. It delegates to the refactored modules in
``src.preprocessing``.

For backward compatibility, the model-independent functions and the
MEPS-specific loader are re-exported here so any existing imports still work.

Usage::

    uv run python preprocess_forecast.py [--date YYYY-MM-DD] [--run HH]
"""

from __future__ import annotations

import datetime
import logging
import os

import geopandas as gpd
from dotenv import load_dotenv

load_dotenv()

import db_utils
import takeoff_utils

# Re-export from refactored modules for backward compatibility
from src.preprocessing.base import (
    run_post_loading_pipeline,
)
from src.preprocessing.meps import (
    extract_timestamp,
    find_latest_meps_file,
    load_meps_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Backward-compatible alias
load_meps_for_location = load_meps_data

MODEL_SOURCE = "meps"

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fetch forecast for a specific date (YYYY-MM-DD). Defaults to today.",
    )
    arg_parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific model run hour (e.g. 06, 09, 12). Defaults to latest available.",
    )
    args = arg_parser.parse_args()
    target_date = (
        datetime.datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    )

    dataset_file_path = find_latest_meps_file(date=target_date, run=args.run)
    forecast_timestamp_str = extract_timestamp(dataset_file_path.split("/")[-1])

    from dateutil import parser

    forecast_timestamp_datetime = parser.isoparse(forecast_timestamp_str)

    # Check in db if forecast already exists
    db = db_utils.Database()
    last_executed_forecast_timestamp = db.read(
        "select max(forecast_timestamp) as max_forecast_timestamp from detailed_forecasts"
        f" where model_source = '{MODEL_SOURCE}'"
    )
    max_forecast_timestamp = last_executed_forecast_timestamp[0, 0]
    no_new_forecast_exists = (max_forecast_timestamp is not None) and (
        max_forecast_timestamp >= forecast_timestamp_datetime
    )

    if no_new_forecast_exists and (os.getenv("TRIGGER_SOURCE") != "push"):
        logger.info(
            "Forecast timestamp: %s, Last executed: %s, Trigger: %s",
            forecast_timestamp_datetime,
            last_executed_forecast_timestamp[0, 0],
            os.getenv("TRIGGER_SOURCE"),
        )
        logger.info("Same or newer forecast already exists in db. Exiting.")
    else:
        # Load MEPS data
        subset = load_meps_data(dataset_file_path)

        # Load spatial reference data
        geojson_path = "Kommuner-S.geojson"
        areas_gdf = gpd.read_file(geojson_path)[["geometry", "name"]]
        takeoffs_gdf = takeoff_utils.fetch_takeoffs_norway()

        # Run the full post-loading pipeline
        run_post_loading_pipeline(
            subset=subset,
            forecast_timestamp=forecast_timestamp_str,
            model_source=MODEL_SOURCE,
            takeoffs_gdf=takeoffs_gdf,
            areas_gdf=areas_gdf,
            db=db,
        )
