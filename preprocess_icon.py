"""ICON forecast preprocessing pipeline.

Entry point for ICON-EU and ICON Global data pipelines.
Downloads GRIB2 data from DWD opendata, computes thermals, and writes
to the same database schema as MEPS (with model_source discriminator).

Usage::

    uv run python preprocess_icon.py [--model icon-eu|icon-global] [--run HH]
    uv run python preprocess_icon.py --model icon-eu --region norway
    uv run python preprocess_icon.py --model icon-eu --region bir
"""

from __future__ import annotations

import datetime as dt
import logging
import os

from dotenv import load_dotenv

load_dotenv()

import db_utils
import takeoff_utils
from src.preprocessing.base import run_post_loading_pipeline
from src.preprocessing.icon import (
    find_latest_icon_eu_run,
    format_icon_timestamp,
    load_icon_eu_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# Region configurations
REGIONS: dict[str, dict] = {
    "norway": {
        "lat_bounds": (57.0, 72.0),
        "lon_bounds": (3.0, 32.0),
        "country_iso": "NO",
        "model_source": "icon-eu",
    },
    "bir": {
        "lat_bounds": (31.0, 33.0),
        "lon_bounds": (76.0, 78.0),
        "country_iso": "IN",
        "model_source": "icon-eu",  # Bir is within ICON-EU domain
    },
    "europe": {
        "lat_bounds": (35.0, 72.0),
        "lon_bounds": (-12.0, 45.0),
        "country_iso": None,  # Multiple countries
        "model_source": "icon-eu",
    },
}


def run_icon_eu_pipeline(
    region: str = "norway",
    run: int | None = None,
    max_forecast_hours: int = 48,
) -> None:
    """Run the full ICON-EU pipeline for a given region."""
    region_config = REGIONS.get(region)
    if region_config is None:
        raise ValueError(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")

    model_source = region_config["model_source"]

    # Discover latest run
    run_id, init_time = find_latest_icon_eu_run(run=run)
    forecast_timestamp = format_icon_timestamp(init_time)

    # Check if forecast already exists in DB
    db = db_utils.Database()
    last_ts = db.read(
        "SELECT max(forecast_timestamp) AS max_ts FROM detailed_forecasts"
        f" WHERE model_source = '{model_source}'"
    )
    max_ts = last_ts[0, 0]
    # Ensure both datetimes are tz-aware for comparison (DB may return naive)
    if max_ts is not None and hasattr(max_ts, "tzinfo") and max_ts.tzinfo is None:
        max_ts = max_ts.replace(tzinfo=dt.timezone.utc)
    if max_ts is not None and max_ts >= init_time:
        if os.getenv("TRIGGER_SOURCE") != "push":
            logger.info(
                "ICON-EU forecast %s already in DB (latest: %s). Skipping.",
                forecast_timestamp,
                max_ts,
            )
            return

    # Load data
    logger.info("Loading ICON-EU data for region=%s...", region)
    subset = load_icon_eu_data(
        run_id=run_id,
        init_time=init_time,
        max_forecast_hours=max_forecast_hours,
        lat_bounds=region_config["lat_bounds"],
        lon_bounds=region_config["lon_bounds"],
    )

    # Fetch takeoffs for the region
    country_iso = region_config["country_iso"]
    if country_iso:
        takeoffs_gdf = takeoff_utils.fetch_takeoffs(country_iso=country_iso)
    else:
        # For multi-country regions, fetch each country separately
        takeoffs_gdf = None
        logger.warning("No takeoff fetching for multi-country region %s", region)
        return

    if takeoffs_gdf is None or len(takeoffs_gdf) == 0:
        logger.error("No takeoffs found for region=%s, country=%s", region, country_iso)
        return

    # No area aggregation for non-Norwegian regions (no municipality GeoJSON)
    areas_gdf = None

    # Run shared pipeline
    run_post_loading_pipeline(
        subset=subset,
        forecast_timestamp=forecast_timestamp,
        model_source=model_source,
        takeoffs_gdf=takeoffs_gdf,
        areas_gdf=areas_gdf,
        db=db,
    )

    logger.info("ICON-EU pipeline complete for region=%s", region)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ICON forecast preprocessing")
    parser.add_argument(
        "--model",
        type=str,
        default="icon-eu",
        choices=["icon-eu", "icon-global"],
        help="Which ICON model to process.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="norway",
        choices=list(REGIONS.keys()),
        help="Region to process.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Specific model run hour (0, 3, 6, ..., 21).",
    )
    parser.add_argument(
        "--max-hours",
        type=int,
        default=48,
        help="Maximum forecast lead time in hours.",
    )
    args = parser.parse_args()

    if args.model == "icon-eu":
        run_icon_eu_pipeline(
            region=args.region,
            run=args.run,
            max_forecast_hours=args.max_hours,
        )
    elif args.model == "icon-global":
        logger.error("ICON Global not yet implemented. Use icon-eu.")
        raise SystemExit(1)
