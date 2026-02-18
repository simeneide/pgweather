"""Utilities for fetching paragliding takeoff locations from ParaglidingEarth.com."""

from __future__ import annotations

import logging
from typing import Optional

import geojson
import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

# ParaglidingEarth API endpoint
_API_URL = "http://www.paraglidingearth.com/api/geojson/getCountrySites.php"


def fetch_takeoffs(
    country_iso: str = "NO",
    limit: Optional[int] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Fetch takeoff locations for a country from ParaglidingEarth.com.

    Parameters
    ----------
    country_iso : str
        ISO country code (e.g. "NO" for Norway, "IN" for India).
    limit : int, optional
        Maximum number of sites to return.

    Returns
    -------
    GeoDataFrame or None
        Takeoff locations with geometry and properties, or None on failure.
    """
    params: dict[str, str | int] = {"iso": country_iso}
    if limit:
        params["limit"] = limit

    try:
        response = requests.get(_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to fetch takeoffs for country=%s", country_iso)
        return None

    data = geojson.loads(response.text)
    if not data.get("features"):
        logger.warning("No takeoff features found for country=%s", country_iso)
        return None

    gdf = gpd.GeoDataFrame.from_features(data["features"])
    logger.info("Fetched %d takeoffs for country=%s", len(gdf), country_iso)
    return gdf


def fetch_takeoffs_norway(limit: Optional[int] = None) -> Optional[gpd.GeoDataFrame]:
    """Fetch Norwegian takeoff locations (backward-compatible wrapper)."""
    return fetch_takeoffs(country_iso="NO", limit=limit)


if __name__ == "__main__":
    df_takeoffs = fetch_takeoffs_norway()
    if df_takeoffs is not None:
        print(f"Fetched {len(df_takeoffs)} Norwegian takeoffs")

    df_india = fetch_takeoffs(country_iso="IN")
    if df_india is not None:
        print(f"Fetched {len(df_india)} Indian takeoffs")
