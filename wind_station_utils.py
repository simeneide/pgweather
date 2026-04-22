"""Wind station catalog — refreshed at the start of each preprocess run.

Aggregates public wind stations from two providers and returns them as a
``geopandas.GeoDataFrame`` that is shape-compatible with the takeoff
GeoDataFrame returned by :mod:`takeoff_utils`.  The preprocess pipeline
then interpolates MEPS (or ICON) forecasts at each station, so the
frontend can show a forecast timeline next to historical observations
when a station popup is opened.

Providers:

* **winds.mobi** — public API ``/api/2/stations/``.  Primarily Holfuy
  paraglider stations.  We grid-paginate around Scandinavia and filter
  to the MEPS domain (roughly lat 55-72 N, lon 3-32 E).
* **met.no Frost** — official Norwegian weather stations.  Requires
  the ``MET_FROST_CLIENT_ID`` environment variable.  Silently skipped
  when unset.

Each station is emitted with a globally unique name:

* winds.mobi: ``station:winds-mobi:<id>``   (e.g. ``station:winds-mobi:holfuy-1013``)
* Frost:      ``station:frost:<id>``        (e.g. ``station:frost:SN18700``)

Keeping the ``station:`` prefix on the ``name`` column means the
forecast service can match them with a single ``LIKE`` clause and they
never collide with takeoff names.

Station catalogs are considered essentially static on the timescale of
a preprocess run so we fetch them fresh but do not attempt to merge
with any prior catalog — the next run will re-add any stations we
missed.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, Optional

import geopandas as gpd
import requests
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MEPS domain bounding box (inclusive).  Stations outside this box can
# still be included but the MEPS interpolation is meaningless beyond
# it, so we filter aggressively.  Matches the coverage documented on
# https://github.com/metno/NWPdocs/wiki/MEPS-dataset .
MEPS_BBOX = (54.0, 72.0, 2.0, 32.0)  # (min_lat, max_lat, min_lon, max_lon)

WINDS_MOBI_URL = "https://winds.mobi/api/2/stations/"
WINDS_MOBI_USER_AGENT = "pgweather.app (paragliding forecast preprocessor)"
WINDS_MOBI_PAGE_LIMIT = 500  # Max per request — be polite.
WINDS_MOBI_TIMEOUT = 30

FROST_SOURCES_URL = "https://frost.met.no/sources/v0.jsonld"
FROST_USER_AGENT = "pgweather.app (paragliding forecast preprocessor)"
FROST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# winds.mobi
# ---------------------------------------------------------------------------


def _in_bbox(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float] = MEPS_BBOX,
) -> bool:
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def fetch_winds_mobi_stations(
    bbox: tuple[float, float, float, float] = MEPS_BBOX,
) -> list[dict]:
    """Fetch winds.mobi stations inside *bbox*.

    winds.mobi's public API supports a bbox query via ``within-pt1`` /
    ``within-pt2`` parameters (opposite corners as ``lat,lon``).  We
    page with ``limit`` + ``start`` if the response is hit with a cap.

    Returns a list of dicts with keys ``id``, ``name``, ``lat``, ``lon``,
    ``alt``, ``provider``.
    """
    min_lat, max_lat, min_lon, max_lon = bbox

    params = {
        "within-pt1": f"{max_lat},{min_lon}",  # NW corner
        "within-pt2": f"{min_lat},{max_lon}",  # SE corner
        "limit": WINDS_MOBI_PAGE_LIMIT,
        "ignore-last-measure": "true",  # we don't need the last-measurement payload
    }

    headers = {"User-Agent": WINDS_MOBI_USER_AGENT}

    out: list[dict] = []
    start = 0
    for _ in range(20):  # hard cap at 20 pages (~10k stations)
        q = dict(params)
        q["start"] = start
        try:
            resp = requests.get(
                WINDS_MOBI_URL, params=q, headers=headers, timeout=WINDS_MOBI_TIMEOUT
            )
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException:
            logger.exception("winds.mobi fetch failed at start=%d", start)
            break

        if not batch:
            break

        for raw in batch:
            sid = raw.get("_id")
            if not sid:
                continue
            loc = raw.get("loc") or {}
            coords = loc.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon = float(coords[0])
            lat = float(coords[1])
            if not _in_bbox(lat, lon, bbox):
                continue
            out.append(
                {
                    "id": str(sid),
                    "name": raw.get("short") or raw.get("name") or str(sid),
                    "lat": lat,
                    "lon": lon,
                    "alt": raw.get("alt"),
                    "provider": raw.get("pv-name", "winds.mobi"),
                }
            )

        if len(batch) < WINDS_MOBI_PAGE_LIMIT:
            break
        start += len(batch)
        time.sleep(0.2)  # gentle rate limiting

    logger.info("winds.mobi: fetched %d stations inside MEPS bbox", len(out))
    return out


# ---------------------------------------------------------------------------
# met.no Frost
# ---------------------------------------------------------------------------


def fetch_frost_stations(
    countries: Iterable[str] = ("NO", "SE", "FI", "DK"),
    client_id: Optional[str] = None,
) -> list[dict]:
    """Fetch the met.no Frost SensorSystem catalog for *countries*.

    Requires the ``MET_FROST_CLIENT_ID`` environment variable (HTTP
    Basic auth: client ID as username, empty password).  Returns an
    empty list if no credential is configured.
    """
    cid = client_id or os.getenv("MET_FROST_CLIENT_ID") or ""
    if not cid:
        logger.info("MET_FROST_CLIENT_ID not set — skipping Frost catalog fetch.")
        return []

    out: list[dict] = []
    for country in countries:
        try:
            resp = requests.get(
                FROST_SOURCES_URL,
                params={"country": country, "types": "SensorSystem"},
                auth=(cid, ""),
                headers={"User-Agent": FROST_USER_AGENT},
                timeout=FROST_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException:
            logger.exception("Frost fetch failed for country=%s", country)
            continue

        for raw in payload.get("data", []):
            sid = raw.get("id") or ""
            if not sid:
                continue
            geom = raw.get("geometry") or {}
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            lon = float(coords[0])
            lat = float(coords[1])
            if not _in_bbox(lat, lon, MEPS_BBOX):
                continue
            out.append(
                {
                    "id": sid,
                    "name": raw.get("name", sid),
                    "lat": lat,
                    "lon": lon,
                    "alt": raw.get("masl"),
                    "provider": "met.no",
                }
            )

    logger.info("Frost: fetched %d stations inside MEPS bbox", len(out))
    return out


# ---------------------------------------------------------------------------
# Combined catalog
# ---------------------------------------------------------------------------

STATION_NAME_PREFIX = "station:"


def station_name(provider_slug: str, station_id: str) -> str:
    """Construct the canonical ``name`` stored in the forecast table.

    Examples
    --------
    >>> station_name("winds-mobi", "holfuy-1013")
    'station:winds-mobi:holfuy-1013'
    >>> station_name("frost", "SN18700")
    'station:frost:SN18700'
    """
    return f"{STATION_NAME_PREFIX}{provider_slug}:{station_id}"


def parse_station_name(name: str) -> Optional[tuple[str, str]]:
    """Reverse of :func:`station_name`.  Returns ``(provider, id)`` or ``None``."""
    if not name.startswith(STATION_NAME_PREFIX):
        return None
    rest = name[len(STATION_NAME_PREFIX) :]
    provider, sep, sid = rest.partition(":")
    if not sep:
        return None
    return provider, sid


def fetch_wind_station_catalog(
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """Fetch the combined wind-station catalog and return it as a GeoDataFrame.

    Parameters
    ----------
    bbox
        Optional ``(min_lat, max_lat, min_lon, max_lon)`` to restrict the
        catalog to a specific region.  Defaults to :data:`MEPS_BBOX`.
        Useful for ICON-EU which covers a different region than MEPS; in
        practice the caller should pass the pipeline's own lat/lon
        bounds so we don't interpolate over stations outside the model
        domain.

    The returned frame has the same column layout as
    :func:`takeoff_utils.fetch_takeoffs`:

    * ``geometry`` — a ``shapely.geometry.Point`` in WGS84
    * ``name``     — the canonical ``station:<provider>:<id>`` identifier
    * ``alt``      — elevation in metres (may be None)
    * ``provider`` — provider short name (``winds.mobi``, ``met.no``, …)
    * ``display_name`` — human-friendly station label

    Duplicates across providers are dropped on the composite name.
    """
    effective_bbox = bbox or MEPS_BBOX
    records: list[dict] = []

    for s in fetch_winds_mobi_stations(bbox=effective_bbox):
        if not _in_bbox(s["lat"], s["lon"], effective_bbox):
            continue
        records.append(
            {
                "name": station_name("winds-mobi", s["id"]),
                "display_name": s["name"],
                "alt": s.get("alt"),
                "provider": s.get("provider") or "winds.mobi",
                "lat": s["lat"],
                "lon": s["lon"],
            }
        )

    for s in fetch_frost_stations():
        if not _in_bbox(s["lat"], s["lon"], effective_bbox):
            continue
        records.append(
            {
                "name": station_name("frost", s["id"]),
                "display_name": s["name"],
                "alt": s.get("alt"),
                "provider": s.get("provider") or "met.no",
                "lat": s["lat"],
                "lon": s["lon"],
            }
        )

    if not records:
        logger.warning("Wind-station catalog is empty — nothing will be stored.")
        return gpd.GeoDataFrame(
            columns=["geometry", "name", "alt", "provider", "display_name"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    geometry = [Point(r["lon"], r["lat"]) for r in records]
    gdf = gpd.GeoDataFrame(records, geometry=geometry, crs="EPSG:4326")
    gdf = gdf.drop_duplicates(subset="name").reset_index(drop=True)

    # Drop the raw lat/lon columns — geometry is the source of truth, and
    # match_takeoffs_to_grid derives latitude_takeoff / longitude_takeoff
    # from ``geometry`` directly.
    gdf = gdf.drop(columns=["lat", "lon"])

    logger.info("Wind-station catalog: %d unique stations", len(gdf))
    return gdf


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    catalog = fetch_wind_station_catalog()
    print(f"Fetched {len(catalog)} stations.")
    print(catalog.head())
