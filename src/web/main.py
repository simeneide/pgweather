from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import forecast_service
from .models import (
    AirgramPayloadRequest,
    AirgramPayloadResponse,
    FrontendMetaRequest,
    FrontendMetaResponse,
    HealthResponse,
    MapPayloadRequest,
    MapPayloadResponse,
    MetaResponse,
    SummaryRequest,
    SummaryResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="pgweather", version="1.1.0")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"
_FRONTEND_ASSETS = _FRONTEND_DIST / "assets"

if _FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_ASSETS)), name="assets")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        latest_ts = forecast_service.get_latest_forecast_timestamp()
        age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600
        return HealthResponse(
            status="ok",
            latest_forecast_timestamp=latest_ts.isoformat(),
            forecast_age_hours=round(age_hours, 2),
        )
    except Exception:
        logger.exception("Health check: data not yet available")
        return HealthResponse(status="ok", detail="warming up, data not loaded yet")


@app.get("/api/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    latest_ts = forecast_service.get_latest_forecast_timestamp()
    times = forecast_service.get_available_times()
    return MetaResponse(
        latest_forecast_timestamp=latest_ts.isoformat(),
        available_times=[t.isoformat() for t in times],
        takeoff_count=len(times),
    )


@app.post("/api/frontend/meta", response_model=FrontendMetaResponse)
def frontend_meta(payload: FrontendMetaRequest) -> FrontendMetaResponse:
    return forecast_service.get_frontend_metadata(
        model_source=payload.model_source,
        forecast_ts=payload.forecast_timestamp,
    )


@app.post("/api/frontend/map", response_model=MapPayloadResponse)
def frontend_map(payload: MapPayloadRequest) -> MapPayloadResponse:
    return forecast_service.build_map_payload(
        selected_time=payload.selected_time,
        selected_name=payload.selected_name,
        zoom=payload.zoom,
        forecast_ts=payload.forecast_timestamp,
        wind_altitude=payload.wind_altitude,
        model_source=payload.model_source,
    )


@app.post("/api/frontend/airgram", response_model=AirgramPayloadResponse)
def frontend_airgram(payload: AirgramPayloadRequest) -> AirgramPayloadResponse:
    return forecast_service.build_airgram_payload(
        target_name=payload.location,
        selected_date=payload.selected_date,
        altitude_max=payload.altitude_max,
        selected_hour=payload.selected_hour,
        forecast_ts=payload.forecast_timestamp,
        model_source=payload.model_source,
    )


@app.post("/api/frontend/summary", response_model=SummaryResponse)
def frontend_summary(payload: SummaryRequest) -> SummaryResponse:
    return forecast_service.get_summary_payload(
        selected_name=payload.selected_name,
        selected_time=payload.selected_time,
        forecast_ts=payload.forecast_timestamp,
        model_source=payload.model_source,
    )


@app.get("/{full_path:path}")
def frontend_app(full_path: str):
    if _FRONTEND_DIST.exists():
        if full_path:
            requested = (_FRONTEND_DIST / full_path).resolve()
            if (
                requested.exists()
                and requested.is_file()
                and _FRONTEND_DIST in requested.parents
            ):
                return FileResponse(requested)
        index_file = _FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return JSONResponse(
        {
            "message": (
                "Frontend build not found. Build it with: "
                "`cd frontend && npm install && npm run build`"
            )
        },
        status_code=503,
    )
