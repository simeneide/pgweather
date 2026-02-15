from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from .dash_ui import create_dash_app
from .forecast_service import get_available_times, get_latest_forecast_timestamp
from .models import HealthResponse, MetaResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="pgweather", version="1.0.0")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        latest_ts = get_latest_forecast_timestamp()
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
    latest_ts = get_latest_forecast_timestamp()
    times = get_available_times()
    return MetaResponse(
        latest_forecast_timestamp=latest_ts.isoformat(),
        available_times=[t.isoformat() for t in times],
        takeoff_count=len(times),
    )


dash_app = create_dash_app()
app.mount("/", WSGIMiddleware(dash_app.server))
