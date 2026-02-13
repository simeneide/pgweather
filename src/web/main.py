from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

from .dash_ui import create_dash_app
from .forecast_service import (
    get_available_times,
    get_latest_forecast_timestamp,
    load_forecast_data,
)

app = FastAPI(title="pgweather", version="1.0.0")


@app.get("/api/health")
def health() -> dict[str, object]:
    df = load_forecast_data()
    latest_ts = get_latest_forecast_timestamp(df)
    age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600
    return {
        "status": "ok",
        "rows": len(df),
        "latest_forecast_timestamp": latest_ts.isoformat(),
        "forecast_age_hours": round(age_hours, 2),
    }


@app.get("/api/meta")
def meta() -> dict[str, object]:
    df = load_forecast_data()
    times = get_available_times(df)
    return {
        "latest_forecast_timestamp": get_latest_forecast_timestamp(df).isoformat(),
        "available_times": [t.isoformat() for t in times],
    }


dash_app = create_dash_app()
app.mount("/", WSGIMiddleware(dash_app.server))
