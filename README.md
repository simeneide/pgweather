# Termikkvarsel (pgweather)

Norwegian paragliding thermal forecast app. Displays MEPS weather model data as interactive maps and airgrams for paragliding takeoff locations across Norway.

## Architecture

- **FastAPI + Dash** — single process serving API endpoints (`/api/*`) and the interactive UI (`/`)
- **Supabase** (Postgres) — stores preprocessed forecast data (`detailed_forecasts`, `area_forecasts`)
- **Fly.io** — hosting (app name: `pgweather`, region: `arn`)
- **GitHub Actions** — cron job runs `preprocess_forecast.py` every 3 hours to fetch MEPS data and write to Supabase

### Key files

```
src/web/main.py              # FastAPI app, mounts Dash, /api/health and /api/meta
src/web/dash_ui.py           # Dash layout + callbacks (date/time picker, location, map, airgram)
src/web/forecast_service.py  # Data loading, map/airgram builders, caching
assets/map.css               # CSS for pill buttons, map attribution, dropdown fixes
db_utils.py                  # Database wrapper (reads SUPABASE_DB_URL)
preprocess_forecast.py       # MEPS forecast pipeline (GitHub Actions)
Kommuner-S.geojson           # Norwegian municipality boundaries for area overlay
app.py                       # Legacy Streamlit UI (no longer the entrypoint)
```

## Run locally

```bash
# Requires SUPABASE_DB_URL env var pointing to Supabase Postgres
uv run uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8080
```

- UI: `http://localhost:8080`
- Health: `http://localhost:8080/api/health`

## Deploy to Fly.io

```bash
# Set the database secret (one-time)
fly secrets set SUPABASE_DB_URL="postgresql://postgres.<PROJECT_REF>:<PASSWORD>@aws-0-eu-north-1.pooler.supabase.com:6543/postgres" -a pgweather

# Deploy
fly deploy -a pgweather

# Check logs
fly logs -a pgweather
```

## Data pipeline

The GitHub Actions workflow (`.github/workflows/preprocess_data.yml`) runs on cron `20 */3 * * *` and on push to main. It fetches the latest MEPS forecast from thredds.met.no and writes preprocessed data to the `detailed_forecasts` table in Supabase.
