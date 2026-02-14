# AGENTS.md — pgweather

## Project Overview

Norwegian paragliding thermal forecast app ("Termikkvarsel"). Displays MEPS weather
model data as interactive maps and airgrams for paragliding takeoffs across Norway.

**Architecture:** FastAPI + Dash in a single process. Supabase (Postgres) for storage.
Fly.io for hosting. GitHub Actions cron job for data pipeline.

**Key entry points:**
- `src/web/main.py` — FastAPI app, mounts Dash, serves `/api/health` and `/api/meta`
- `src/web/dash_ui.py` — Dash layout and callbacks (UI)
- `src/web/forecast_service.py` — Data loading, map/airgram builders, caching
- `db_utils.py` — Database wrapper (Polars + ADBC driver to Postgres)
- `preprocess_forecast.py` — MEPS data pipeline (runs via GitHub Actions)
- `app.py` — **Legacy** Streamlit UI, no longer the entrypoint

## Build & Run Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run locally (requires SUPABASE_DB_URL in .env or .envrc)
uv run uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8080

# Run the data preprocessing pipeline
uv run python preprocess_forecast.py

# Deploy to Fly.io
fly deploy -a pgweather

# Check production logs
fly logs -a pgweather
```

## Lint & Format

```bash
# Format code
uv run ruff format .

# Lint (check)
uv run ruff check .

# Lint (auto-fix)
uv run ruff check --fix .
```

No ruff configuration exists in `pyproject.toml` — defaults are used.

## Tests

No test suite exists yet. If you add tests, use `pytest`:

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a single test function
uv run pytest tests/test_example.py::test_function_name

# Run with verbose output
uv run pytest -v
```

## Package Management

- **Always use `uv`**, never pip directly
- `uv sync` to install from `pyproject.toml` + `uv.lock`
- `uv add <package>` to add a dependency
- `uv run <command>` to execute in the project environment
- `requirements.txt` exists only for the Dockerfile — keep it in sync if you change deps

## Code Style Guidelines

### Python Version
- Target Python >=3.9 (per `pyproject.toml`), Docker uses 3.11
- Use modern syntax where compatible with 3.9+

### Imports
- Use `from __future__ import annotations` in all `src/` modules
- Order: stdlib → third-party → local (ruff default isort)
- Load env early: `dotenv.load_dotenv()` is called at module level where needed
- Prefer `import datetime as dt` (convention in newer files)

### Type Hints
- Add type hints to all function signatures in `src/` code
- Use `Optional[X]` from `typing` (for 3.9 compat) or `X | None` with future annotations
- Return types should be annotated: `def health() -> dict[str, object]:`
- Older root-level files (`db_utils.py`, `app.py`) have incomplete typing — improve when touching

### Naming
- Functions: `snake_case` — `load_forecast_data`, `build_map_figure`
- Private functions: underscore prefix — `_fetch_yr_forecast`, `_load_geojson`
- Constants: `UPPER_SNAKE_CASE` — `DATA_TTL_SECONDS`, `LOCAL_TZ`
- Classes: `PascalCase` — `Database`
- Module-level caches: `_CACHE`, `_YR_CACHE` (dict-based mutable globals)

### Data Libraries
- **Polars** (`pl`) is the primary DataFrame library — use it for all new code
- **xarray** for NetCDF weather model data (preprocessing pipeline only)
- **NumPy** for numerical operations, especially in visualization code
- Avoid pandas in new code (legacy only, used for GeoPandas interop)

### Visualization
- **Plotly** `go.Figure()` for all charts and maps
- Dash for interactive UI components
- Color scales and styling follow paragliding-specific conventions (thermal colors)

### Database
- Use `db_utils.Database` class for all DB access
- Queries use inline SQL strings (no ORM)
- Polars `read_database_uri` with ADBC engine for reads
- Polars `write_database` with ADBC engine for writes
- Connection string comes from `SUPABASE_DB_URL` env var

### File I/O
- Use `pathlib.Path` over `os.path`
- Use `Path.open()` over built-in `open()` where practical

### Error Handling
- Log exceptions with `logger.exception()` — use `logging.getLogger(__name__)`
- Raise `RuntimeError` for missing configuration or startup failures
- For external API calls: catch `Exception`, log, and fall back to cached data
- Do not use bare `assert` for input validation in production code
- Do not litter code with try/except unless genuinely needed

### Caching Pattern
The codebase uses module-level dict caches with TTL:
```python
_CACHE: dict[str, object] = {"loaded_at": None, "df": None}
DATA_TTL_SECONDS = 600

def load_data(force_refresh: bool = False) -> pl.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    if not force_refresh and cache_is_valid():
        return _CACHE["df"]
    # ... fetch fresh data ...
    _CACHE["loaded_at"] = now
    _CACHE["df"] = df
    return df
```

### Environment Variables
- `SUPABASE_DB_URL` — Postgres connection string (required)
- `TRIGGER_SOURCE` — set by GitHub Actions to identify pipeline trigger
- Load via `dotenv.load_dotenv()` + `os.getenv()`
- direnv (`.envrc`) is used locally
- Never commit `.env` files or secrets

## Project Structure

```
src/web/                  # Main application (FastAPI + Dash)
  main.py                 # FastAPI app entry point
  dash_ui.py              # Dash layout and callbacks
  forecast_service.py     # Data loading, figure builders, caching
assets/                   # Static files (CSS, favicon)
db_utils.py               # Database wrapper
preprocess_forecast.py    # Data pipeline (GitHub Actions)
takeoff_utils.py          # Fetches takeoff locations from paraglidingearth.com
utils.py                  # Coordinate projection + color utilities
Kommuner-S.geojson        # Norwegian municipality boundaries
app.py                    # Legacy Streamlit UI (deprecated)
```

## CI/CD

- **GitHub Actions** (`.github/workflows/preprocess_data.yml`):
  - Cron every 3h at :20 + on push to main + manual dispatch
  - Runs `uv sync` then `uv run python preprocess_forecast.py`
- **Fly.io** deployment is manual via `fly deploy`
- No automated test or lint CI — run `ruff check` locally before pushing

## Deployment

- Dockerfile builds from `python:3.11-slim` with `uv` and `gdal`
- Entrypoint: `uvicorn src.web.main:app --host 0.0.0.0 --port ${PORT:-8080}`
- Health check: `GET /api/health`
- Fly.io config in `fly.toml` (app: `pgweather`, region: `arn`)
