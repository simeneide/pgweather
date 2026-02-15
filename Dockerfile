FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PYTHON=/usr/local/bin/python3.11 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uv run --no-sync uvicorn src.web.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
