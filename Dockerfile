FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.web.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
