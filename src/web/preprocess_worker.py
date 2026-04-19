"""Batch-job worker — runs preprocess once, persists logs to Supabase, exits.

Deployed on Verda Container Batch Jobs. Each POST to /job spins up a
fresh replica; the replica acks 202, runs MEPS + ICON-EU in a background
task, writes a row to ``preprocess_runs`` (stdout/stderr/exit_code), and
calls ``os._exit`` when done. Verda scales the replica down.

Kept separate from ``main.py`` (the Fly-served web API) so the
customer-facing process never calls ``os._exit``.

Logs are accessible via the ``preprocess_runs`` table (Supabase console
or any psql client) and via the ``GET /last-run`` endpoint — see
``_last_run_query`` below.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="pgweather-preprocess")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_RUNS_TABLE = "preprocess_runs"
_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_RUNS_TABLE} (
    id bigserial PRIMARY KEY,
    started_at timestamptz NOT NULL,
    ended_at timestamptz NOT NULL,
    exit_code int NOT NULL,
    step text NOT NULL,
    stdout_tail text,
    stderr_tail text,
    image_tag text
)
"""
_INSERT_SQL = f"""
INSERT INTO {_RUNS_TABLE}
(started_at, ended_at, exit_code, step, stdout_tail, stderr_tail, image_tag)
VALUES ($1::timestamptz, $2::timestamptz, $3, $4, $5, $6, $7)
"""
_LAST_RUN_SQL = f"""
SELECT started_at, ended_at, exit_code, step, stdout_tail, stderr_tail, image_tag
FROM {_RUNS_TABLE}
ORDER BY id DESC
LIMIT 5
"""


def _db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_DB_URL / DATABASE_URL not set")
    return url


def _log_run(
    started_at: dt.datetime, step: str, proc: subprocess.CompletedProcess
) -> None:
    """Append a row to preprocess_runs. Swallow DB errors — logs are diagnostic only."""
    import adbc_driver_postgresql.dbapi as pg

    try:
        # Run DDL in its own transaction so the table survives even if the
        # subsequent INSERT fails (adbc rolls back the whole tx otherwise).
        with pg.connect(_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
        with pg.connect(_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_SQL,
                    (
                        started_at.isoformat(),
                        dt.datetime.now(dt.timezone.utc).isoformat(),
                        proc.returncode,
                        step,
                        (proc.stdout or "")[-10000:],
                        (proc.stderr or "")[-10000:],
                        os.environ.get("IMAGE_TAG", "unknown"),
                    ),
                )
            conn.commit()
    except Exception:
        logger.exception("Failed to persist preprocess_runs row (continuing)")


def _run_one(cmd: list[str], step: str) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    logger.info("Starting %s: %s (cwd=%s)", step, " ".join(cmd), _REPO_ROOT)
    try:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        # Capture spawn-time failures (FileNotFoundError, PermissionError, ...)
        # as a synthetic CompletedProcess so we still persist a log row.
        logger.exception("subprocess.run raised before the child started")
        proc = subprocess.CompletedProcess(
            args=cmd, returncode=127, stdout="", stderr=f"spawn error: {exc!r}\n"
        )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    logger.info("%s finished exit=%d", step, proc.returncode)
    _log_run(started, step, proc)
    return proc.returncode


def _run_preprocess() -> None:
    # Always record that the BG task fired, even before any subprocess runs,
    # so a failure in env discovery or subprocess spawn is still visible in
    # preprocess_runs (hunting blind was painful).
    startup_started = dt.datetime.now(dt.timezone.utc)
    startup_info = [
        f"python={sys.executable}",
        f"cwd={_REPO_ROOT}",
        f"cwd_listing={sorted(p.name for p in _REPO_ROOT.iterdir())[:60]}",
    ]
    _log_run(
        startup_started,
        "startup",
        subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n".join(startup_info), stderr=""
        ),
    )

    exit_code = 0
    try:
        for cmd, step in (
            ([sys.executable, "preprocess_forecast.py"], "meps"),
            (
                [
                    sys.executable,
                    "preprocess_icon.py",
                    "--model",
                    "icon-eu",
                    "--region",
                    "norway",
                    "--max-hours",
                    "30",
                ],
                "icon-eu",
            ),
        ):
            rc = _run_one(cmd, step)
            if rc != 0:
                exit_code = rc
                logger.error("%s step failed (code=%d) — skipping the rest", step, rc)
                break
    except Exception as exc:
        logger.exception("Uncaught error in _run_preprocess")
        _log_run(
            dt.datetime.now(dt.timezone.utc),
            "fatal",
            subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr=f"{exc!r}"
            ),
        )
        exit_code = 1
    logger.info("All done. Exiting with code %d.", exit_code)
    os._exit(exit_code)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/job")
def job(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Entry point hit by Verda's batch-job dispatcher.

    Acks immediately; the actual preprocess runs after the response is
    flushed, per FastAPI's BackgroundTasks contract.
    """
    background_tasks.add_task(_run_preprocess)
    return {"status": "started"}


@app.get("/last-run")
def last_run() -> JSONResponse:
    """Return the last 5 preprocess runs with stdout/stderr tails.

    Useful fallback for log access when Verda's console doesn't surface
    container logs. Same data is queryable via the ``preprocess_runs``
    Supabase table.
    """
    import adbc_driver_postgresql.dbapi as pg

    rows: list[dict] = []
    try:
        with pg.connect(_db_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(_LAST_RUN_SQL)
                cols = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    rows.append(
                        {
                            c: (v.isoformat() if hasattr(v, "isoformat") else v)
                            for c, v in zip(cols, row)
                        }
                    )
    except Exception as exc:
        return JSONResponse(
            {"error": f"could not query {_RUNS_TABLE}: {exc}"},
            status_code=500,
        )
    return JSONResponse({"runs": rows})
