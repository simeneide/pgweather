"""Batch-job worker — runs preprocess once, then exits the process.

Deployed on Verda Container Batch Jobs. Each POST to /job spins up a
fresh replica; the replica acks 202, runs MEPS + ICON-EU in a background
task, and calls ``os._exit(0)`` when done. Verda notices the replica
terminated and scales it down.

Kept separate from ``main.py`` (the Fly-served web API) so the
customer-facing process never calls ``os._exit``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="pgweather-preprocess")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_preprocess() -> None:
    """Run MEPS then ICON-EU sequentially, then terminate the container.

    Uses ``os._exit`` (not ``sys.exit``) so uvicorn's signal handlers can't
    trap the exit — Verda treats a non-zero exit code as a failed job.
    """
    python = sys.executable
    try:
        logger.info("Starting MEPS preprocess...")
        subprocess.run(
            [python, "preprocess_forecast.py"],
            check=True,
            cwd=_REPO_ROOT,
        )
        logger.info("MEPS done. Starting ICON-EU preprocess...")
        subprocess.run(
            [
                python,
                "preprocess_icon.py",
                "--model",
                "icon-eu",
                "--region",
                "norway",
                "--max-hours",
                "30",
            ],
            check=True,
            cwd=_REPO_ROOT,
        )
        logger.info("Preprocess complete. Exiting 0.")
        os._exit(0)
    except subprocess.CalledProcessError as exc:
        logger.error("Preprocess failed: %s", exc)
        os._exit(1)
    except Exception:
        logger.exception("Unexpected error in preprocess worker")
        os._exit(1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/job")
def job(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Entry point hit by Verda's batch-job dispatcher.

    Returns immediately; the actual preprocess runs after the response is
    flushed, per FastAPI's BackgroundTasks contract.
    """
    background_tasks.add_task(_run_preprocess)
    return {"status": "started"}
