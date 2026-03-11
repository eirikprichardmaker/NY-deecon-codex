"""
Minimal FastAPI service for n8n integration.

Endpoints:
  POST /runs/start             → starter en pipeline-kjøring, returnerer run_id
  GET  /runs/{run_id}/status   → returnerer status (started/running/completed/failed)
  GET  /runs/{run_id}/artifacts → returnerer liste over output-filer

Kjøres med:
  uvicorn src.api:app --host 0.0.0.0 --port 8080

Krever at DEECON_CONFIG og DEECON_RUNS_DIR er satt som miljøvariabler,
eller at standardstier fungerer.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Deecon API", version="1.0")

_RUNS: dict[str, dict] = {}  # In-memory run registry (erstattes med SQLite i prod)
_CONFIG_PATH = os.environ.get("DEECON_CONFIG", "config/config.yaml")
_RUNS_DIR = Path(os.environ.get("DEECON_RUNS_DIR", "runs"))


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    asof_date: str
    config_path: str = _CONFIG_PATH
    steps: str = "all"          # "all" eller kommaseparert liste
    include_agents: bool = False  # True = legg til agents-steg
    callback_url: Optional[str] = None


class RunResponse(BaseModel):
    run_id: str
    status: str


class RunStatus(BaseModel):
    run_id: str
    status: str
    asof_date: str
    return_code: Optional[int] = None
    error: Optional[str] = None


class ArtifactIndex(BaseModel):
    run_id: str
    run_dir: str
    files: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/runs/start", response_model=RunResponse)
async def start_run(req: RunRequest, bg: BackgroundTasks) -> RunResponse:
    """Start en pipeline-kjøring i bakgrunnen."""
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    _RUNS[run_id] = {
        "run_id": run_id,
        "status": "started",
        "asof_date": req.asof_date,
        "return_code": None,
        "error": None,
    }
    bg.add_task(_execute_pipeline, run_id, req)
    return RunResponse(run_id=run_id, status="started")


@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_status(run_id: str) -> RunStatus:
    """Hent kjøringsstatus."""
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail=f"Run {run_id} ikke funnet")
    return RunStatus(**_RUNS[run_id])


@app.get("/runs/{run_id}/artifacts", response_model=ArtifactIndex)
async def get_artifacts(run_id: str) -> ArtifactIndex:
    """List opp alle output-filer for en kjøring."""
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail=f"Run {run_id} ikke funnet")

    run_dir = _find_run_dir(run_id, _RUNS[run_id].get("asof_date", ""))
    if run_dir is None or not run_dir.exists():
        return ArtifactIndex(run_id=run_id, run_dir="", files=[])

    files = sorted(str(f.relative_to(run_dir)) for f in run_dir.iterdir() if f.is_file())
    return ArtifactIndex(run_id=run_id, run_dir=str(run_dir), files=files)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "active_runs": len([r for r in _RUNS.values() if r["status"] == "running"])}


# ---------------------------------------------------------------------------
# Bakgrunns-executor
# ---------------------------------------------------------------------------

def _execute_pipeline(run_id: str, req: RunRequest) -> None:
    """Kjør pipeline som subprocess. Oppdaterer _RUNS med status."""
    _RUNS[run_id]["status"] = "running"

    steps = req.steps
    if req.include_agents and "agents" not in steps:
        steps = steps.rstrip(",") + ",agents" if steps != "all" else steps

    cmd = [
        sys.executable, "-m", "src.run_weekly",
        "--asof", req.asof_date,
        "--config", req.config_path,
    ]
    if steps != "all":
        cmd += ["--steps", steps]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 time maks
        )
        _RUNS[run_id]["return_code"] = result.returncode
        _RUNS[run_id]["status"] = "completed" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            _RUNS[run_id]["error"] = (result.stderr or result.stdout or "")[:500]
    except subprocess.TimeoutExpired:
        _RUNS[run_id]["status"] = "failed"
        _RUNS[run_id]["error"] = "Pipeline timeout etter 3600 sekunder"
    except Exception as e:
        _RUNS[run_id]["status"] = "failed"
        _RUNS[run_id]["error"] = str(e)[:500]

    # Prøv å parse run_dir fra manifest.json
    run_dir = _find_run_dir(run_id, req.asof_date)
    if run_dir:
        _RUNS[run_id]["run_dir"] = str(run_dir)

        # Valider output-kontrakt
        try:
            from src.output_contract import validate_run_outputs
            ok, missing = validate_run_outputs(run_dir)
            if not ok:
                _RUNS[run_id]["missing_outputs"] = missing
        except Exception:
            pass


def _find_run_dir(run_id: str, asof: str) -> Optional[Path]:
    """Finn run-dir basert på asof-dato. Enkel heuristikk."""
    base = _RUNS_DIR / asof if asof else _RUNS_DIR
    if not base.exists():
        return None
    # Finn nyeste mappe som inneholder manifest.json
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "manifest.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
