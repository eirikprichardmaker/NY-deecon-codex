"""
Run manifest: captures all inputs needed for reproducibility.
Written to runs/<run_id>/manifest.json.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def compute_config_hash(config_path: Path) -> str:
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()[:12]
    except Exception:
        return "unknown"


def write_manifest(
    run_dir: Path,
    asof: str,
    config_path: Path,
    seed: int,
    ticker_count: int,
    data_sources: list[str],
    agent_config: Optional[dict] = None,
) -> Path:
    manifest = {
        "asof_date": asof,
        "config_hash": compute_config_hash(config_path) if config_path.exists() else "missing",
        "git_commit": get_git_commit(),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ticker_count": ticker_count,
        "data_sources": data_sources,
        "agent_config": agent_config or {},
        "python_version": sys.version,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path
