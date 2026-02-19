# src/common/config.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml  # pip install pyyaml

@dataclass(frozen=True)
class RunContext:
    asof: str
    project_root: Path
    config_path: Path
    run_id: str
    run_dir: Path
    cfg: Dict[str, Any]

def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = load_yaml(config_path)

    includes = cfg.get("includes", {}) or {}
    # forventet: includes: {sources: "...", thresholds: "..."}
    for key, rel in includes.items():
        inc_path = (config_path.parent.parent / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if inc_path.exists():
            cfg = deep_merge(cfg, {key: load_yaml(inc_path)})
        else:
            # ikke hard-fail her; ingest kan feile hvis den trenger det
            cfg = deep_merge(cfg, {key: {"_missing_path": str(inc_path)}})

    return cfg

def make_run_id(asof: str) -> str:
    ts = time.strftime("%H%M%S")
    return f"{asof.replace('-', '')}_{ts}"

def resolve_paths(cfg: Dict[str, Any], root: Path) -> Dict[str, Path]:
    paths = cfg.get("paths", {}) or {}
    runs_dir = root / paths.get("runs_dir", "runs")
    data_dir = root / paths.get("data_dir", "data")
    raw_dir = root / paths.get("raw_dir", str(data_dir / "raw"))
    processed_dir = root / paths.get("processed_dir", str(data_dir / "processed"))
    return {
        "runs_dir": Path(runs_dir),
        "data_dir": Path(data_dir),
        "raw_dir": Path(raw_dir),
        "processed_dir": Path(processed_dir),
    }

def build_run_context(asof: str, config_path: str, run_dir: Optional[str] = None) -> RunContext:
    root = project_root_from_file()
    cfg_path = (root / config_path).resolve()
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg, root)

    run_id = make_run_id(asof)
    rd = Path(run_dir) if run_dir else (paths["runs_dir"] / run_id)
    rd = (root / rd).resolve() if not rd.is_absolute() else rd
    rd.mkdir(parents=True, exist_ok=True)

    manifest = {
        "asof": asof,
        "run_id": run_id,
        "config_path": str(cfg_path),
    }
    (rd / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return RunContext(
        asof=asof,
        project_root=root,
        config_path=cfg_path,
        run_id=run_id,
        run_dir=rd,
        cfg=cfg,
    )
