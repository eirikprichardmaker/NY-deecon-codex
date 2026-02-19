from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import resolve_paths
from src.common.errors import SchemaError


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)


def _atomic_write_text(dst: Path, text: str) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dst)


def run(ctx, log) -> int:
    """
    Reads:  data/processed/fundamentals.parquet
    Writes: data/golden/fundamentals.parquet
            data/golden/history/fundamentals_<ASOF>.parquet
            data/golden/manifest_fundamentals.json
    """
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    processed = paths["processed_dir"]

    src_path = processed / "fundamentals.parquet"
    if not src_path.exists():
        raise SchemaError(f"freeze_golden: missing {src_path} (run transform_fundamentals first)")

    golden_dir = Path(ctx.cfg.get("golden_dir", "data/golden"))
    if not golden_dir.is_absolute():
        golden_dir = ctx.project_root / golden_dir

    hist_dir = golden_dir / "history"
    golden_dir.mkdir(parents=True, exist_ok=True)
    hist_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src_path)
    rows = int(len(df))
    cols = int(len(df.columns))

    latest_path = golden_dir / "fundamentals.parquet"
    hist_path = hist_dir / f"fundamentals_{ctx.asof}.parquet"

    # write parquet atomically via bytes buffer
    buf_latest = df.to_parquet(index=False)
    _atomic_write_bytes(latest_path, buf_latest)

    buf_hist = df.to_parquet(index=False)
    _atomic_write_bytes(hist_path, buf_hist)

    manifest = {
        "kind": "golden_fundamentals",
        "asof": ctx.asof,
        "run_id": ctx.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(src_path),
        "latest": str(latest_path),
        "history": str(hist_path),
        "sha256_latest": _sha256_file(latest_path),
        "rows": rows,
        "cols": cols,
        "columns": list(df.columns),
    }

    manifest_path = golden_dir / "manifest_fundamentals.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    log.info(f"freeze_golden: wrote {latest_path}")
    log.info(f"freeze_golden: wrote {hist_path}")
    log.info(f"freeze_golden: wrote {manifest_path}")
    return 0
