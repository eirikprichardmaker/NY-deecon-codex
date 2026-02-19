from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.common.config import resolve_paths
from src.common.io import read_csv, read_parquet, write_parquet


def _load_any(path: Path) -> pd.DataFrame:
    return read_parquet(path) if path.suffix.lower() == ".parquet" else read_csv(path)

def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    raw_asof_dir = paths["raw_dir"] / ctx.asof
    processed_dir = paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    cand = list(raw_asof_dir.glob("fundamentals.*"))
    if not cand:
        log.info("TRANSFORM fundamentals: skipped (no fundamentals.* in raw).")
        return 0

    df = _std_cols(_load_any(cand[0]))
    out = processed_dir / "fundamentals.parquet"
    write_parquet(out, df)
    log.info(f"TRANSFORM fundamentals: wrote {out}")
    return 0
