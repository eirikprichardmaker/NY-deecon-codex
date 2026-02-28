from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from src.common.config import load_config, project_root_from_file, resolve_paths
from src.common.io import read_csv, read_parquet, write_parquet
from src.transform_prices import _append_missing_indices, _ensure_price_schema, _filter_non_model_markets


def _load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return read_parquet(path)
    return read_csv(path)


def _resolve_source_path(root: Path, p: str) -> Path:
    pp = Path(str(p).strip())
    if pp.is_absolute():
        return pp
    return (root / pp).resolve()


def _pick_price_source(root: Path, cfg: dict) -> Path:
    prices_cfg = ((cfg.get("sources") or {}).get("prices") or {})
    candidates = []
    for k in ["prices_panel_input", "prices_latest_input"]:
        v = str(prices_cfg.get(k, "")).strip()
        if v:
            candidates.append(_resolve_source_path(root, v))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "No prices source found. Set sources.prices.prices_panel_input or prices_latest_input in config."
    )


def refresh_prices(asof: str, config_path: str, require_fresh_days: int = 3) -> int:
    root = project_root_from_file()
    cfg_path = _resolve_source_path(root, config_path)
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg, root)

    src = _pick_price_source(root, cfg)
    df = _load_any(src)
    df = _ensure_price_schema(df)
    df = _filter_non_model_markets(df, cfg=cfg, log=_StdoutLog())
    df = _append_missing_indices(df, asof=asof, log=_StdoutLog())

    asof_ts = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_ts):
        raise ValueError(f"Invalid --asof date: {asof}")
    df = df[pd.to_datetime(df["date"], errors="coerce") <= asof_ts].copy()
    if df.empty:
        raise ValueError("Price dataframe became empty after as-of filter.")

    max_dt = pd.to_datetime(df["date"], errors="coerce").max()
    lag_days = int((asof_ts.normalize() - max_dt.normalize()).days)
    if require_fresh_days >= 0 and lag_days > require_fresh_days:
        raise RuntimeError(
            f"Prices are stale: max_date={max_dt.date().isoformat()} asof={asof} lag_days={lag_days} > require_fresh_days={require_fresh_days}"
        )

    processed_dir = paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_processed = processed_dir / "prices.parquet"
    write_parquet(out_processed, df)

    raw_asof = paths["raw_dir"] / asof
    raw_asof.mkdir(parents=True, exist_ok=True)
    out_raw = raw_asof / "prices.parquet"
    write_parquet(out_raw, df)

    print(f"REFRESH prices: source={src}")
    print(f"REFRESH prices: rows={len(df)} max_date={max_dt.date().isoformat()} lag_days={lag_days}")
    print(f"REFRESH prices: wrote {out_processed}")
    print(f"REFRESH prices: wrote {out_raw}")
    return 0


class _StdoutLog:
    def info(self, msg: str) -> None:
        print(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--asof", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--require-fresh-days", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return refresh_prices(
        asof=str(args.asof),
        config_path=str(args.config),
        require_fresh_days=int(args.require_fresh_days),
    )


if __name__ == "__main__":
    raise SystemExit(main())

