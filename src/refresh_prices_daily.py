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


def _norm_symbol(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    return s


def _stock_symbol_set(df: pd.DataFrame) -> set[str]:
    if "ticker" not in df.columns:
        return set()
    s = df["ticker"].astype(str).map(_norm_symbol)
    s = s[(s != "") & (~s.str.startswith("^"))]
    return set(s.tolist())


def _coverage_ratio(df: pd.DataFrame, universe: set[str]) -> float:
    if not universe:
        return 1.0
    have = _stock_symbol_set(df)
    return float(len(have & universe) / len(universe))


def _load_universe_tickers(root: Path, cfg: dict) -> set[str]:
    refresh_cfg = cfg.get("refresh_prices_daily", {}) or {}
    prices_cfg = ((cfg.get("sources") or {}).get("prices") or {})

    candidates: list[Path] = []
    explicit = str(
        refresh_cfg.get("universe_tickers_input", "") or prices_cfg.get("universe_tickers_input", "")
    ).strip()
    if explicit:
        candidates.append(_resolve_source_path(root, explicit))

    candidates.extend(
        [
            root / "config" / "tickers.csv",
            root / "config" / "tickers_with_insid_clean.csv",
            root / "config" / "tickers_with_insid.csv",
        ]
    )

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = _load_any(path)
        except Exception:
            continue
        low = {str(c).strip().lower(): c for c in df.columns}
        col = low.get("yahoo_ticker") or low.get("ticker") or low.get("symbol")
        if col is None:
            continue
        tick = df[col].astype(str).map(_norm_symbol)
        tick = tick[(tick != "") & (~tick.str.startswith("^"))]
        out = set(tick.tolist())
        if out:
            return out
    return set()


def _prepare_prices(df: pd.DataFrame, cfg: dict, asof_ts: pd.Timestamp) -> pd.DataFrame:
    out = _ensure_price_schema(df)
    out = _filter_non_model_markets(out, cfg=cfg, log=_StdoutLog())
    out = _append_missing_indices(out, asof=asof_ts.date().isoformat(), log=_StdoutLog())
    out = out[pd.to_datetime(out["date"], errors="coerce") <= asof_ts].copy()
    return out


def refresh_prices(asof: str, config_path: str, require_fresh_days: int = 3) -> int:
    root = project_root_from_file()
    cfg_path = _resolve_source_path(root, config_path)
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg, root)

    asof_ts = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_ts):
        raise ValueError(f"Invalid --asof date: {asof}")

    src = _pick_price_source(root, cfg)
    cand_df = _prepare_prices(_load_any(src), cfg=cfg, asof_ts=asof_ts)

    out_processed = paths["processed_dir"] / "prices.parquet"
    existing_df = None
    if out_processed.exists():
        existing_df = _prepare_prices(_load_any(out_processed), cfg=cfg, asof_ts=asof_ts)
        if existing_df.empty:
            existing_df = None

    refresh_cfg = cfg.get("refresh_prices_daily", {}) or {}
    min_cov = float(refresh_cfg.get("min_universe_coverage", 0.60))
    min_rel_vs_existing = float(refresh_cfg.get("min_relative_coverage_vs_existing", 0.80))
    max_abs_drop = float(refresh_cfg.get("max_absolute_coverage_drop_vs_existing", 0.15))
    universe = _load_universe_tickers(root, cfg)
    cand_cov = _coverage_ratio(cand_df, universe)
    chosen_df = cand_df
    chosen_src = src
    guard_reason = ""

    if existing_df is not None and universe:
        existing_cov = _coverage_ratio(existing_df, universe)
        cov_drop = existing_cov - cand_cov
        severe_drop = (cand_cov < (existing_cov * min_rel_vs_existing)) and (cov_drop > max_abs_drop)
        below_floor = cand_cov < min_cov <= existing_cov
        if severe_drop or below_floor:
            chosen_df = existing_df
            chosen_src = out_processed
            guard_reason = (
                f"coverage_guard_keep_existing "
                f"(candidate={cand_cov:.3f}, existing={existing_cov:.3f}, "
                f"min_cov={min_cov:.3f}, min_rel={min_rel_vs_existing:.3f}, max_drop={max_abs_drop:.3f})"
            )
            print(f"REFRESH prices: {guard_reason}")

    df = chosen_df
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
    write_parquet(out_processed, df)

    raw_asof = paths["raw_dir"] / asof
    raw_asof.mkdir(parents=True, exist_ok=True)
    out_raw = raw_asof / "prices.parquet"
    write_parquet(out_raw, df)

    print(f"REFRESH prices: source={chosen_src}")
    if universe:
        print(f"REFRESH prices: universe_coverage={_coverage_ratio(df, universe):.3f} (n={len(universe)})")
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
