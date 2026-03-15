"""
tools/build_historical_master.py
=================================
Option B: Build per-year master_valued snapshots for WFT historical backtesting.

For each year 2016-2025, creates data/snapshots/YYYY/master_valued.parquet
using fundamentals_history.parquet with proper point-in-time cutoffs and
historically-appropriate WACC estimates.

Usage:
    python tools/build_historical_master.py
    python tools/build_historical_master.py --start-year 2017 --end-year 2025
    python tools/build_historical_master.py --out-dir data/snapshots --pub-lag 90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Historical Norwegian 10-year government bond yields (annual approximate avg)
# Source: Norges Bank / market consensus
# ---------------------------------------------------------------------------
HIST_RF_RATE: dict[int, float] = {
    2014: 0.026,
    2015: 0.017,
    2016: 0.015,
    2017: 0.018,
    2018: 0.021,
    2019: 0.016,
    2020: 0.009,
    2021: 0.016,
    2022: 0.035,
    2023: 0.042,
    2024: 0.038,
    2025: 0.039,
}

ERP = 0.050          # Nordic equity risk premium (stable)
DEFAULT_BETA = 1.0   # Market-average beta for companies without historical beta
TERMINAL_GROWTH = 0.02
PUB_LAG_DAYS = 90
THREE_YEAR_NORM_DAYS = 3 * 365
FIVE_YEAR_NORM_DAYS  = 5 * 365

# Damodaran sector labels that trigger a longer FCF normalisation window.
# Matches config/config.yaml cyclicality.cyclical_damodaran_sectors.
CYCLICAL_DAMODARAN_SECTORS: list[str] = [
    "Materials",    # steel, metals, mining, chemicals
    "Oil & Gas",    # energy E&P
    "Shipping",     # bulk, tankers
    "Seafood",      # salmon, white fish
]

# Default metadata path (relative to project root)
DEFAULT_META_PATH = "data/processed/instrument_metadata.parquet"


def _wacc_for_year(year: int, beta: float = DEFAULT_BETA) -> float:
    rf = HIST_RF_RATE.get(year, 0.030)
    return rf + beta * ERP


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper().lstrip("^")
    if "." in s:
        s = s.split(".", 1)[0]
    return s.strip()


def _suffix_from_symbol(x) -> str:
    s = str(x).strip().upper()
    return s.rsplit(".", 1)[-1] if "." in s else ""


_SUFFIX_TO_INDEX = {"OL": "^OSEAX", "ST": "^OMXS", "CO": "^OMXC25", "HE": "^HEX"}


def _to_num(s: "pd.Series | float") -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series([s])


def _to_decimal_rate(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    med = x.abs().median(skipna=True)
    if np.isfinite(med) and med > 2.0:
        x = x / 100.0
    return x


def build_snapshot(
    hist_df: pd.DataFrame,
    year: int,
    pub_lag_days: int = PUB_LAG_DAYS,
    terminal_growth: float = TERMINAL_GROWTH,
    meta_df: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Build a master_valued-equivalent DataFrame for the given year.

    Uses only fundamentals data published before (Dec 31 of year - pub_lag_days),
    so strictly point-in-time.

    meta_df (optional): instrument_metadata DataFrame with damodaran_sector column.
    When provided, cyclical sectors use a 5-year FCF normalisation window.
    """
    dec31 = pd.Timestamp(f"{year}-12-31")
    cutoff = dec31 - pd.Timedelta(days=pub_lag_days)

    h = hist_df.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h[h["date"].notna() & (h["date"] <= cutoff)].copy()
    if h.empty:
        return pd.DataFrame()

    h["k"] = h["yahoo_ticker"].map(_norm_ticker)
    h = h[h["k"].ne("")].copy()

    # ── Sector / cyclicality lookup ──────────────────────────────────────────
    # Build per-k damodaran_sector map from instrument_metadata (if supplied)
    sector_by_k: dict[str, str] = {}
    if meta_df is not None and not meta_df.empty and "damodaran_sector" in meta_df.columns:
        _m = meta_df[["yahoo_ticker", "damodaran_sector"]].dropna(subset=["yahoo_ticker"]).copy()
        _m["k"] = _m["yahoo_ticker"].map(_norm_ticker)
        _m = _m[_m["k"].ne("") & _m["damodaran_sector"].notna()]
        sector_by_k = dict(zip(_m["k"], _m["damodaran_sector"].str.strip()))

    def _is_cyclical(k: str) -> bool:
        return sector_by_k.get(str(k), "") in CYCLICAL_DAMODARAN_SECTORS

    # ── FCF normalisation: 3yr for non-cyclicals, 5yr for cyclicals ──────────
    three_yr_cutoff = cutoff - pd.Timedelta(days=THREE_YEAR_NORM_DAYS)
    five_yr_cutoff  = cutoff - pd.Timedelta(days=FIVE_YEAR_NORM_DAYS)

    fcf_h = h[h["metric"] == "fcf_m"].copy()
    norm_fcf_3yr = (
        fcf_h[fcf_h["date"] >= three_yr_cutoff].groupby("k")["value"].median().rename("norm_fcf_3yr_m")
    )
    norm_fcf_5yr = (
        fcf_h[fcf_h["date"] >= five_yr_cutoff].groupby("k")["value"].median().rename("norm_fcf_5yr_m")
    )

    # Merge both into a combined frame; choose per-ticker based on cyclicality
    _fcf_combined = pd.concat([norm_fcf_3yr, norm_fcf_5yr], axis=1).reset_index()
    _fcf_combined["is_cyclical_flag"] = _fcf_combined["k"].map(_is_cyclical)
    # Use 5yr median for cyclicals when available, else fall back to 3yr
    _fcf_combined["norm_fcf_m"] = np.where(
        _fcf_combined["is_cyclical_flag"] & _fcf_combined["norm_fcf_5yr_m"].notna(),
        _fcf_combined["norm_fcf_5yr_m"],
        _fcf_combined["norm_fcf_3yr_m"],
    )
    norm_fcf = _fcf_combined.set_index("k")["norm_fcf_m"].rename("norm_fcf_m")

    # Latest value per (k, metric) before cutoff
    latest = (
        h.sort_values(["k", "metric", "date"])
        .groupby(["k", "metric"], as_index=False)
        .tail(1)[["k", "metric", "value"]]
    )
    wide = latest.pivot_table(index="k", columns="metric", values="value", aggfunc="last")
    wide.columns.name = None
    wide = wide.reset_index().merge(norm_fcf.reset_index(), on="k", how="left")

    # Yahoo ticker lookup for suffix → index mapping
    yahoo_map = (
        h[["k", "yahoo_ticker"]]
        .drop_duplicates("k")
        .set_index("k")["yahoo_ticker"]
    )
    wide["yahoo_ticker"] = wide["k"].map(yahoo_map)

    def _gc(name: str) -> pd.Series:
        return pd.to_numeric(wide[name], errors="coerce") if name in wide.columns else pd.Series(np.nan, index=wide.index)

    # Core financial metrics
    roic = _to_decimal_rate(_gc("roic")).where(lambda x: x <= 2.0, other=np.nan)

    fcf_m_latest = _gc("fcf_m")
    norm_fcf_m = _gc("norm_fcf_m")
    # Use 3yr median; fall back to latest single-period FCF
    fcf_m = norm_fcf_m.where(norm_fcf_m.notna(), other=fcf_m_latest)

    mcap_m = _gc("mcap_m")
    netdebt_m = _gc("netdebt_m")
    ebitda_m = _gc("ebitda_m")
    ev_ebit = _gc("ev_ebit")
    gross_profit_m = _gc("gross_profit_m")
    total_assets_m = _gc("total_assets_m")

    nd_ebitda = _gc("netdebt_ebitda")
    computed_nd = (netdebt_m / ebitda_m.where(ebitda_m.abs() > 0))
    nd_ebitda = nd_ebitda.where(nd_ebitda.notna(), other=computed_nd)

    market_cap = (mcap_m * 1_000_000.0).where(mcap_m.gt(0))
    fcf_yield = (fcf_m / mcap_m.where(mcap_m.gt(0))).where(lambda x: x <= 0.50)
    gp_a = (gross_profit_m / total_assets_m.where(total_assets_m.gt(0)))

    # Historically-appropriate WACC
    wacc_val = _wacc_for_year(year)
    wacc = pd.Series(wacc_val, index=wide.index)

    # Simplified DCF → intrinsic equity + MoS
    # Require fcf_yield to be positive and non-null: companies whose 3yr-median FCF
    # exceeds 50% of market cap get fcf_yield=NaN (cap) and are excluded from DCF MoS.
    # This prevents cyclical-peak-FCF value traps (e.g. steel companies after a boom).
    valid_dcf = fcf_m.gt(0) & mcap_m.gt(0) & fcf_yield.notna() & fcf_yield.gt(0)
    iv_ev_m = fcf_m / (wacc - terminal_growth)
    iv_eq_m = iv_ev_m - netdebt_m.fillna(0.0)
    intrinsic_equity = (iv_eq_m * 1_000_000.0).where(valid_dcf)
    mos = (iv_eq_m / mcap_m - 1.0).where(valid_dcf)

    # Risk / quality flags
    high_risk_flag = nd_ebitda.fillna(0) >= 3.5

    weak_roic = roic.isna() | (roic <= 0.0)
    weak_fcf = fcf_yield.isna() | (fcf_yield <= 0.0)
    weak_nd = nd_ebitda.isna() | (nd_ebitda > 3.5)
    weak_ev = ev_ebit.isna() | (ev_ebit <= 0) | (ev_ebit > 20.0)
    quality_weak_count = (
        weak_roic.astype(int) + weak_fcf.astype(int)
        + weak_nd.astype(int) + weak_ev.astype(int)
    )
    value_creation_ok_base = roic.notna() & (roic > wacc)

    # Index mapping via suffix
    suffix = wide["yahoo_ticker"].map(_suffix_from_symbol)
    rel_idx_sym = suffix.map(_SUFFIX_TO_INDEX).fillna("")
    rel_idx_key = rel_idx_sym.str.lstrip("^")

    # Cyclicality flags for downstream guards
    is_cyclical_series = wide["k"].map(_is_cyclical).fillna(False).astype(bool)
    sector_series = wide["k"].map(lambda k: sector_by_k.get(str(k), "")).fillna("")

    snap = pd.DataFrame({
        "ticker": wide["k"],
        "yahoo_ticker": wide["yahoo_ticker"],
        "company": pd.Series("", index=wide.index),
        "intrinsic_equity": intrinsic_equity,
        "market_cap": market_cap,
        "mos": mos,
        "high_risk_flag": high_risk_flag.astype(bool),
        "quality_weak_count": quality_weak_count.astype(int),
        "value_creation_ok_base": value_creation_ok_base.astype(bool),
        "roic": roic,
        "fcf_yield": fcf_yield,
        "gp_a": gp_a,
        "nd_ebitda": nd_ebitda,
        "ev_ebit": ev_ebit,
        "wacc_used": wacc,
        "relevant_index_symbol": rel_idx_sym,
        "relevant_index_key": rel_idx_key,
        "is_bank_proxy": pd.Series(False, index=wide.index),
        "is_cyclical": is_cyclical_series,
        "sector": sector_series,
        "model": pd.Series("dcf_pit", index=wide.index),
    })
    snap = snap.dropna(subset=["ticker"]).reset_index(drop=True)
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hist-path",
        default="data/raw/2026-03-14/fundamentals_history.parquet",
        help="Path to fundamentals_history.parquet",
    )
    ap.add_argument("--out-dir", default="data/snapshots", help="Output directory for snapshots")
    ap.add_argument("--start-year", type=int, default=2016)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--pub-lag", type=int, default=PUB_LAG_DAYS, help="Publication lag days")
    ap.add_argument(
        "--meta-path",
        default=DEFAULT_META_PATH,
        help="Path to instrument_metadata.parquet (for cyclicality flags)",
    )
    args = ap.parse_args()

    hist_path = Path(args.hist_path)
    if not hist_path.exists():
        print(f"ERROR: {hist_path} not found", file=sys.stderr)
        sys.exit(1)

    hist_df = pd.read_parquet(hist_path)
    print(f"Loaded fundamentals_history: {len(hist_df):,} rows, "
          f"{hist_df['yahoo_ticker'].nunique()} tickers, "
          f"{pd.to_datetime(hist_df['date'], errors='coerce').min().date()} – "
          f"{pd.to_datetime(hist_df['date'], errors='coerce').max().date()}")

    meta_df: "pd.DataFrame | None" = None
    meta_path = Path(args.meta_path)
    if meta_path.exists():
        meta_df = pd.read_parquet(meta_path)
        n_cyclical = meta_df["damodaran_sector"].isin(CYCLICAL_DAMODARAN_SECTORS).sum() if "damodaran_sector" in meta_df.columns else 0
        print(f"Loaded instrument_metadata: {len(meta_df):,} rows, {n_cyclical} cyclical sector entries")
    else:
        print(f"WARNING: {meta_path} not found — cyclicality flags will be False for all tickers")
    print()

    out_root = Path(args.out_dir)
    for year in range(args.start_year, args.end_year + 1):
        snap = build_snapshot(hist_df, year, pub_lag_days=args.pub_lag, meta_df=meta_df)
        if snap.empty:
            print(f"  {year}: SKIP — no data before cutoff")
            continue

        out_dir = out_root / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "master_valued.parquet"
        snap.to_parquet(out_path, index=False)

        n_mos_pos = int((snap["mos"] > 0).sum())
        n_roic_pos = int((snap["roic"] > 0).sum()) if snap["roic"].notna().any() else 0
        n_vc_ok = int(snap["value_creation_ok_base"].sum())
        n_cyc = int(snap["is_cyclical"].sum()) if "is_cyclical" in snap.columns else 0
        wacc_val = _wacc_for_year(year)
        print(
            f"  {year}: {len(snap):4d} companies  "
            f"MoS>0: {n_mos_pos:3d}  ROIC>0: {n_roic_pos:3d}  "
            f"VC_ok: {n_vc_ok:3d}  Cyclical: {n_cyc:3d}  WACC: {wacc_val:.1%}  -> {out_path}"
        )

    print("\nDone. Run WFT with --snapshots-dir data/snapshots")


if __name__ == "__main__":
    main()
