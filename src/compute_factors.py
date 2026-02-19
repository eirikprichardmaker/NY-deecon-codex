from __future__ import annotations

import re
import numpy as np
import pandas as pd

from src.common.config import resolve_paths
from src.common.errors import SchemaError
from src.common.io import read_parquet, write_parquet
from src.common.utils import safe_div, zscore


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def canon(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")
    df.columns = [canon(c) for c in df.columns]
    return df

def _need(df: pd.DataFrame, col: str, where: str):
    if col not in df.columns:
        raise SchemaError(f"{where}: missing '{col}'. available={list(df.columns)[:60]}...")

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    processed = paths["processed_dir"]

    master_path = processed / "master.parquet"
    if not master_path.exists():
        raise SchemaError(f"compute_factors: missing {master_path}")

    df = _canon_cols(read_parquet(master_path))
    _need(df, "ticker", "compute_factors")

    out = df.copy()

    # --- map your fields (after canon) ---
    # market_cap_-_current  -> market_cap_current
    # fcf_-_millions        -> fcf_millions
    # ocf_-_millions        -> ocf_millions
    # capex_-_millions      -> capex_millions
    # net_debt_-_current    -> net_debt_current
    # n.debt/ebitda_-_current -> n_debt_ebitda_current
    # roic_-_current        -> roic_current
    # ev/ebit_-_current     -> ev_ebit_current

    # Required for "real" outputs (best-effort)
    col_mcap = "market_cap_current" if "market_cap_current" in out.columns else None
    col_nd   = "net_debt_current" if "net_debt_current" in out.columns else None
    col_roic = "roic_current" if "roic_current" in out.columns else None

    col_fcf_m = "fcf_millions" if "fcf_millions" in out.columns else None
    col_ocf_m = "ocf_millions" if "ocf_millions" in out.columns else None
    col_capex_m = "capex_millions" if "capex_millions" in out.columns else None

    col_nd_ebitda = "n_debt_ebitda_current" if "n_debt_ebitda_current" in out.columns else None
    col_ev_ebit = "ev_ebit_current" if "ev_ebit_current" in out.columns else None

    # ROIC (already provided)
    out["roic"] = _num(out[col_roic]) if col_roic else np.nan
    out["roic_invalid"] = out["roic"].isna().astype(int)

    # Cash flows: *_millions -> base unit
    scale = float(ctx.cfg.get("cashflow_millions_multiplier", 1_000_000))

    if col_fcf_m:
        fcf = _num(out[col_fcf_m]) * scale
    elif col_ocf_m and col_capex_m:
        fcf = (_num(out[col_ocf_m]) - _num(out[col_capex_m])) * scale
    else:
        fcf = pd.Series(np.nan, index=out.index)

    out["fcf_ttm"] = fcf
    out["fcf_invalid"] = out["fcf_ttm"].isna().astype(int)

    # Net debt / market cap (assumed same unit; add a simple sanity flag)
    mcap = _num(out[col_mcap]) if col_mcap else pd.Series(np.nan, index=out.index)
    nd   = _num(out[col_nd]) if col_nd else pd.Series(np.nan, index=out.index)

    out["market_cap"] = mcap
    out["net_debt"] = nd

    med_mcap = float(np.nanmedian(np.abs(mcap.values))) if np.isfinite(np.nanmedian(np.abs(mcap.values))) else np.nan
    med_nd = float(np.nanmedian(np.abs(nd.values))) if np.isfinite(np.nanmedian(np.abs(nd.values))) else np.nan
    units_mismatch = 0
    if np.isfinite(med_mcap) and np.isfinite(med_nd) and med_mcap > 0 and med_nd > 0:
        ratio = med_mcap / med_nd
        if ratio > 1e3 or ratio < 1e-3:
            units_mismatch = 1
    out["units_mismatch_flag"] = units_mismatch

    # EV proxy
    ev = mcap + nd
    if units_mismatch == 1:
        ev = pd.Series(np.nan, index=out.index)
    out["enterprise_value_proxy"] = ev

    # ND/EBITDA: already provided as ratio
    out["nd_ebitda"] = _num(out[col_nd_ebitda]) if col_nd_ebitda else np.nan
    out["nd_ebitda_invalid"] = out["nd_ebitda"].isna().astype(int)

    # EV/EBIT: already provided as ratio
    out["ev_ebit"] = _num(out[col_ev_ebit]) if col_ev_ebit else np.nan
    out["ev_ebit_invalid"] = out["ev_ebit"].isna().astype(int)

    # Derive EBIT/EBITDA (proxy) when ratios exist
    out["ebit_proxy"] = np.nan
    if col_ev_ebit:
        out["ebit_proxy"] = safe_div(out["enterprise_value_proxy"], out["ev_ebit"])

    out["ebitda_proxy"] = np.nan
    if col_nd_ebitda:
        out["ebitda_proxy"] = safe_div(out["net_debt"], out["nd_ebitda"])

    # FCF yield: prefer EV if available, else market cap
    out["fcf_yield"] = np.nan
    denom = out["enterprise_value_proxy"]
    use_ev = denom.notna() & (denom > 0)
    out.loc[use_ev, "fcf_yield"] = safe_div(out.loc[use_ev, "fcf_ttm"], denom.loc[use_ev])

    use_mcap = out["fcf_yield"].isna() & out["market_cap"].notna() & (out["market_cap"] > 0)
    out.loc[use_mcap, "fcf_yield"] = safe_div(out.loc[use_mcap, "fcf_ttm"], out.loc[use_mcap, "market_cap"])
    out["fcf_yield_invalid"] = out["fcf_yield"].isna().astype(int)

    # GP/A: not available with current master schema
    out["gp_a"] = np.nan
    out["gp_a_invalid"] = 1

    # OSoV score (fallback without GP/A): z(ROIC) + z(FCF-yield) - z(EV/EBIT)
    out["osov_score"] = zscore(out["roic"]) + zscore(out["fcf_yield"]) - zscore(out["ev_ebit"])

    # Save
    out_path = processed / "master_factors.parquet"
    write_parquet(out_path, out)
    log.info(f"compute_factors: wrote {out_path}")
    return 0
