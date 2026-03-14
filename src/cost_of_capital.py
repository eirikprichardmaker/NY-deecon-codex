"""
cost_of_capital.py — per-company WACC estimation

Beta hierarchy (runtime, per asof):
  T1: OLS monthly log-returns vs country index (≥24m required, ≥36m preferred)
  T2: Damodaran sector unlevered beta re-levered with company D/E
      — placeholder: requires sector column in master (currently not populated)
  T3: Country-median of T1 betas computed from universe
  T4: Nordic total-median — WARN + beta_low_confidence flag

COE  = rf_country + beta × erp
CoD  = rf_country + cod_spread(ND/EBITDA bucket)
WACC = (E/V) × COE + (D/V) × CoD × (1 − tax_rate)
       clamped to [wacc_min, wacc_max]

Country → benchmark index:
  Norway  → ^OSEAX
  Sweden  → ^OMXS
  Denmark → ^OMXC25
  Finland → ^HEX
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import resolve_paths
from src.common.errors import SchemaError
from src.common.io import read_parquet, write_parquet
from src.common.utils import safe_div

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Country → index ticker
# ---------------------------------------------------------------------------
_COUNTRY_INDEX: dict[str, str] = {
    "Norway":  "^OSEAX",
    "Sweden":  "^OMXS",
    "Denmark": "^OMXC25",
    "Finland": "^HEX",
}

# ---------------------------------------------------------------------------
# Damodaran unlevered sector betas (Europe, Jan 2024)
# Used when sector column is available — currently a placeholder.
# Re-lever with: beta_levered = beta_unlevered × (1 + (1-t) × D/E)
# ---------------------------------------------------------------------------
_DAMODARAN_UNLEVERED: dict[str, float] = {
    "Technology":             0.88,
    "Software":               0.88,
    "Healthcare":             0.68,
    "Pharmaceuticals":        0.68,
    "Financial Services":     0.38,
    "Banking":                0.28,
    "Insurance":              0.48,
    "Real Estate":            0.45,
    "Energy":                 0.85,
    "Oil & Gas":              0.85,
    "Materials":              0.72,
    "Industrials":            0.72,
    "Consumer Staples":       0.58,
    "Consumer Discretionary": 0.78,
    "Utilities":              0.38,
    "Telecom":                0.62,
    "Shipping":               0.90,
    "Seafood":                0.65,
    "Default":                0.75,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _pick(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _need(df: pd.DataFrame, col: str, where: str) -> None:
    if col not in df.columns:
        raise SchemaError(
            f"{where}: missing required column '{col}'. "
            f"available={list(df.columns)[:60]}..."
        )


def _cfg_coc(full_cfg: dict) -> dict:
    """Return the cost_of_capital sub-config with safe defaults."""
    return full_cfg.get("cost_of_capital") or {}


def _rf_for_country(country: str | None, coc: dict) -> float:
    """Return risk-free rate for a country based on rf_mode config."""
    rf_mode = coc.get("rf_mode", "country")
    if rf_mode == "nordic_single":
        return float(coc.get("nordic_single_rf", 0.035))
    country_rf: dict = coc.get("country_rf") or {}
    fallback = float(coc.get("nordic_single_rf", 0.035))
    if not country:
        return fallback
    return float(country_rf.get(country, fallback))


def _tax_for_country(country: str | None, coc: dict) -> float:
    country_tax: dict = coc.get("country_tax") or {}
    fallback = float(coc.get("tax_rate_default", 0.22))
    if not country:
        return fallback
    return float(country_tax.get(country, fallback))


def _cod_spread(
    nd_ebitda: float | None,
    net_debt: float | None,
    spreads: dict,
) -> tuple[float, str]:
    """
    Return (spread_above_rf, bucket_name).
    Uses net_debt sign as primary signal; nd_ebitda for bucket selection.
    """
    # Net cash → minimal debt risk
    if net_debt is not None and np.isfinite(net_debt) and net_debt < 0:
        return float(spreads.get("net_cash", 0.010)), "net_cash"

    if nd_ebitda is None or not np.isfinite(nd_ebitda):
        return float(spreads.get("no_data", 0.040)), "no_data"

    # Negative ND/EBITDA with positive net debt → negative EBITDA, debt exists → high risk
    if nd_ebitda < 0:
        return float(spreads.get("nd_ebitda_4_6", 0.050)), "nd_neg_ebitda"

    if nd_ebitda <= 1.0:
        return float(spreads.get("nd_ebitda_0_1", 0.015)), "nd_ebitda_0_1"
    if nd_ebitda <= 2.0:
        return float(spreads.get("nd_ebitda_1_2", 0.020)), "nd_ebitda_1_2"
    if nd_ebitda <= 3.0:
        return float(spreads.get("nd_ebitda_2_3", 0.025)), "nd_ebitda_2_3"
    if nd_ebitda <= 4.0:
        return float(spreads.get("nd_ebitda_3_4", 0.035)), "nd_ebitda_3_4"
    if nd_ebitda <= 6.0:
        return float(spreads.get("nd_ebitda_4_6", 0.050)), "nd_ebitda_4_6"
    return float(spreads.get("nd_ebitda_gt6", 0.075)), "nd_ebitda_gt6"


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------
def _load_prices(paths: dict, asof_ts: pd.Timestamp | None) -> pd.DataFrame:
    """
    Load price history for beta computation.
    Prefers data/golden/prices_wft_combined.parquet (longer history) over
    data/processed/prices.parquet.
    Returns DataFrame: ticker (Yahoo format), date (datetime), adj_close.
    """
    data_dir = Path(paths.get("data_dir", "data"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))

    golden = data_dir / "golden" / "prices_wft_combined.parquet"
    regular = processed_dir / "prices.parquet"

    if golden.exists():
        px = read_parquet(golden)
        logger.debug("cost_of_capital: using golden prices for beta")
    elif regular.exists():
        px = read_parquet(regular)
        logger.debug("cost_of_capital: using processed prices for beta")
    else:
        raise SchemaError(
            f"cost_of_capital: no prices file found "
            f"(tried {golden}, {regular})"
        )

    px = _std_cols(px)
    date_col = _pick(px, ["date", "dato", "time", "timestamp"])
    price_col = _pick(px, ["adj_close", "close", "price", "last"])
    if not date_col or not price_col:
        raise SchemaError(
            f"prices file must have date+price columns. "
            f"cols={list(px.columns)[:80]}"
        )

    px = px.rename(columns={date_col: "date", price_col: "adj_close"})
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date", "ticker", "adj_close"])

    if asof_ts is not None:
        px = px[px["date"] <= asof_ts]

    return px[["ticker", "date", "adj_close"]].copy()


def _load_ticker_map(raw_dir: Path) -> dict[str, str]:
    """
    Load the most recent ticker_map.csv and return {yahoo_ticker → short_ticker}.
    Falls back to stripping the exchange suffix if no map found.
    """
    import glob as _glob
    maps = sorted(_glob.glob(str(raw_dir / "*" / "ticker_map.csv")))
    if not maps:
        return {}
    df = pd.read_csv(maps[-1])
    df.columns = [c.strip().lower() for c in df.columns]
    if "yahoo_ticker" not in df.columns or "ticker" not in df.columns:
        return {}
    return dict(zip(df["yahoo_ticker"].dropna(), df["ticker"].dropna()))


def _normalize_price_tickers(
    px: pd.DataFrame,
    yahoo_to_short: dict[str, str],
    index_tickers: set[str],
) -> pd.DataFrame:
    """
    Map yahoo_ticker format to short ticker format used in master.
    Index tickers (^OSEAX etc.) are kept as-is.
    Falls back to stripping exchange suffix (e.g. 'EQNR.OL' → 'EQNR').
    """
    def _map(t: str) -> str:
        if t in index_tickers:
            return t
        if t in yahoo_to_short:
            return yahoo_to_short[t]
        # Fallback: strip exchange suffix
        return t.split(".")[0] if "." in t else t

    px = px.copy()
    px["ticker"] = px["ticker"].map(_map)
    return px


# ---------------------------------------------------------------------------
# Monthly log-returns
# ---------------------------------------------------------------------------
def _monthly_log_returns(px: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to month-end, compute log-returns.
    Returns: ticker, date (month-end), ret.
    """
    out: list[pd.DataFrame] = []
    for t, g in px.groupby("ticker", sort=False):
        s = g.set_index("date")["adj_close"].sort_index()
        monthly = s.resample("ME").last().dropna()
        if len(monthly) < 2:
            continue
        log_ret = np.log(monthly / monthly.shift(1)).dropna()
        if log_ret.empty:
            continue
        out.append(pd.DataFrame({
            "ticker": t,
            "date": log_ret.index,
            "ret": log_ret.values,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["ticker", "date", "ret"]
    )


# ---------------------------------------------------------------------------
# Beta computation
# ---------------------------------------------------------------------------
def _compute_betas(
    monthly_rets: pd.DataFrame,
    ticker_country: pd.Series,   # ticker → country string
    country_index: dict[str, str],
    lookback_months: int,
    warn_months: int,
    fail_months: int,
    clamp_low: float,
    clamp_high: float,
    asof_ts: pd.Timestamp,
    log,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """
    Compute OLS betas.

    Returns:
        beta_map:    ticker → float | None
        beta_source: ticker → 'T1_full' | 'T1_warn' | '' (insufficient)
    """
    # Build month-end index return series per country index ticker
    idx_rets: dict[str, pd.Series] = {}
    for idx_ticker in set(country_index.values()):
        rows = monthly_rets[monthly_rets["ticker"] == idx_ticker]
        if not rows.empty:
            idx_rets[idx_ticker] = rows.set_index("date")["ret"].sort_index()
        else:
            logger.warning(
                f"cost_of_capital: index {idx_ticker} not in monthly returns"
            )

    if not idx_rets:
        logger.error("cost_of_capital: no index return series — all betas NaN")
        return {}, {}

    # Window start
    window_start = asof_ts - pd.DateOffset(months=lookback_months)

    beta_map: dict[str, float | None] = {}
    beta_source: dict[str, str] = {}

    idx_tickers_set = set(country_index.values())
    all_tickers = [t for t in monthly_rets["ticker"].unique() if t not in idx_tickers_set]

    for t in all_tickers:
        _country_raw = ticker_country.get(t)
        country = str(_country_raw) if (_country_raw and pd.notna(_country_raw)) else ""
        preferred_idx = country_index.get(country) if country else None

        stock_ret = (
            monthly_rets[monthly_rets["ticker"] == t]
            .set_index("date")["ret"]
            .sort_index()
        )
        stock_windowed = stock_ret[stock_ret.index >= window_start]

        # Select best index: prefer country index, else max overlap
        best_beta: float | None = None
        best_n = 0
        best_source_tag = ""

        candidates = (
            [preferred_idx] + [k for k in idx_rets if k != preferred_idx]
            if preferred_idx and preferred_idx in idx_rets
            else list(idx_rets.keys())
        )

        for idx_ticker in candidates:
            mkt_ret = idx_rets[idx_ticker]
            mkt_windowed = mkt_ret[mkt_ret.index >= window_start]

            # Inner join on dates
            aligned = pd.DataFrame({"s": stock_windowed, "m": mkt_windowed}).dropna()
            n = len(aligned)

            if n <= best_n:
                continue  # not better

            if n < fail_months:
                # Even this index doesn't give enough observations
                continue

            var_m = float(aligned["m"].var())
            if var_m == 0 or not np.isfinite(var_m):
                continue

            cov_val = float(np.cov(aligned["s"].values, aligned["m"].values)[0, 1])
            if not np.isfinite(cov_val):
                continue

            raw_beta = cov_val / var_m
            best_beta = float(np.clip(raw_beta, clamp_low, clamp_high))
            best_n = n
            best_source_tag = "T1_full" if n >= lookback_months else "T1_warn"

        beta_map[t] = best_beta
        if best_beta is not None:
            beta_source[t] = best_source_tag
            if best_source_tag == "T1_warn":
                logger.debug(
                    f"cost_of_capital: {t} beta computed with {best_n}m "
                    f"(< {lookback_months}m preferred) — WARN"
                )

    n_full = sum(1 for v in beta_source.values() if v == "T1_full")
    n_warn = sum(1 for v in beta_source.values() if v == "T1_warn")
    n_none = sum(1 for v in beta_map.values() if v is None)
    log.info(
        f"cost_of_capital: beta T1_full={n_full}, T1_warn={n_warn}, "
        f"insufficient_history={n_none}"
    )

    return beta_map, beta_source


# ---------------------------------------------------------------------------
# Damodaran re-levering (placeholder — not applied until sector data exists)
# ---------------------------------------------------------------------------
def _damodaran_levered_beta(
    sector: str | None,
    de_ratio: float,
    tax: float,
) -> float | None:
    """Re-lever Damodaran unlevered beta: β_L = β_U × (1 + (1-t) × D/E)."""
    if not sector:
        return None
    bu = _DAMODARAN_UNLEVERED.get(sector) or _DAMODARAN_UNLEVERED.get("Default")
    if bu is None:
        return None
    levered = bu * (1 + (1 - tax) * max(de_ratio, 0.0))
    return float(np.clip(levered, 0.1, 3.0))


# ---------------------------------------------------------------------------
# Main pipeline step
# ---------------------------------------------------------------------------
def run(ctx, log) -> int:
    """
    Compute per-company rf, erp, beta, coe, wacc and write master_cost.parquet.
    """
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    processed = Path(paths["processed_dir"])

    # Load master fundamentals
    master_path = processed / "master_factors.parquet"
    if not master_path.exists():
        master_path = processed / "master.parquet"
    if not master_path.exists():
        raise SchemaError(
            f"cost_of_capital: missing master input "
            f"({processed}/master*.parquet)"
        )

    m = _std_cols(read_parquet(master_path))
    _need(m, "ticker", "cost_of_capital")

    coc = _cfg_coc(ctx.cfg)
    erp = float(coc.get("erp", ctx.cfg.get("equity_risk_premium", 0.05)))
    lookback = int(coc.get("beta_lookback_months", 36))
    warn_months = int(coc.get("beta_warn_months", 24))
    fail_months = int(coc.get("beta_fail_months", 12))
    clamp_low = float(coc.get("beta_clamp_low", 0.3))
    clamp_high = float(coc.get("beta_clamp_high", 2.5))
    wacc_min = float(coc.get("wacc_min", 0.06))
    wacc_max = float(coc.get("wacc_max", 0.18))
    wacc_fallback_val = float(coc.get("wacc_fallback", ctx.cfg.get("wacc_fallback", 0.09)))
    spreads: dict = coc.get("cod_spreads") or {}

    asof_ts = pd.Timestamp(ctx.asof) if ctx.asof else pd.Timestamp.now()

    # -----------------------------------------------------------------------
    # Country and sector metadata
    # Merge instrument_metadata (sector, branch, damodaran_sector) if available
    # -----------------------------------------------------------------------
    meta_path = processed / "instrument_metadata.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)[
            ["ticker", "sector_en", "branch_se", "damodaran_sector",
             "market", "country"]
        ].drop_duplicates(subset="ticker")
        # Keep country from master (more reliable for our pipeline) —
        # only pull in sector columns that master lacks
        m = m.merge(
            meta[["ticker", "sector_en", "branch_se", "damodaran_sector", "market"]],
            on="ticker", how="left",
        )
        log.info(
            f"cost_of_capital: instrument_metadata merged — "
            f"sector coverage "
            f"{m['sector_en'].notna().sum()}/{len(m)}"
        )
    else:
        log.warning(
            "cost_of_capital: instrument_metadata.parquet not found — "
            "run tools/fetch_borsdata_metadata.py to enable T2/T3-sector betas"
        )
        m["sector_en"] = None
        m["branch_se"] = None
        m["damodaran_sector"] = None
        m["market"] = None

    country_col = _pick(m, ["info_country", "country"])
    sector_col = "damodaran_sector"   # always present after merge above

    # Deduplicate before indexing (5 tickers appear twice in master)
    m_dedup = m.drop_duplicates(subset="ticker")
    ticker_country: pd.Series = (
        m_dedup.set_index("ticker")[country_col]
        if country_col
        else pd.Series(dtype=str)
    )
    ticker_sector: pd.Series = m_dedup.set_index("ticker")["damodaran_sector"]

    # -----------------------------------------------------------------------
    # Load prices and compute monthly log-returns
    # -----------------------------------------------------------------------
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    yahoo_to_short = _load_ticker_map(raw_dir)
    index_tickers_set = set(_COUNTRY_INDEX.values())
    log.info(
        f"cost_of_capital: ticker_map loaded "
        f"({len(yahoo_to_short)} entries)"
    )

    try:
        px = _load_prices(paths, asof_ts)
        px = _normalize_price_tickers(px, yahoo_to_short, index_tickers_set)
    except SchemaError as e:
        log.warning(f"cost_of_capital: {e} — betas set NaN, WACC = fallback")
        px = pd.DataFrame(columns=["ticker", "date", "adj_close"])

    monthly_rets = _monthly_log_returns(px)

    # -----------------------------------------------------------------------
    # T1 beta: OLS from price history
    # -----------------------------------------------------------------------
    beta_map: dict[str, float | None] = {}
    beta_source_map: dict[str, str] = {}

    if not monthly_rets.empty:
        beta_map, beta_source_map = _compute_betas(
            monthly_rets=monthly_rets,
            ticker_country=ticker_country,
            country_index=_COUNTRY_INDEX,
            lookback_months=lookback,
            warn_months=warn_months,
            fail_months=fail_months,
            clamp_low=clamp_low,
            clamp_high=clamp_high,
            asof_ts=asof_ts,
            log=log,
        )

    # -----------------------------------------------------------------------
    # T3: country+sector-median of T1 betas
    # T4: Nordic total-median
    # -----------------------------------------------------------------------
    t1_betas = pd.Series({t: v for t, v in beta_map.items() if v is not None})

    # Country medians from T1
    country_median: dict[str, float] = {}
    if not t1_betas.empty and country_col:
        for c in ticker_country.dropna().unique():
            tickers_in_country = ticker_country[ticker_country == c].index
            vals = t1_betas.reindex(tickers_in_country).dropna()
            if not vals.empty:
                country_median[c] = float(vals.median())

    # Country × sector medians from T1
    country_sector_median: dict[tuple[str, str], float] = {}
    if not t1_betas.empty and not ticker_sector.empty:
        combined = pd.DataFrame({
            "beta": t1_betas,
            "country": ticker_country,
            "sector": ticker_sector,
        }).dropna(subset=["beta", "country", "sector"])
        for (c, s), grp in combined.groupby(["country", "sector"]):
            if len(grp) >= 3:   # min 3 T1 betas to form a sector median
                country_sector_median[(c, s)] = float(grp["beta"].median())

    nordic_median = float(t1_betas.median()) if not t1_betas.empty else 1.0

    log.info(
        f"cost_of_capital: T1 betas available={len(t1_betas)}, "
        f"country medians={dict((k, round(v,2)) for k, v in country_median.items())}, "
        f"country×sector medians={len(country_sector_median)} buckets, "
        f"nordic_median={nordic_median:.2f}"
    )

    # -----------------------------------------------------------------------
    # Build output row by row
    # -----------------------------------------------------------------------
    out = m.copy()

    rf_list: list[float] = []
    erp_list: list[float] = []
    beta_list: list[float] = []
    beta_source_list: list[str] = []
    beta_low_conf_list: list[bool] = []
    coe_list: list[float] = []
    cod_list: list[float] = []
    cod_bucket_list: list[str] = []
    wacc_list: list[float] = []
    wacc_source_list: list[str] = []

    col_mcap = _pick(out, ["market_cap", "marketcap", "equity_value"])
    col_nd = _pick(out, ["net_debt_current", "net_debt", "t2_net_debt"])
    col_nd_ebitda = _pick(out, ["n_debt_ebitda_current", "nd_ebitda", "net_debt_ebitda"])

    for _, row in out.iterrows():
        ticker = str(row["ticker"])
        country = str(row.get(country_col, "") or "") if country_col else ""
        sector = str(row.get(sector_col, "") or "") if sector_col else ""

        rf = _rf_for_country(country or None, coc)
        tax = _tax_for_country(country or None, coc)

        # --- Beta resolution ---
        raw_beta = beta_map.get(ticker)
        src = beta_source_map.get(ticker, "")
        low_conf = False

        if raw_beta is None:
            dam_sector = (
                str(row.get("damodaran_sector") or "")
                if "damodaran_sector" in out.columns else ""
            )

            # T2: Damodaran unlevered beta re-levered with company D/E
            if dam_sector and dam_sector not in ("", "nan", "Default"):
                nd_raw = row.get(col_nd) if col_nd else None
                mcap_raw = row.get(col_mcap) if col_mcap else None
                nd_scalar = float(pd.to_numeric(nd_raw, errors="coerce") or 0)
                mcap_scalar = float(pd.to_numeric(mcap_raw, errors="coerce") or 0)
                D_val = max(nd_scalar, 0.0)
                E_val = max(mcap_scalar, 0.0)
                de_ratio = D_val / E_val if E_val > 0 else 0.0
                t2_beta = _damodaran_levered_beta(dam_sector, de_ratio, tax)
                if t2_beta is not None:
                    raw_beta = float(np.clip(t2_beta, clamp_low, clamp_high))
                    src = f"T2_damodaran({dam_sector})"
                    low_conf = True
                    logger.debug(
                        f"cost_of_capital: {ticker} beta → T2 Damodaran "
                        f"sector={dam_sector} D/E={de_ratio:.2f} β={raw_beta:.2f}"
                    )

            # T3a: country × sector median
            if raw_beta is None:
                cs_key = (country, dam_sector) if country and dam_sector else None
                if cs_key and cs_key in country_sector_median:
                    raw_beta = country_sector_median[cs_key]
                    src = f"T3_country_sector_median({country},{dam_sector})"
                    low_conf = True
                    logger.debug(
                        f"cost_of_capital: {ticker} beta → T3a "
                        f"country×sector {cs_key}={raw_beta:.2f}"
                    )

            # T3b: country median
            if raw_beta is None and country and country in country_median:
                raw_beta = country_median[country]
                src = "T3_country_median"
                low_conf = True
                logger.debug(
                    f"cost_of_capital: {ticker} beta → T3b country_median "
                    f"({country}={raw_beta:.2f})"
                )

            # T4: Nordic total-median
            if raw_beta is None:
                raw_beta = nordic_median
                src = "T4_nordic_median"
                low_conf = True
                logger.warning(
                    f"cost_of_capital: {ticker} beta → T4 nordic_median "
                    f"({nordic_median:.2f}) — WARN low_confidence"
                )

        beta_val = float(raw_beta)
        coe_val = float(np.clip(rf + beta_val * erp, wacc_min, wacc_max))

        # --- Cost of debt ---
        net_debt_val = (
            float(pd.to_numeric(row.get(col_nd), errors="coerce"))
            if col_nd else None
        )
        nd_ebitda_val = (
            float(pd.to_numeric(row.get(col_nd_ebitda), errors="coerce"))
            if col_nd_ebitda else None
        )
        if net_debt_val is not None and not np.isfinite(net_debt_val):
            net_debt_val = None
        if nd_ebitda_val is not None and not np.isfinite(nd_ebitda_val):
            nd_ebitda_val = None

        spread, cod_bucket = _cod_spread(nd_ebitda_val, net_debt_val, spreads)

        if cod_bucket in ("nd_ebitda_4_6", "nd_ebitda_gt6", "nd_neg_ebitda"):
            logger.debug(
                f"cost_of_capital: {ticker} high leverage "
                f"(ND/EBITDA={nd_ebitda_val}) bucket={cod_bucket} — WARN"
            )

        cod_val = float(rf + spread)

        # --- Debt / equity weights ---
        mcap = (
            float(pd.to_numeric(row.get(col_mcap), errors="coerce"))
            if col_mcap else None
        )
        if mcap is None or not np.isfinite(mcap) or mcap <= 0:
            # No market cap → pure COE
            wacc_val = coe_val
            wacc_src = "coe_only_no_mcap"
        else:
            D = max(net_debt_val or 0.0, 0.0)   # only count positive net debt
            E = mcap
            V = E + D
            if V <= 0:
                wacc_val = coe_val
                wacc_src = "coe_only_V_zero"
            else:
                ew = E / V
                dw = D / V
                wacc_raw = ew * coe_val + dw * cod_val * (1 - tax)
                wacc_val = float(np.clip(wacc_raw, wacc_min, wacc_max))
                wacc_src = "computed"

        # Final sanity
        if not np.isfinite(wacc_val):
            wacc_val = wacc_fallback_val
            wacc_src = "fallback_nan"

        rf_list.append(rf)
        erp_list.append(erp)
        beta_list.append(beta_val)
        beta_source_list.append(src)
        beta_low_conf_list.append(low_conf)
        coe_list.append(coe_val)
        cod_list.append(cod_val)
        cod_bucket_list.append(cod_bucket)
        wacc_list.append(wacc_val)
        wacc_source_list.append(wacc_src)

    out["rf"] = rf_list
    out["erp"] = erp_list
    out["beta"] = beta_list
    out["beta_source"] = beta_source_list
    out["beta_low_confidence"] = beta_low_conf_list
    out["coe"] = coe_list
    out["cod"] = cod_list
    out["cod_bucket"] = cod_bucket_list
    out["wacc"] = wacc_list
    out["wacc_source"] = wacc_source_list

    # Summary stats
    wacc_ser = pd.Series(wacc_list)
    beta_ser = pd.Series(beta_list)
    log.info(
        f"cost_of_capital: WACC p10={wacc_ser.quantile(0.10):.3f} "
        f"median={wacc_ser.median():.3f} "
        f"p90={wacc_ser.quantile(0.90):.3f}"
    )
    log.info(
        f"cost_of_capital: beta p10={beta_ser.quantile(0.10):.2f} "
        f"median={beta_ser.median():.2f} "
        f"p90={beta_ser.quantile(0.90):.2f}"
    )
    src_counts = pd.Series(beta_source_list).value_counts().to_dict()
    log.info(f"cost_of_capital: beta_source breakdown = {src_counts}")

    out_path = processed / "master_cost.parquet"
    write_parquet(out_path, out)
    log.info(f"cost_of_capital: wrote {out_path} ({len(out)} rows)")
    return 0
