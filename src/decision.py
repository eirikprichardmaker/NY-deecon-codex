from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.common.config import resolve_paths
from src.common.errors import SchemaError
from src.common.io import read_parquet
from src.common.utils import safe_div, zscore

REASON_DATA_MISSING_MA200 = "DATA_MISSING_MA200"
REASON_DATA_MISSING_BENCHMARK = "DATA_MISSING_BENCHMARK"


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def canon(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    df.columns = [canon(c) for c in df.columns]
    return df


def _pick(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def _find_latest_valuation_csv(project_root: Path) -> Optional[Path]:
    runs_dir = project_root / "runs"
    if not runs_dir.exists():
        return None
    hits = list(runs_dir.rglob("valuation.csv"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    s = s.lstrip("^")
    if "." in s:
        s = s.split(".", 1)[0]
    return s.strip()


def _md_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "_(ingen)_"
    d = df.head(max_rows).copy()
    d = d.where(d.notna(), "")
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(
                lambda x: "" if x == "" else (f"{x:.3g}" if isinstance(x, (float, np.floating)) else str(x))
            )
        else:
            d[c] = d[c].astype(str)

    cols = list(d.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(d.iloc[i].tolist()) + " |" for i in range(len(d))]
    return "\n".join([header, sep] + rows)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def _join_reasons(parts: list[str]) -> str:
    return "; ".join([p for p in parts if p])


def _as_num_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    col = _pick(df, aliases)
    if col:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _to_decimal_rate(s: pd.Series) -> pd.Series:
    """
    Normalize rate-like columns to decimal when source is in percent points.
    Example: 12.0 -> 0.12 (if series median suggests percent-scale input).
    """
    x = pd.to_numeric(s, errors="coerce")
    med = x.abs().median(skipna=True)
    if np.isfinite(med) and med > 2.0:
        x = x / 100.0
    return x


def _quality_gate(df: pd.DataFrame, dec_cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    roic = _to_decimal_rate(_as_num_series(df, ["roic", "roic_current"]))
    fcf_yield = _as_num_series(df, ["fcf_yield"])
    nd_ebitda = _as_num_series(df, ["nd_ebitda", "n_debt_ebitda_current"])
    ev_ebit = _as_num_series(df, ["ev_ebit", "ev_ebit_current"])

    weak_roic = roic.isna() | (roic <= float(dec_cfg.get("quality_roic_min", 0.0)))
    weak_fcf_yield = fcf_yield.isna() | (fcf_yield <= float(dec_cfg.get("quality_fcf_yield_min", 0.0)))
    weak_nd_ebitda = nd_ebitda.isna() | (nd_ebitda > float(dec_cfg.get("quality_nd_ebitda_max", 3.5)))
    weak_ev_ebit = ev_ebit.isna() | (ev_ebit <= 0) | (ev_ebit > float(dec_cfg.get("quality_ev_ebit_max", 20.0)))

    out["weak_roic"] = weak_roic.astype(int)
    out["weak_fcf_yield"] = weak_fcf_yield.astype(int)
    out["weak_nd_ebitda"] = weak_nd_ebitda.astype(int)
    out["weak_ev_ebit"] = weak_ev_ebit.astype(int)

    out["quality_weak_count"] = (
        out["weak_roic"] +
        out["weak_fcf_yield"] +
        out["weak_nd_ebitda"] +
        out["weak_ev_ebit"]
    )

    fail_min = int(dec_cfg.get("quality_weak_fail_min", 2))
    out["quality_gate_ok"] = out["quality_weak_count"] < fail_min
    out["quality_gate_reason"] = np.where(
        out["quality_gate_ok"],
        "",
        f"quality_weak_count_gte_{fail_min}",
    )
    return out


def _value_creation_gate(df: pd.DataFrame, dec_cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    roic = _to_decimal_rate(_as_num_series(df, ["roic", "roic_current"]))
    wacc = _to_decimal_rate(_as_num_series(df, ["wacc_used", "wacc"]))
    spread = roic - wacc

    yearly_decay = float(dec_cfg.get("value_creation_spread_decay_per_year", 0.01))
    out["roic_dec"] = roic
    out["wacc_dec"] = wacc
    out["roic_wacc_spread"] = spread
    out["roic_wacc_spread_y1"] = spread - yearly_decay
    out["roic_wacc_spread_y2"] = spread - (2.0 * yearly_decay)
    out["roic_wacc_spread_y3"] = spread - (3.0 * yearly_decay)

    has_inputs = roic.notna() & wacc.notna()
    out["value_creation_ok"] = (
        has_inputs &
        (out["roic_wacc_spread_y1"] > 0.0) &
        (out["roic_wacc_spread_y2"] > 0.0) &
        (out["roic_wacc_spread_y3"] > 0.0)
    )
    out["value_creation_reason"] = np.where(
        has_inputs,
        np.where(out["value_creation_ok"], "", "roic_wacc_not_persistent_3y"),
        "missing_roic_or_wacc",
    )
    return out


def _suffix_from_symbol(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    if "." not in s:
        return ""
    return s.rsplit(".", 1)[-1].strip()


def _country_to_suffix(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    c = str(x).strip().upper()
    mapping = {
        "NO": "OL",
        "NORWAY": "OL",
        "SE": "ST",
        "SWEDEN": "ST",
        "DK": "CO",
        "DENMARK": "CO",
        "FI": "HE",
        "FINLAND": "HE",
    }
    return mapping.get(c, "")


def _resolve_index_symbol_by_suffix(dec_cfg: dict) -> dict[str, str]:
    out = {
        "OL": "^OSEAX",
        "ST": "^OMXS",
        "CO": "^OMXC25",
        "HE": "^HEX",
    }
    user = dec_cfg.get("index_ticker_by_suffix", {})
    if isinstance(user, dict):
        for k, v in user.items():
            if v is None:
                continue
            out[str(k).strip().upper()] = str(v).strip()
    return out


def _attach_relevant_index(df: pd.DataFrame, dec_cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["market_suffix"] = ""

    if "yahoo_ticker" in df.columns:
        sfx = df["yahoo_ticker"].map(_suffix_from_symbol)
        out.loc[sfx.ne(""), "market_suffix"] = sfx[sfx.ne("")]

    needs = out["market_suffix"].eq("")
    if needs.any() and "ticker" in df.columns:
        sfx_t = df["ticker"].map(_suffix_from_symbol)
        out.loc[needs & sfx_t.ne(""), "market_suffix"] = sfx_t[needs & sfx_t.ne("")]

    needs = out["market_suffix"].eq("")
    if needs.any():
        country_col = _pick(df, ["info_country", "country"])
        if country_col:
            sfx_c = df[country_col].map(_country_to_suffix)
            out.loc[needs & sfx_c.ne(""), "market_suffix"] = sfx_c[needs & sfx_c.ne("")]

    idx_map = _resolve_index_symbol_by_suffix(dec_cfg)
    out["relevant_index_symbol"] = out["market_suffix"].map(idx_map).fillna("")
    out["relevant_index_key"] = out["relevant_index_symbol"].map(_norm_ticker)
    return out


def _build_index_snapshot(prices_df: pd.DataFrame, asof: str, index_keys: list[str]) -> pd.DataFrame:
    cols = [
        "index_key", "index_price_date", "index_price", "index_ma21",
        "index_ma200", "index_mad", "index_above_ma200",
    ]
    if prices_df is None or prices_df.empty or not index_keys:
        return pd.DataFrame(columns=cols)

    px = _canon_cols(prices_df)
    ticker_col = _pick(px, ["ticker", "symbol", "yahoo_ticker"])
    date_col = _pick(px, ["date", "price_date", "datetime", "timestamp", "time"])
    price_col = _pick(px, ["adj_close", "close", "price", "last"])
    if not ticker_col or not date_col or not price_col:
        return pd.DataFrame(columns=cols)

    px = px.copy()
    px[date_col] = pd.to_datetime(px[date_col], errors="coerce")
    asof_dt = pd.to_datetime(asof)
    px = px.dropna(subset=[date_col, price_col])
    px = px[px[date_col] <= asof_dt]
    px["index_key"] = px[ticker_col].map(_norm_ticker)
    px = px[px["index_key"].isin(index_keys)].copy()
    if px.empty:
        return pd.DataFrame(columns=cols)

    px = px.sort_values(["index_key", date_col])
    px["_price"] = pd.to_numeric(px[price_col], errors="coerce")
    g = px.groupby("index_key", group_keys=False)
    px["_ma21"] = g["_price"].transform(lambda s: s.rolling(21, min_periods=21).mean())
    px["_ma200"] = g["_price"].transform(lambda s: s.rolling(200, min_periods=200).mean())
    px["_mad"] = (px["_ma21"] - px["_ma200"]) / px["_ma200"]
    px["_above"] = px["_price"] > px["_ma200"]

    snap = (
        px.sort_values(["index_key", date_col])
        .groupby("index_key", as_index=False)
        .tail(1)
        .copy()
    )
    snap = snap.rename(
        columns={
            date_col: "index_price_date",
            "_price": "index_price",
            "_ma21": "index_ma21",
            "_ma200": "index_ma200",
            "_mad": "index_mad",
            "_above": "index_above_ma200",
        }
    )
    return snap[cols]


def _apply_index_technical_filter(df: pd.DataFrame, prices_df: pd.DataFrame, asof: str, dec_cfg: dict, mad_min: float) -> pd.DataFrame:
    out = df.copy()
    idx_meta = _attach_relevant_index(out, dec_cfg)
    for c in idx_meta.columns:
        out[c] = idx_meta[c]

    idx_keys = sorted([k for k in out["relevant_index_key"].dropna().astype(str).unique().tolist() if k])
    idx_snap = _build_index_snapshot(prices_df, asof, idx_keys)
    out = out.merge(idx_snap, left_on="relevant_index_key", right_on="index_key", how="left")
    if "index_key" in out.columns:
        out = out.drop(columns=["index_key"])

    require_index_ma200 = bool(dec_cfg.get("require_index_ma200", True))
    require_index_mad = bool(dec_cfg.get("require_index_mad", False))
    idx_price = pd.to_numeric(out.get("index_price"), errors="coerce")
    idx_ma200 = pd.to_numeric(out.get("index_ma200"), errors="coerce")
    idx_mad = pd.to_numeric(out.get("index_mad"), errors="coerce")
    idx_has_price_ma = idx_price.notna() & idx_ma200.notna()
    idx_has_mad = idx_mad.notna()
    idx_above = idx_price > idx_ma200

    out["index_above_ma200"] = pd.Series(
        np.where(idx_has_price_ma, idx_above, pd.NA),
        index=out.index,
        dtype="boolean",
    )
    out["index_data_ok"] = out["relevant_index_key"].astype(str).ne("") & idx_has_price_ma
    out["index_ma200_ok"] = np.where(
        require_index_ma200,
        idx_has_price_ma & idx_above,
        True,
    )
    out["index_mad_ok"] = np.where(
        require_index_mad,
        idx_has_mad & (idx_mad >= mad_min),
        True,
    )
    out["index_tech_ok"] = out["index_ma200_ok"].astype(bool) & (out["index_mad_ok"].astype(bool) if require_index_mad else True)
    return out


def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    processed = paths["processed_dir"]

    in_path = processed / "master_valued.parquet"
    if not in_path.exists():
        raise SchemaError(f"decision: missing {in_path} (run valuation first)")

    df = _canon_cols(read_parquet(in_path))
    if "ticker" not in df.columns:
        raise SchemaError("decision: missing 'ticker' in master_valued.parquet")

    # --- valuation.csv: prefer same run_dir, else latest in runs/ ---
    vcsv = ctx.run_dir / "valuation.csv"
    if not vcsv.exists():
        vcsv = _find_latest_valuation_csv(ctx.project_root)
    if not vcsv or not vcsv.exists():
        raise SchemaError("decision: no valuation.csv found (run valuation first)")

    v = _canon_cols(pd.read_csv(vcsv))
    if "ticker" not in v.columns:
        raise SchemaError(f"decision: valuation.csv missing 'ticker': {vcsv}")

    keep = [c for c in ["yahoo_ticker", "ticker", "model", "intrinsic_equity", "intrinsic_ev", "net_debt_used", "wacc_used", "coe_used", "reason"] if c in v.columns]
    if ("intrinsic_equity" not in keep) and ("intrinsic_ev" not in keep):
        raise SchemaError(f"decision: valuation.csv has no intrinsic_equity/intrinsic_ev: {vcsv}")

    v = v[keep].copy()
    v["k"] = v["ticker"].map(_norm_ticker)
    v = v[v["k"] != ""].drop_duplicates(subset=["k"], keep="last").set_index("k")

    # normalized keys in master
    df["k_ticker"] = df["ticker"].map(_norm_ticker)
    df["k_base"] = df["base_ticker"].map(_norm_ticker) if "base_ticker" in df.columns else ""
    df["k_yahoo"] = df["yahoo_ticker"].map(_norm_ticker) if "yahoo_ticker" in df.columns else ""

    # map valuation fields in priority: yahoo -> base -> ticker

    map_cols = [c for c in ["intrinsic_equity", "intrinsic_ev", "net_debt_used", "wacc_used", "coe_used", "model", "reason"] if c in v.columns]


    # build lookups (dedupe to last just in case)

    # ensure key columns exist (robust)
    if "k" not in v.columns and "ticker" in v.columns:
        v["k"] = v["ticker"].map(_norm_ticker)
    if "k" not in df.columns and "ticker" in df.columns:
        df["k"] = df["ticker"].map(_norm_ticker)
    if "k_base" not in df.columns and "base_ticker" in df.columns:
        df["k_base"] = df["base_ticker"].map(_norm_ticker)
    if "k_yahoo" not in df.columns and "yahoo_ticker" in df.columns:
        df["k_yahoo"] = df["yahoo_ticker"].map(_norm_ticker)

    v_by_ticker = v.dropna(subset=["k"]).drop_duplicates(subset=["k"], keep="last").set_index("k")

    v_by_yahoo = None

    if "yahoo_ticker" in v.columns:

        v["k_yahoo"] = v["yahoo_ticker"].map(_norm_ticker)

        v_by_yahoo = v.dropna(subset=["k_yahoo"]).drop_duplicates(subset=["k_yahoo"], keep="last").set_index("k_yahoo")


    for col in map_cols:

        if col not in df.columns:

            df[col] = pd.NA


        # 1) yahoo (unique instrument id)

        if v_by_yahoo is not None and col in v_by_yahoo.columns and "k_yahoo" in df.columns:

            df[col] = df[col].where(df[col].notna(), df["k_yahoo"].map(v_by_yahoo[col]))


        # 2) base_ticker fallback

        if col in v_by_ticker.columns and "k_base" in df.columns:

            df[col] = df[col].where(df[col].notna(), df["k_base"].map(v_by_ticker[col]))


        # 3) ticker fallback

        if col in v_by_ticker.columns and "k" in df.columns:

            df[col] = df[col].where(df[col].notna(), df["k"].map(v_by_ticker[col]))


    # sikkerhetsnett (unngå dobbeltrader per instrument)

    if "yahoo_ticker" in df.columns:

        df = df.drop_duplicates(subset=["yahoo_ticker"], keep="first")


    ie = pd.to_numeric(df.get("intrinsic_equity"), errors="coerce")
    iev = pd.to_numeric(df.get("intrinsic_ev"), errors="coerce")
    log.info(f"decision: valuation source={vcsv}")
    log.info(f"decision: intrinsic_equity coverage={float(ie.notna().mean()):.3f} intrinsic_ev coverage={float(iev.notna().mean()):.3f}")

    # --- market cap (fix units) ---
    col_mcap = _pick(df, ["market_cap_current", "market_cap"])
    if not col_mcap:
        raise SchemaError("decision: missing market cap column (market_cap_current/market_cap)")

    df["market_cap"] = pd.to_numeric(df[col_mcap], errors="coerce")

    # market cap ser ut til å være i millioner -> skaler til valutaenheter hvis median er "liten"
    try:
        med = float(pd.to_numeric(df["market_cap"], errors="coerce").median())
    except Exception:
        med = None
    if med is not None and med < 1e6:
        df["market_cap"] = df["market_cap"] * 1_000_000.0
        log.info("decision: scaled market_cap by 1e6 (input looked like millions)")

    # EV
    col_net_debt = _pick(df, ["net_debt_current", "net_debt", "net_debt_used"])
    df["net_debt"] = pd.to_numeric(df[col_net_debt], errors="coerce") if col_net_debt else np.nan
    df["ev"] = df["market_cap"] + df["net_debt"].fillna(0)

    # --- MOS basis ---
    if ie.notna().any():
        df["intrinsic_value"] = ie
        df["mos_basis"] = "equity_vs_mcap"
        df["mos"] = safe_div(df["intrinsic_value"], df["market_cap"]) - 1.0
    elif iev.notna().any():
        df["intrinsic_value"] = iev
        df["mos_basis"] = "ev_vs_ev"
        df["mos"] = safe_div(df["intrinsic_value"], df["ev"]) - 1.0
    else:
        raise SchemaError("decision: intrinsic_equity/intrinsic_ev are empty after merge")

    # --- thresholds ---
    dec_cfg = (ctx.cfg.get("decision") or {})
    mos_min = float(dec_cfg.get("mos_min", 0.30))
    mos_high = float(dec_cfg.get("mos_high_uncertainty", 0.40))
    require_above_ma200 = bool(dec_cfg.get("require_above_ma200", True))
    mad_min = float(dec_cfg.get("mad_min", -0.05))
    technical_gate_mode = str(dec_cfg.get("technical_gate_mode", "two_of_three_index_primary")).strip().lower()
    top_n = int(dec_cfg.get("top_n", 25))
    if technical_gate_mode == "strict":
        tech_rule_text = "Teknisk regel: aksje + relevant indeks må være over 200d, og MAD må være over terskel."
    elif technical_gate_mode == "two_of_three":
        tech_rule_text = "Teknisk regel: 2 av 3 må være oppfylt (aksje>MA200, indeks>MA200, MAD>=terskel)."
    else:
        tech_rule_text = "Teknisk regel: 2 av 3 må være oppfylt med indeks-trend som primær (må være over MA200)."

    # --- risk flags ---
    df["beta"] = pd.to_numeric(df.get("beta", pd.NA), errors="coerce")
    df["nd_ebitda"] = pd.to_numeric(df.get("nd_ebitda", df.get("n_debt_ebitda_current", pd.NA)), errors="coerce")

    df["high_risk_flag"] = (
        (df["beta"].fillna(0) >= float(dec_cfg.get("beta_high", 1.5))) |
        (df["nd_ebitda"].fillna(0) >= float(dec_cfg.get("nd_ebitda_high", 3.5)))
    ).astype(int)
    df["mos_req"] = np.where(df["high_risk_flag"] == 1, mos_high, mos_min)

    prices_path = processed / "prices.parquet"
    if prices_path.exists():
        prices_df = read_parquet(prices_path)
    else:
        prices_df = pd.DataFrame()
        log.info(f"decision: missing {prices_path} -> index filter forces CASH")

    # --- technical filter ---
    stock_price = _as_num_series(df, ["adj_close", "close", "price", "last"])
    stock_ma200 = _as_num_series(df, ["ma200"])
    mad_s = _as_num_series(df, ["mad"])
    stock_has_price_ma = stock_price.notna() & stock_ma200.notna()
    stock_above_ma200 = stock_price > stock_ma200
    stock_data_ok = stock_has_price_ma

    above_ma200_series = pd.Series(
        np.where(stock_has_price_ma, stock_above_ma200, pd.NA),
        index=df.index,
        dtype="boolean",
    )
    df["above_ma200"] = above_ma200_series
    df["stock_data_ok"] = stock_data_ok

    stock_ma200_ok = pd.Series(True, index=df.index)
    if require_above_ma200:
        stock_ma200_ok = stock_has_price_ma & stock_above_ma200

    stock_mad_ok = mad_s.notna() & (mad_s >= mad_min)

    df["stock_ma200_ok"] = stock_ma200_ok
    df["stock_mad_ok"] = stock_mad_ok
    df = _apply_index_technical_filter(df, prices_df=prices_df, asof=ctx.asof, dec_cfg=dec_cfg, mad_min=mad_min)
    df["tech_signal_stock_trend"] = df["stock_ma200_ok"].astype(bool)
    df["tech_signal_index_trend"] = df["index_ma200_ok"].astype(bool)
    df["tech_signal_mad"] = df["stock_mad_ok"].astype(bool)
    df["tech_signal_count"] = (
        df["tech_signal_stock_trend"].astype(int) +
        df["tech_signal_index_trend"].astype(int) +
        df["tech_signal_mad"].astype(int)
    )
    tech_data_ready = df["stock_data_ok"].astype(bool) & df["index_data_ok"].astype(bool)

    if technical_gate_mode == "strict":
        df["tech_ok"] = (
            tech_data_ready &
            df["stock_ma200_ok"].astype(bool) &
            df["stock_mad_ok"].astype(bool) &
            df["index_tech_ok"].astype(bool)
        )
    elif technical_gate_mode == "two_of_three":
        df["tech_ok"] = tech_data_ready & df["tech_signal_count"].ge(2)
    else:
        df["tech_ok"] = tech_data_ready & df["tech_signal_index_trend"].astype(bool) & df["tech_signal_count"].ge(2)

    # --- quality score (lightweight) ---
    comps, wts = [], []
    if "roic" in df.columns and pd.to_numeric(df["roic"], errors="coerce").notna().any():
        comps.append(zscore(pd.to_numeric(df["roic"], errors="coerce"))); wts.append(0.60)
    elif "roic_current" in df.columns and pd.to_numeric(df["roic_current"], errors="coerce").notna().any():
        comps.append(zscore(pd.to_numeric(df["roic_current"], errors="coerce"))); wts.append(0.60)

    if "fcf_yield" in df.columns and pd.to_numeric(df["fcf_yield"], errors="coerce").notna().any():
        comps.append(zscore(pd.to_numeric(df["fcf_yield"], errors="coerce"))); wts.append(0.40)

    if comps:
        w = np.array(wts, dtype=float); w = w / w.sum()
        df["quality_score"] = sum(w[i] * comps[i] for i in range(len(comps)))
    else:
        df["quality_score"] = 0.0

    value_gate = _value_creation_gate(df, dec_cfg)
    quality_gate = _quality_gate(df, dec_cfg)
    for c in value_gate.columns:
        df[c] = value_gate[c]
    for c in quality_gate.columns:
        df[c] = quality_gate[c]

    df["fundamental_ok"] = (
        df["mos"].notna() &
        (df["mos"] >= df["mos_req"]) &
        df["value_creation_ok"].fillna(False) &
        df["quality_gate_ok"].fillna(False)
    )
    df["technical_ok"] = df["tech_ok"]

    df["ma200_ok"] = df["stock_ma200_ok"].astype(bool)
    df["reason_fundamental_fail"] = ""
    mos_fail = ~(df["mos"].notna() & (df["mos"] >= df["mos_req"]))
    vc_fail = ~df["value_creation_ok"].fillna(False)
    q_fail = ~df["quality_gate_ok"].fillna(False)

    df.loc[mos_fail, "reason_fundamental_fail"] = "mos_below_required"
    df.loc[vc_fail, "reason_fundamental_fail"] = df.loc[vc_fail, "reason_fundamental_fail"].map(
        lambda x: _join_reasons([x, "value_creation_fail"])
    )
    df.loc[q_fail, "reason_fundamental_fail"] = df.loc[q_fail, "reason_fundamental_fail"].map(
        lambda x: _join_reasons([x, "quality_gate_fail"])
    )

    if "value_creation_reason" in df.columns:
        needs_vc_reason = vc_fail & df["value_creation_reason"].astype(str).ne("")
        df.loc[needs_vc_reason, "reason_fundamental_fail"] = df.loc[needs_vc_reason].apply(
            lambda r: _join_reasons([r["reason_fundamental_fail"], str(r["value_creation_reason"])]),
            axis=1,
        )
    if "quality_gate_reason" in df.columns:
        needs_q_reason = q_fail & df["quality_gate_reason"].astype(str).ne("")
        df.loc[needs_q_reason, "reason_fundamental_fail"] = df.loc[needs_q_reason].apply(
            lambda r: _join_reasons([r["reason_fundamental_fail"], str(r["quality_gate_reason"])]),
            axis=1,
        )

    df["reason_technical_fail"] = ""
    missing_stock_ma200 = stock_ma200.isna()
    missing_stock_tech = stock_price.isna() & ~missing_stock_ma200
    stock_below_ma200 = df["stock_data_ok"].astype(bool) & ~df["stock_ma200_ok"].astype(bool)
    bad_mad = df["stock_data_ok"].astype(bool) & mad_s.notna() & ~df["stock_mad_ok"].astype(bool)

    df.loc[missing_stock_ma200, "reason_technical_fail"] = df.loc[missing_stock_ma200, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, REASON_DATA_MISSING_MA200])
    )
    df.loc[missing_stock_tech, "reason_technical_fail"] = df.loc[missing_stock_tech, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, "missing_stock_technical_data"])
    )
    df.loc[stock_below_ma200, "reason_technical_fail"] = df.loc[stock_below_ma200, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, "below_ma200"])
    )
    df.loc[bad_mad, "reason_technical_fail"] = df.loc[bad_mad, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, "mad_below_min"])
    )

    unknown_idx = df["relevant_index_key"].astype(str).eq("")
    missing_idx_data = df["relevant_index_key"].astype(str).ne("") & ~df["index_data_ok"].astype(bool)
    idx_has_price_ma = pd.to_numeric(df.get("index_price"), errors="coerce").notna() & pd.to_numeric(df.get("index_ma200"), errors="coerce").notna()
    idx_mad = pd.to_numeric(df.get("index_mad"), errors="coerce")
    idx_below_ma200 = idx_has_price_ma & ~df["index_ma200_ok"].astype(bool)
    idx_mad_below = idx_mad.notna() & ~df["index_mad_ok"].astype(bool)

    df.loc[unknown_idx, "reason_technical_fail"] = df.loc[unknown_idx, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, REASON_DATA_MISSING_BENCHMARK])
    )
    df.loc[missing_idx_data, "reason_technical_fail"] = df.loc[missing_idx_data, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, REASON_DATA_MISSING_BENCHMARK])
    )
    df.loc[idx_below_ma200, "reason_technical_fail"] = df.loc[idx_below_ma200, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, "index_below_ma200"])
    )
    df.loc[idx_mad_below, "reason_technical_fail"] = df.loc[idx_mad_below, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, "index_mad_below_min"])
    )
    df.loc[df["technical_ok"].astype(bool), "reason_technical_fail"] = ""

    df["eligible"] = df["technical_ok"] & df["fundamental_ok"]

    screen_cols = [c for c in [
        "ticker", "company", "market_cap", "intrinsic_value", "mos", "mos_req",
        "fundamental_ok", "technical_ok", "reason_fundamental_fail", "reason_technical_fail",
        "value_creation_ok", "roic_wacc_spread", "roic_wacc_spread_y3", "value_creation_reason",
        "quality_gate_ok", "quality_weak_count", "quality_gate_reason",
        "stock_ma200_ok", "stock_mad_ok", "ma200_ok",
        "relevant_index_symbol", "relevant_index_key", "index_price_date", "index_price", "index_ma200", "index_mad", "index_above_ma200",
        "index_data_ok", "index_ma200_ok", "index_mad_ok", "index_tech_ok",
        "mad", "ma21", "ma200", "above_ma200", "quality_score"
    ] if c in df.columns]
    _atomic_write_csv(ctx.run_dir / "screen_basic.csv", df[screen_cols])

    eligible = df[df["eligible"]].copy()
    eligible = eligible.sort_values(by=["quality_score", "mos", "market_cap"], ascending=[False, False, False], na_position="last")

    out_cols = [c for c in [
        "ticker", "company",
        "market_cap", "intrinsic_value", "mos", "mos_req", "mos_basis",
        "quality_score", "beta", "coe_used", "wacc_used",
        "value_creation_ok", "roic_wacc_spread", "roic_wacc_spread_y1", "roic_wacc_spread_y2", "roic_wacc_spread_y3", "value_creation_reason",
        "quality_gate_ok", "quality_weak_count", "quality_gate_reason",
        "above_ma200", "mad", "ma21", "ma200", "stock_ma200_ok", "stock_mad_ok",
        "relevant_index_symbol", "relevant_index_key", "index_price_date", "index_price", "index_ma200", "index_mad", "index_above_ma200",
        "index_data_ok", "ma200_ok", "index_ma200_ok", "index_mad_ok", "index_tech_ok", "high_risk_flag",
        "fundamental_ok", "technical_ok", "reason_fundamental_fail", "reason_technical_fail",
        "model", "reason",
    ] if c in df.columns]

    out_csv = ctx.run_dir / "decision.csv"
    out_md = ctx.run_dir / "decision.md"

    if eligible.empty:
        diag = df.sort_values(by=["quality_score", "mos"], ascending=[False, False], na_position="last")
        _atomic_write_csv(out_csv, diag[out_cols].head(top_n))
        _atomic_write_csv(ctx.run_dir / "shortlist.csv", pd.DataFrame(columns=out_cols))
        md = [
            f"# Decision ({ctx.asof})",
            "",
            "**Anbefaling:** CASH (ingen kandidater bestod filter).",
            "",
            "## Regelstatus",
            f"- MoS-regel aktiv: min {mos_min:.0%} (høy risiko {mos_high:.0%})",
            "- Verdiskaping-regel aktiv: ROIC > WACC i 3-års konservativ bane",
            "- Kvalitetsregel aktiv: >=2 svekkede kvalitetsindikatorer => CASH",
            f"- {tech_rule_text}",
            "",
            "## Topp (diagnostikk – før filter)",
            _md_table(diag[out_cols], max_rows=10),
        ]
        _atomic_write_text(out_md, "\n".join(md))
        log.info(f"decision: wrote {out_csv}")
        log.info(f"decision: wrote {out_md}")
        return 0

    pick = eligible.iloc[0]
    _atomic_write_csv(out_csv, eligible[out_cols].head(top_n))
    _atomic_write_csv(ctx.run_dir / "shortlist.csv", eligible[out_cols].head(top_n))

    md = []
    md.append(f"# Decision ({ctx.asof})")
    md.append("")
    md.append(f"**Anbefaling:** Kandidat = `{pick['ticker']}` (beste blant de som bestod filter).")
    md.append("")
    md.append("## Nøkkeltall")
    md.append(f"- MoS-basis: {pick.get('mos_basis','')}")
    md.append(f"- Market cap: {float(pick['market_cap']):.3g}")
    md.append(f"- Intrinsic: {float(pick['intrinsic_value']):.3g}")
    md.append(f"- MoS: {float(pick['mos'])*100:.1f}% (krav: {float(pick['mos_req'])*100:.0f}%)")
    if np.isfinite(pick.get("beta", np.nan)): md.append(f"- Beta: {float(pick['beta']):.2f}")
    if np.isfinite(pick.get("quality_score", np.nan)): md.append(f"- Quality score: {float(pick['quality_score']):.3f}")
    md.append("")
    md.append("## Årsaker (5-10 punkter)")
    md.append(f"- Krav MoS >= {mos_min:.0%} (høy risiko: {mos_high:.0%})")
    md.append("- Verdiskaping: ROIC-WACC må være positiv i 3-års konservativ bane")
    md.append("- Kvalitetsgate: minst 2 svekkede indikatorer forkaster kandidat")
    md.append(f"- {tech_rule_text}")
    md.append(f"- Valgt ticker har MoS {float(pick['mos']):.1%} og quality_score {float(pick.get('quality_score', 0.0)):.3f}")
    if np.isfinite(pick.get("roic_wacc_spread", np.nan)):
        md.append(f"- ROIC-WACC spread (normalisert): {float(pick.get('roic_wacc_spread')):.3%}")
    if np.isfinite(pick.get("quality_weak_count", np.nan)):
        md.append(f"- Svekkede kvalitetsindikatorer: {int(pick.get('quality_weak_count'))}")
    if "ma200" in pick.index and np.isfinite(pick.get("ma200", np.nan)):
        md.append(f"- Teknisk: pris over MA200={bool(pick.get('above_ma200', False))}, MA200={float(pick.get('ma200')):.3g}")
    if "mad" in pick.index and np.isfinite(pick.get("mad", np.nan)):
        md.append(f"- Momentum (MAD)={float(pick.get('mad')):.3f}, terskel={mad_min:.3f}")
    if str(pick.get("relevant_index_symbol", "")):
        md.append(
            f"- Relevant indeks: {pick.get('relevant_index_symbol')} | over MA200={bool(pick.get('index_ma200_ok', False))} | "
            f"index MAD ok={bool(pick.get('index_mad_ok', False))}"
        )
    worst = df.sort_values(by=["mos"], ascending=True, na_position="last").head(1)
    if not worst.empty:
        md.append(f"- Worst-case kandidat nå: {worst.iloc[0].get('ticker','')} med MoS={float(worst.iloc[0].get('mos', float('nan'))):.1%}")
    md.append("- Break-case: Hvis valgt aksje faller under MA200 eller MOS < krav ved neste kjøring => gå til CASH.")
    md.append("")
    md.append("## Topp 10 (eligible)")
    md.append(_md_table(eligible[out_cols], max_rows=10))

    _atomic_write_text(out_md, "\n".join(md))
    log.info(f"decision: wrote {out_csv}")
    log.info(f"decision: wrote {out_md}")
    return 0
