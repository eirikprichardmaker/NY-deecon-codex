from __future__ import annotations

import json
import html
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
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
REASON_DATA_NOT_SUFFICIENT = "DATA_NOT_SUFFICIENT_FOR_EVALUATION"
REASON_DATA_INVALID = "DATA_INVALID"

DEFAULT_CANDIDATE_REQUIRED_FIELDS = [
    "intrinsic_value",
    "market_cap",
    "mos",
    "mos_req",
    "quality_score",
    "roic_wacc_spread",
    "adj_close",
    "ma200",
    "mad",
    "index_price",
    "index_ma200",
]

DEFAULT_MEDIA_RED_FLAG_TERMS = [
    "bankrupt",
    "bankruptcy",
    "insolvency",
    "investigation",
    "fraud",
    "lawsuit",
    "warning",
    "profit warning",
    "guidance cut",
    "downgrade",
    "delisting",
    "rights issue",
    "dilution",
    "loss",
    "operating loss",
    "covenant breach",
    "liquidity crisis",
    "accounting",
]

DEFAULT_VALUE_QC_SPECS: list[dict] = [
    {"metric": "market_cap", "col": "market_cap", "hard_min": 1e6, "hard_max": 1e14, "source_cols": ["market_cap", "market_cap_current"]},
    {"metric": "intrinsic_value", "col": "intrinsic_value", "hard_min": -1e14, "hard_max": 1e15, "source_cols": ["intrinsic_value", "intrinsic_equity", "intrinsic_ev"]},
    {"metric": "mos", "col": "mos", "hard_min": -2.0, "hard_max": 50.0, "source_cols": ["intrinsic_value", "market_cap"], "formula": "mos"},
    {"metric": "roic", "col": "roic_dec", "hard_min": -5.0, "hard_max": 10.0, "source_cols": ["roic", "roic_current"]},
    {"metric": "wacc", "col": "wacc_dec", "hard_min": 0.01, "hard_max": 0.40, "source_cols": ["wacc_used", "wacc", "coe_used"]},
    {"metric": "roic_wacc_spread", "col": "roic_wacc_spread", "hard_min": -5.0, "hard_max": 10.0, "source_cols": ["roic_dec", "wacc_dec"], "formula": "roic_wacc_spread"},
    {"metric": "quality_score", "col": "quality_score", "hard_min": -10.0, "hard_max": 10.0, "source_cols": ["quality_score"]},
    {"metric": "ev_ebit", "col": "ev_ebit", "hard_min": -50.0, "hard_max": 200.0, "source_cols": ["ev_ebit", "ev_ebit_current"]},
    {"metric": "nd_ebitda", "col": "nd_ebitda", "hard_min": -20.0, "hard_max": 50.0, "source_cols": ["nd_ebitda", "n_debt_ebitda_current"]},
    {"metric": "fcf_yield", "col": "fcf_yield", "hard_min": -2.0, "hard_max": 2.0, "source_cols": ["fcf_yield"]},
    {"metric": "adj_close", "col": "adj_close", "hard_min": 0.0, "hard_max": 1e6, "source_cols": ["adj_close"]},
    {"metric": "ma200", "col": "ma200", "hard_min": 0.0, "hard_max": 1e6, "source_cols": ["ma200"]},
    {"metric": "mad", "col": "mad", "hard_min": -5.0, "hard_max": 5.0, "source_cols": ["mad", "ma21", "ma200"], "formula": "mad"},
    {"metric": "index_price", "col": "index_price", "hard_min": 0.0, "hard_max": 1e6, "source_cols": ["index_price"]},
    {"metric": "index_ma200", "col": "index_ma200", "hard_min": 0.0, "hard_max": 1e6, "source_cols": ["index_ma200"]},
    {"metric": "index_mad", "col": "index_mad", "hard_min": -5.0, "hard_max": 5.0, "source_cols": ["index_mad", "index_ma21", "index_ma200"], "formula": "index_mad"},
]

DIVIDEND_CRITERIA = [
    ("dividend_history", "Dividend History"),
    ("dividend_growth", "Dividend Growth"),
    ("payout_ratio", "Payout ratio"),
    ("share_count", "Increase of the number of shares"),
    ("market_value", "Market value"),
    ("fcf_margin", "Free cash flow margin"),
    ("ocf_margin", "Operating cash flow margin"),
    ("profit_margin", "Profit Margin"),
    ("roa", "Return on assets"),
    ("yield", "Yield"),
    ("pe", "P/E"),
]

GRAHAM_CRITERIA = [
    ("size", "Size"),
    ("financial_strength", "Financial strength"),
    ("earning_stability", "Earning stability-requirements for profit"),
    ("dividend_history", "Dividend History-requirements for the dividend"),
    ("profit_growth", "Profit GROWTH - Total profit growth"),
    ("moderate_pe", "Moderate P/E ratio"),
    ("moderate_equity_price", "Moderate Equity Price"),
]

PRODUCT_FIELD_SPECS: list[tuple[str, str, str]] = [
    ("revenue_goods", "Goods sales", "PRODUCT_GOODS"),
    ("revenue_services", "Services sales", "PRODUCT_SERVICES"),
    ("ins_earned_premiums", "Insurance premiums", "PRODUCT_INSURANCE_PREMIUMS"),
    ("bank_total_loans_gross", "Bank lending products", "PRODUCT_BANK_LOANS"),
    ("bank_customer_deposits", "Bank deposit products", "PRODUCT_BANK_DEPOSITS"),
    ("bank_fee_commission_income", "Bank fee services", "PRODUCT_BANK_FEE_SERVICES"),
    ("bank_net_interest_income", "Bank net interest products", "PRODUCT_BANK_NET_INTEREST"),
]


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
    d = d.astype(object).where(d.notna(), "")
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


def _to_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _series_has_value(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s.notna()
    t = s.astype(str).str.strip()
    return t.ne("") & t.ne("nan") & t.ne("None")


def _to_float(x: object) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _calc_formula_value(metric_formula: str, row: pd.Series) -> float:
    if metric_formula == "mos":
        iv = _to_float(row.get("intrinsic_value"))
        mc = _to_float(row.get("market_cap"))
        if np.isfinite(iv) and np.isfinite(mc) and mc != 0:
            return (iv / mc) - 1.0
        return float("nan")
    if metric_formula == "roic_wacc_spread":
        ro = _to_float(row.get("roic_dec"))
        wc = _to_float(row.get("wacc_dec"))
        if np.isfinite(ro) and np.isfinite(wc):
            return ro - wc
        return float("nan")
    if metric_formula == "mad":
        m21 = _to_float(row.get("ma21"))
        m200 = _to_float(row.get("ma200"))
        if np.isfinite(m21) and np.isfinite(m200) and m200 != 0:
            return (m21 - m200) / m200
        return float("nan")
    if metric_formula == "index_mad":
        m21 = _to_float(row.get("index_ma21"))
        m200 = _to_float(row.get("index_ma200"))
        if np.isfinite(m21) and np.isfinite(m200) and m200 != 0:
            return (m21 - m200) / m200
        return float("nan")
    return float("nan")


def _formula_consistent(observed: float, expected: float, rel_tol: float = 0.02, abs_tol: float = 1e-6) -> bool:
    if not np.isfinite(observed) or not np.isfinite(expected):
        return False
    return abs(observed - expected) <= max(abs_tol, abs(expected) * rel_tol)


def _build_value_qc_flags(df: pd.DataFrame, dec_cfg: dict) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    qc_cfg = dec_cfg.get("value_qc", {}) or {}
    iqr_mult = float(qc_cfg.get("iqr_multiplier", 3.0))
    min_samples = int(qc_cfg.get("min_samples_for_distribution", 30))
    enabled = bool(qc_cfg.get("enabled", True))
    specs = list(DEFAULT_VALUE_QC_SPECS)

    out = pd.DataFrame(index=df.index)
    summary_rows: list[dict] = []
    if not enabled:
        for spec in specs:
            m = spec["metric"]
            out[f"value_qc_{m}_flag"] = False
            out[f"value_qc_{m}_lower"] = np.nan
            out[f"value_qc_{m}_upper"] = np.nan
        out["value_qc_alert_count"] = 0
        out["value_qc_has_alerts"] = False
        out["value_qc_alert_metrics"] = ""
        summary = pd.DataFrame(
            [{"metric": s["metric"], "status": "disabled", "alert_count": 0, "sample_count": 0} for s in specs]
        )
        return out, specs, summary

    for spec in specs:
        metric = str(spec["metric"])
        col = str(spec["col"])
        hard_min = spec.get("hard_min", None)
        hard_max = spec.get("hard_max", None)
        flag_col = f"value_qc_{metric}_flag"
        lo_col = f"value_qc_{metric}_lower"
        hi_col = f"value_qc_{metric}_upper"

        if col not in df.columns:
            out[flag_col] = False
            out[lo_col] = np.nan
            out[hi_col] = np.nan
            summary_rows.append(
                {
                    "metric": metric,
                    "source_col": col,
                    "sample_count": 0,
                    "lower": np.nan,
                    "upper": np.nan,
                    "alert_count": 0,
                    "status": "missing_column",
                }
            )
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        finite = s[np.isfinite(s)]
        lower = -np.inf
        upper = np.inf
        status = "ok"
        if len(finite) >= min_samples:
            q1 = float(finite.quantile(0.25))
            q3 = float(finite.quantile(0.75))
            iqr = q3 - q1
            if np.isfinite(iqr) and iqr > 0:
                lower = q1 - (iqr_mult * iqr)
                upper = q3 + (iqr_mult * iqr)
            else:
                status = "zero_iqr"
        else:
            status = "small_sample"

        if hard_min is not None and np.isfinite(float(hard_min)):
            lower = max(lower, float(hard_min))
        if hard_max is not None and np.isfinite(float(hard_max)):
            upper = min(upper, float(hard_max))
        if not np.isfinite(lower):
            lower = -np.inf
        if not np.isfinite(upper):
            upper = np.inf
        if lower > upper:
            lower, upper = upper, lower

        flags = s.isna() | ~np.isfinite(s) | (s < lower) | (s > upper)
        out[flag_col] = flags.astype(bool)
        out[lo_col] = lower
        out[hi_col] = upper
        summary_rows.append(
            {
                "metric": metric,
                "source_col": col,
                "sample_count": int(finite.shape[0]),
                "lower": lower,
                "upper": upper,
                "alert_count": int(flags.sum()),
                "status": status,
            }
        )

    flag_cols = [f"value_qc_{s['metric']}_flag" for s in specs if f"value_qc_{s['metric']}_flag" in out.columns]
    if flag_cols:
        out["value_qc_alert_count"] = out[flag_cols].astype(int).sum(axis=1)
        out["value_qc_has_alerts"] = out["value_qc_alert_count"] > 0
    else:
        out["value_qc_alert_count"] = 0
        out["value_qc_has_alerts"] = False

    def _join_alert_metrics(r: pd.Series) -> str:
        hits = []
        for spec in specs:
            m = spec["metric"]
            c = f"value_qc_{m}_flag"
            if c in r.index and _to_bool(r.get(c)):
                hits.append(m)
        return ", ".join(hits)

    out["value_qc_alert_metrics"] = out.apply(_join_alert_metrics, axis=1)
    summary = pd.DataFrame(summary_rows)
    return out, specs, summary


def _analyze_candidate_value_qc(pick: pd.Series, specs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in specs:
        metric = str(spec["metric"])
        col = str(spec["col"])
        flag_col = f"value_qc_{metric}_flag"
        lo_col = f"value_qc_{metric}_lower"
        hi_col = f"value_qc_{metric}_upper"
        is_alert = _to_bool(pick.get(flag_col))
        observed = _to_float(pick.get(col))
        lower = _to_float(pick.get(lo_col))
        upper = _to_float(pick.get(hi_col))
        sources = [str(c) for c in spec.get("source_cols", [])]

        source_parts: list[str] = []
        source_values: list[float] = []
        for c in sources:
            if c not in pick.index:
                continue
            v = _to_float(pick.get(c))
            if np.isfinite(v):
                source_values.append(v)
                source_parts.append(f"{c}={v:.6g}")
            else:
                source_parts.append(f"{c}=n/a")

        unique_vals = {round(v, 8) for v in source_values}
        source_unique_count = int(len(unique_vals))

        formula_name = str(spec.get("formula", "")).strip()
        formula_expected = float("nan")
        formula_ok = False
        formula_note = ""
        if formula_name:
            formula_expected = _calc_formula_value(formula_name, pick)
            formula_ok = _formula_consistent(observed, formula_expected)
            formula_note = (
                f"{formula_name}: observed={observed:.6g}, expected={formula_expected:.6g}, ok={formula_ok}"
                if np.isfinite(formula_expected) and np.isfinite(observed)
                else f"{formula_name}: insufficient_inputs"
            )

        resolved = (source_unique_count >= 2) or formula_ok or (not is_alert)
        rows.append(
            {
                "metric": metric,
                "value": observed,
                "lower": lower,
                "upper": upper,
                "is_alert": bool(is_alert),
                "resolved": bool(resolved),
                "source_unique_values": source_unique_count,
                "source_snapshot": "; ".join(source_parts),
                "formula_check": formula_note,
                "note": "" if (not is_alert or resolved) else "Uvanlig verdi uten robust forklaring i tilgjengelige kilder.",
            }
        )

    out = pd.DataFrame(rows)
    return out


def _candidate_data_sufficiency(
    df: pd.DataFrame,
    dec_cfg: dict,
) -> tuple[pd.DataFrame, list[str], int, float]:
    raw = dec_cfg.get("candidate_required_fields", DEFAULT_CANDIDATE_REQUIRED_FIELDS)
    if isinstance(raw, str):
        required = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, list):
        required = [str(x).strip() for x in raw if str(x).strip()]
    else:
        required = list(DEFAULT_CANDIDATE_REQUIRED_FIELDS)
    if not required:
        required = list(DEFAULT_CANDIDATE_REQUIRED_FIELDS)

    min_count_default = max(1, int(round(len(required) * 0.75)))
    min_count = int(dec_cfg.get("candidate_min_required_fields", min_count_default))
    min_ratio = float(dec_cfg.get("candidate_min_required_ratio", 0.75))
    if min_count < 1:
        min_count = 1
    if min_ratio < 0:
        min_ratio = 0.0
    if min_ratio > 1:
        min_ratio = 1.0

    out = pd.DataFrame(index=df.index)
    available_cols: list[pd.Series] = []
    for col in required:
        if col in df.columns:
            available_cols.append(_series_has_value(df[col]))
        else:
            available_cols.append(pd.Series(False, index=df.index, dtype=bool))

    if available_cols:
        avail_mat = pd.concat(available_cols, axis=1)
        avail_mat.columns = required
        avail_count = avail_mat.sum(axis=1).astype(int)
    else:
        avail_mat = pd.DataFrame(index=df.index)
        avail_count = pd.Series(0, index=df.index, dtype=int)

    required_count = int(len(required))
    ratio = avail_count / float(required_count) if required_count > 0 else pd.Series(1.0, index=df.index)
    out["candidate_required_fields_count"] = required_count
    out["candidate_available_fields_count"] = avail_count
    out["candidate_data_coverage_ratio"] = ratio.astype(float)
    out["candidate_data_ok"] = (avail_count >= min_count) & (ratio >= min_ratio)

    if required:
        missing_fields = []
        for i in df.index:
            miss = [c for c in required if not bool(avail_mat.loc[i, c])]
            missing_fields.append(", ".join(miss))
        out["candidate_data_missing_fields"] = pd.Series(missing_fields, index=df.index, dtype=object)
    else:
        out["candidate_data_missing_fields"] = ""

    return out, required, min_count, min_ratio


def _empty_candidate_products() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "company",
            "product_id",
            "product_label",
            "source_field",
            "value",
            "value_share",
            "detail",
        ]
    )


def _empty_candidate_product_demand_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "company",
            "product_id",
            "product_label",
            "scenario",
            "year_offset",
            "forecast_year",
            "demand_index",
            "product_demand_index",
            "demand_cagr_assumption",
            "model_confidence",
            "driver_signal_z",
            "anchor_growth_used",
            "available_metric_count",
        ]
    )


def _empty_candidate_product_demand_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "company",
            "asof",
            "base_demand_cagr",
            "bear_demand_cagr",
            "bull_demand_cagr",
            "model_confidence",
            "driver_signal_z",
            "anchor_growth_used",
            "available_metric_count",
            "driver_metrics_used",
            "forecast_source",
            "index_symbol",
            "market_history_days",
            "market_ann_return",
            "market_ann_vol",
        ]
    )


def _empty_candidate_market_position() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "company",
            "market_position_metric",
            "value",
            "comment",
        ]
    )


def _extract_candidate_products(pick: pd.Series) -> pd.DataFrame:
    ticker = str(pick.get("ticker", ""))
    company = str(pick.get("company", ""))
    rows: list[dict] = []

    for field, label, product_id in PRODUCT_FIELD_SPECS:
        val = _to_float(pick.get(field, np.nan))
        if np.isfinite(val) and abs(val) > 0:
            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "product_id": product_id,
                    "product_label": label,
                    "source_field": field,
                    "value": float(abs(val)),
                    "value_share": np.nan,
                    "detail": "structured_field",
                }
            )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "company": company,
                    "product_id": "PRODUCT_UNSPECIFIED",
                    "product_label": "Core offerings (not available as structured product fields)",
                    "source_field": "",
                    "value": np.nan,
                    "value_share": np.nan,
                    "detail": "fallback_no_product_fields",
                }
            ]
        )

    out = pd.DataFrame(rows)
    total = pd.to_numeric(out["value"], errors="coerce").replace([np.inf, -np.inf], np.nan).sum(skipna=True)
    if np.isfinite(total) and total > 0:
        out["value_share"] = pd.to_numeric(out["value"], errors="coerce") / float(total)
    else:
        out["value_share"] = np.nan
    out = out.sort_values(["value", "product_id"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return out


def _robust_z_for_pick(universe_df: pd.DataFrame, pick: pd.Series, col: str) -> float:
    if col not in universe_df.columns:
        return float("nan")
    x = pd.to_numeric(universe_df[col], errors="coerce")
    finite = x[np.isfinite(x)]
    if int(finite.shape[0]) < 8:
        return float("nan")
    val = _to_float(pick.get(col, np.nan))
    if not np.isfinite(val):
        return float("nan")

    med = float(finite.median())
    mad = float((finite - med).abs().median())
    if np.isfinite(mad) and mad > 0:
        z = 0.6745 * (val - med) / mad
    else:
        sd = float(finite.std(ddof=1))
        if (not np.isfinite(sd)) or sd <= 0:
            return float("nan")
        z = (val - float(finite.mean())) / sd
    return float(np.clip(z, -3.0, 3.0))


def _estimate_candidate_demand_inputs(pick: pd.Series, universe_df: pd.DataFrame) -> dict[str, object]:
    driver_specs: list[tuple[str, float]] = [
        ("profit_growth_5y", 0.40),
        ("fcf_yield", 0.20),
        ("t2_ebit_margin", 0.15),
        ("t2_gross_margin", 0.10),
        ("roic_wacc_spread", 0.10),
        ("quality_score", 0.05),
    ]

    used_metrics: list[str] = []
    z_parts: list[float] = []
    w_parts: list[float] = []
    for col, w in driver_specs:
        z = _robust_z_for_pick(universe_df, pick, col)
        if np.isfinite(z):
            used_metrics.append(col)
            z_parts.append(float(z))
            w_parts.append(float(w))

    driver_signal_z = float("nan")
    if w_parts:
        driver_signal_z = float(np.average(np.array(z_parts, dtype=float), weights=np.array(w_parts, dtype=float)))

    anchor = _to_float(pick.get("profit_growth_5y", np.nan))
    if not np.isfinite(anchor):
        anchor = _to_float(pick.get("dividend_growth_5y", np.nan))
    if np.isfinite(anchor):
        anchor = float(np.clip(anchor, -0.25, 0.25))

    if np.isfinite(anchor):
        base_cagr = 0.70 * anchor + (0.03 * np.tanh(driver_signal_z) if np.isfinite(driver_signal_z) else 0.0) + 0.01
    else:
        base_cagr = (0.02 + (0.04 * np.tanh(driver_signal_z))) if np.isfinite(driver_signal_z) else 0.02
    base_cagr = float(np.clip(base_cagr, -0.20, 0.25))

    bear_cagr = float(np.clip(base_cagr - 0.05, -0.25, 0.25))
    bull_cagr = float(np.clip(base_cagr + 0.05, -0.25, 0.35))

    metric_n = int(len(used_metrics))
    confidence_score = min(0.95, 0.25 + 0.12 * metric_n + (0.20 if np.isfinite(anchor) else 0.0))
    if confidence_score >= 0.75:
        confidence = "HIGH"
    elif confidence_score >= 0.50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "base_cagr": base_cagr,
        "bear_cagr": bear_cagr,
        "bull_cagr": bull_cagr,
        "model_confidence": confidence,
        "driver_signal_z": driver_signal_z,
        "anchor_growth_used": anchor,
        "available_metric_count": metric_n,
        "driver_metrics_used": ";".join(used_metrics),
    }


def _estimate_market_forecast_from_prices(
    prices_df: pd.DataFrame,
    pick: pd.Series,
    asof: str,
    lookback_years: int = 5,
) -> dict[str, object]:
    if prices_df is None or prices_df.empty:
        return {"source": "fundamental_fallback"}

    px = _canon_cols(prices_df)
    t_col = _pick(px, ["ticker", "symbol", "yahoo_ticker"])
    d_col = _pick(px, ["date", "price_date", "datetime", "timestamp", "time"])
    p_col = _pick(px, ["adj_close", "close", "price", "last"])
    if not t_col or not d_col or not p_col:
        return {"source": "fundamental_fallback"}

    idx_raw = str(pick.get("relevant_index_symbol", "")).strip()
    if not idx_raw:
        idx_raw = str(pick.get("market_index_ticker", "")).strip()
    if not idx_raw:
        return {"source": "fundamental_fallback"}

    idx_candidates = {idx_raw, idx_raw.lstrip("^")}
    px = px.copy()
    px[t_col] = px[t_col].astype(str).str.strip()
    px[d_col] = pd.to_datetime(px[d_col], errors="coerce")
    px[p_col] = pd.to_numeric(px[p_col], errors="coerce")
    px = px[px[d_col].notna() & np.isfinite(px[p_col])]
    if px.empty:
        return {"source": "fundamental_fallback"}

    asof_ts = pd.to_datetime(asof, errors="coerce")
    if pd.notna(asof_ts):
        start_ts = asof_ts - pd.DateOffset(years=max(1, int(lookback_years)))
        px = px[(px[d_col] <= asof_ts) & (px[d_col] >= start_ts)]

    x = px[px[t_col].isin(idx_candidates)].sort_values(d_col)
    if len(x) < 60:
        return {"source": "fundamental_fallback"}

    rets = np.log(x[p_col].astype(float) / x[p_col].astype(float).shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 40:
        return {"source": "fundamental_fallback"}

    mu = float(rets.mean()) * 252.0
    vol = float(rets.std(ddof=1)) * np.sqrt(252.0)
    if not np.isfinite(mu):
        return {"source": "fundamental_fallback"}
    if not np.isfinite(vol) or vol <= 0:
        vol = 0.15

    base = float(np.clip(mu, -0.20, 0.25))
    bear = float(np.clip(base - (0.5 * vol), -0.25, 0.20))
    bull = float(np.clip(base + (0.5 * vol), -0.20, 0.35))
    conf = "HIGH" if len(rets) >= 500 else ("MEDIUM" if len(rets) >= 250 else "LOW")
    return {
        "source": "market_data",
        "index_symbol": idx_raw,
        "history_days": int(len(rets)),
        "ann_return": float(mu),
        "ann_vol": float(vol),
        "base_cagr": base,
        "bear_cagr": bear,
        "bull_cagr": bull,
        "model_confidence": conf,
        "driver_metrics_used": "index_log_returns",
    }


def _build_candidate_market_position(pick: pd.Series, universe_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ticker = str(pick.get("ticker", ""))
    company = str(pick.get("company", ""))
    u = universe_df.copy() if universe_df is not None else pd.DataFrame()

    def _pctile(col: str) -> float:
        if col not in u.columns:
            return float("nan")
        s = pd.to_numeric(u[col], errors="coerce")
        v = _to_float(pick.get(col, np.nan))
        s = s[np.isfinite(s)]
        if (not np.isfinite(v)) or s.empty:
            return float("nan")
        return float((s <= v).mean())

    mcap_pct = _pctile("market_cap")
    q_pct = _pctile("quality_score")
    spread_pct = _pctile("roic_wacc_spread")
    mos_pct = _pctile("mos")

    for label, val in [
        ("Market cap percentile (univers)", mcap_pct),
        ("Quality percentile (univers)", q_pct),
        ("ROIC-WACC percentile (univers)", spread_pct),
        ("MoS percentile (univers)", mos_pct),
    ]:
        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "market_position_metric": label,
                "value": f"{val * 100:.1f}%" if np.isfinite(val) else "n/a",
                "comment": "Hoyere er bedre" if np.isfinite(val) else "Data ikke tilgjengelig",
            }
        )

    mcap_pick = _to_float(pick.get("market_cap", np.nan))
    seg = u.copy()
    sector = str(pick.get("sector", "")).strip()
    country = str(pick.get("country", "")).strip()
    if sector and "sector" in seg.columns:
        seg = seg[seg["sector"].astype(str).str.strip() == sector]
    if country and "country" in seg.columns:
        seg = seg[seg["country"].astype(str).str.strip() == country]
    seg_m = pd.to_numeric(seg.get("market_cap", pd.Series([], dtype=float)), errors="coerce")
    seg_m = seg_m[np.isfinite(seg_m)]
    if np.isfinite(mcap_pick) and not seg_m.empty:
        rank = int((seg_m > mcap_pick).sum()) + 1
        rows.append(
            {
                "ticker": ticker,
                "company": company,
                "market_position_metric": "Storrelse-rang i segment",
                "value": f"{rank}/{int(len(seg_m))}",
                "comment": f"Segment=sector:{sector or 'n/a'} country:{country or 'n/a'}",
            }
        )

    if not rows:
        return _empty_candidate_market_position()
    return pd.DataFrame(rows)


def _build_candidate_product_demand_forecast(
    pick: pd.Series,
    universe_df: pd.DataFrame,
    asof: str,
    products: pd.DataFrame,
    market_inputs: dict[str, object] | None = None,
    horizon_years: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if products is None or products.empty:
        return _empty_candidate_product_demand_forecast(), _empty_candidate_product_demand_summary()

    inputs = _estimate_candidate_demand_inputs(pick=pick, universe_df=universe_df)
    forecast_source = "fundamental_model"
    if isinstance(market_inputs, dict) and str(market_inputs.get("source", "")).strip().lower() == "market_data":
        for k in ["base_cagr", "bear_cagr", "bull_cagr", "model_confidence", "driver_metrics_used"]:
            if k in market_inputs:
                inputs[k] = market_inputs[k]
        forecast_source = "market_data"

    scenario_cfg = [
        ("bear", float(inputs["bear_cagr"])),
        ("base", float(inputs["base_cagr"])),
        ("bull", float(inputs["bull_cagr"])),
    ]

    asof_ts = pd.to_datetime(asof, errors="coerce")
    base_year = int(asof_ts.year) if pd.notna(asof_ts) else 0
    rows: list[dict] = []

    for _, p_row in products.iterrows():
        share = _to_float(p_row.get("value_share", np.nan))
        if not np.isfinite(share) or share <= 0:
            share = 1.0
        for scenario, cagr in scenario_cfg:
            for y in range(1, int(horizon_years) + 1):
                demand_index = float(100.0 * ((1.0 + float(cagr)) ** y))
                rows.append(
                    {
                        "ticker": str(p_row.get("ticker", pick.get("ticker", ""))),
                        "company": str(p_row.get("company", pick.get("company", ""))),
                        "product_id": str(p_row.get("product_id", "")),
                        "product_label": str(p_row.get("product_label", "")),
                        "scenario": scenario,
                        "year_offset": int(y),
                        "forecast_year": int(base_year + y) if base_year > 0 else "",
                        "demand_index": demand_index,
                        "product_demand_index": float(demand_index * share),
                        "demand_cagr_assumption": float(cagr),
                        "model_confidence": str(inputs["model_confidence"]),
                        "driver_signal_z": float(inputs["driver_signal_z"]) if np.isfinite(_to_float(inputs["driver_signal_z"])) else np.nan,
                        "anchor_growth_used": float(inputs["anchor_growth_used"]) if np.isfinite(_to_float(inputs["anchor_growth_used"])) else np.nan,
                        "available_metric_count": int(inputs["available_metric_count"]),
                    }
                )

    forecast = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "ticker": str(pick.get("ticker", "")),
                "company": str(pick.get("company", "")),
                "asof": str(asof),
                "base_demand_cagr": float(inputs["base_cagr"]),
                "bear_demand_cagr": float(inputs["bear_cagr"]),
                "bull_demand_cagr": float(inputs["bull_cagr"]),
                "model_confidence": str(inputs["model_confidence"]),
                "driver_signal_z": float(inputs["driver_signal_z"]) if np.isfinite(_to_float(inputs["driver_signal_z"])) else np.nan,
                "anchor_growth_used": float(inputs["anchor_growth_used"]) if np.isfinite(_to_float(inputs["anchor_growth_used"])) else np.nan,
                "available_metric_count": int(inputs["available_metric_count"]),
                "driver_metrics_used": str(inputs["driver_metrics_used"]),
                "forecast_source": forecast_source,
                "index_symbol": str((market_inputs or {}).get("index_symbol", "")),
                "market_history_days": int(pd.to_numeric(pd.Series([(market_inputs or {}).get("history_days")]), errors="coerce").fillna(0).iloc[0]),
                "market_ann_return": float((market_inputs or {}).get("ann_return")) if np.isfinite(_to_float((market_inputs or {}).get("ann_return"))) else np.nan,
                "market_ann_vol": float((market_inputs or {}).get("ann_vol")) if np.isfinite(_to_float((market_inputs or {}).get("ann_vol"))) else np.nan,
            }
        ]
    )
    return forecast, summary


def _fmt_num(x: object, digits: int = 3) -> str:
    v = _to_float(x)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}g}"


def _fmt_pct(x: object, digits: int = 1) -> str:
    v = _to_float(x)
    if not np.isfinite(v):
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def _annual_source_status(pick: pd.Series) -> tuple[str, str]:
    fin_src = str(pick.get("financials_source", "")).strip().lower()
    period_type = str(pick.get("reporting_period_type", "")).strip().lower()
    fcf_src = str(pick.get("fcf_source", "")).strip().lower()
    period_end = str(pick.get("financials_period_end", "")).strip()

    annual = ("reports_y" in fin_src) or ("annual" in period_type)
    has_r12 = ("reports_r12" in fcf_src) or ("r12" in fcf_src)

    if annual and has_r12:
        return (
            "Delvis",
            f"Arsdata fra {fin_src or 'ukjent'} ({period_type or 'ukjent'}) + FCF fra {fcf_src or 'ukjent'} (period_end={period_end or 'ukjent'})",
        )
    if annual:
        return (
            "Ja",
            f"Arsdata fra {fin_src or 'ukjent'} ({period_type or 'ukjent'}, period_end={period_end or 'ukjent'})",
        )
    return (
        "Nei",
        f"Kilde={fin_src or 'ukjent'}, period_type={period_type or 'ukjent'}, fcf_source={fcf_src or 'ukjent'}",
    )


def _ascii_sparkline(values: list[float]) -> str:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return "n/a"
    lo = min(finite)
    hi = max(finite)
    charset = " .:-=+*#%@"
    if hi <= lo:
        return "=" * min(len(finite), 48)
    out = []
    for v in finite[:48]:
        ratio = (v - lo) / (hi - lo)
        idx = int(round(ratio * (len(charset) - 1)))
        idx = max(0, min(idx, len(charset) - 1))
        out.append(charset[idx])
    return "".join(out)


def _price_sparkline_from_prices(prices_df: pd.DataFrame, pick: pd.Series, asof: str, max_points: int = 48) -> str:
    if prices_df is None or prices_df.empty:
        return "n/a"
    px = _canon_cols(prices_df)
    t_col = _pick(px, ["ticker", "symbol", "yahoo_ticker"])
    d_col = _pick(px, ["date", "price_date", "datetime", "timestamp", "time"])
    p_col = _pick(px, ["adj_close", "close", "price", "last"])
    if not t_col or not d_col or not p_col:
        return "n/a"

    candidates = [
        str(pick.get("yahoo_ticker", "")).strip(),
        str(pick.get("ticker", "")).strip(),
    ]
    candidates = [c for c in candidates if c]
    if not candidates:
        return "n/a"

    x = px.copy()
    x[d_col] = pd.to_datetime(x[d_col], errors="coerce")
    asof_ts = pd.to_datetime(asof, errors="coerce")
    x = x[x[d_col].notna()]
    if pd.notna(asof_ts):
        x = x[x[d_col] <= asof_ts]
    x[t_col] = x[t_col].astype(str).str.strip()
    x = x[x[t_col].isin(candidates)]
    if x.empty:
        return "n/a"

    x = x.sort_values(d_col)
    vals = pd.to_numeric(x[p_col], errors="coerce")
    vals = vals[np.isfinite(vals)].tolist()
    if len(vals) > max_points:
        step = max(1, int(len(vals) / max_points))
        vals = vals[::step][:max_points]
    return _ascii_sparkline(vals)


def _base_demand_series(demand_forecast: pd.DataFrame) -> pd.DataFrame:
    if demand_forecast is None or demand_forecast.empty:
        return pd.DataFrame(columns=["forecast_year", "demand_index"])
    d = demand_forecast.copy()
    d = d[d["scenario"].astype(str).str.lower() == "base"]
    if d.empty:
        return pd.DataFrame(columns=["forecast_year", "demand_index"])
    d["forecast_year"] = pd.to_numeric(d.get("forecast_year"), errors="coerce")
    d["product_demand_index"] = pd.to_numeric(d.get("product_demand_index"), errors="coerce")
    g = (
        d.groupby("forecast_year", dropna=True)["product_demand_index"]
        .sum()
        .reset_index()
        .rename(columns={"product_demand_index": "demand_index"})
        .sort_values("forecast_year")
    )
    return g


def _render_ascii_bar_chart(series_df: pd.DataFrame, width: int = 28) -> list[str]:
    if series_df is None or series_df.empty:
        return ["- Graf: ikke tilgjengelig."]
    vals = pd.to_numeric(series_df["demand_index"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return ["- Graf: ikke tilgjengelig."]
    vmax = float(vals.max())
    if vmax <= 0:
        return ["- Graf: ikke tilgjengelig."]
    lines: list[str] = []
    for _, r in series_df.iterrows():
        y = int(pd.to_numeric(pd.Series([r.get("forecast_year")]), errors="coerce").fillna(0).iloc[0])
        v = _to_float(r.get("demand_index"))
        if not np.isfinite(v):
            continue
        bar_n = max(1, int(round((v / vmax) * width)))
        lines.append(f"- {y}: {'#' * bar_n} {v:.1f}")
    return lines if lines else ["- Graf: ikke tilgjengelig."]


def _render_candidate_product_demand_md(
    products: pd.DataFrame,
    demand_forecast: pd.DataFrame,
    demand_summary: pd.DataFrame,
    market_position: pd.DataFrame | None = None,
) -> list[str]:
    lines: list[str] = []
    lines.append("## 3) Produkter og markedsforventning")
    lines.append("")
    lines.append("### Products and Demand Outlook")

    if products is None or products.empty:
        lines.append("- Produktdata: ikke tilgjengelig i strukturerte felt.")
        lines.append("")
        return lines

    product_cols = [c for c in ["product_label", "source_field", "value", "value_share", "detail"] if c in products.columns]
    lines.append(_md_table(products[product_cols], max_rows=20))

    if market_position is not None and not market_position.empty:
        lines.append("")
        lines.append("### Markedsposisjon")
        pos_cols = [c for c in ["market_position_metric", "value", "comment"] if c in market_position.columns]
        lines.append(_md_table(market_position[pos_cols], max_rows=20))

    if demand_summary is not None and not demand_summary.empty:
        s = demand_summary.iloc[0]
        base_cagr = _to_float(s.get("base_demand_cagr", np.nan))
        bear_cagr = _to_float(s.get("bear_demand_cagr", np.nan))
        bull_cagr = _to_float(s.get("bull_demand_cagr", np.nan))
        conf = str(s.get("model_confidence", ""))
        lines.append("")
        lines.append(
            f"- Demand model CAGR (bear/base/bull): {bear_cagr:.1%} / {base_cagr:.1%} / {bull_cagr:.1%} | confidence={conf}"
            if np.isfinite(base_cagr) and np.isfinite(bear_cagr) and np.isfinite(bull_cagr)
            else f"- Demand model confidence: {conf}"
        )
        metrics_txt = str(s.get("driver_metrics_used", "")).strip()
        if metrics_txt:
            lines.append(f"- Driver metrics used: {metrics_txt}")
        source_txt = str(s.get("forecast_source", "")).strip()
        if source_txt:
            lines.append(f"- Forecast source: {source_txt}")
        idx_txt = str(s.get("index_symbol", "")).strip()
        hist_days = int(pd.to_numeric(pd.Series([s.get("market_history_days")]), errors="coerce").fillna(0).iloc[0])
        ann_ret = _to_float(s.get("market_ann_return"))
        ann_vol = _to_float(s.get("market_ann_vol"))
        if idx_txt:
            ann_txt = (
                f", annualized return={ann_ret:.1%}, annualized vol={ann_vol:.1%}"
                if np.isfinite(ann_ret) and np.isfinite(ann_vol)
                else ""
            )
            lines.append(f"- Market data used: index={idx_txt}, history_days={hist_days}{ann_txt}")
        comment = "Ettersporsel ser robust ut i base/bull." if (np.isfinite(base_cagr) and base_cagr >= 0.05) else "Ettersporsel ser moderat/usikker ut."
        lines.append(f"- Kommentar: {comment}")

    if demand_forecast is not None and not demand_forecast.empty:
        base = demand_forecast[demand_forecast["scenario"].astype(str) == "base"].copy()
        base_cols = [c for c in ["product_label", "forecast_year", "demand_index", "product_demand_index", "demand_cagr_assumption"] if c in base.columns]
        if not base.empty and base_cols:
            lines.append("")
            lines.append("### Base Scenario (Demand Index)")
            lines.append(_md_table(base[base_cols], max_rows=50))
            lines.append("")
            lines.append("### Base Scenario (ASCII Chart)")
            lines.extend(_render_ascii_bar_chart(_base_demand_series(demand_forecast)))

    lines.append("")
    return lines


def _decision_schema_rows(
    pick: pd.Series,
    mos_min: float,
    mos_high: float,
    mad_min: float,
    max_price_age_days: int,
    required_fields: list[str],
    min_count: int,
    min_ratio: float,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(parameter: str, value: str, threshold: str, status: bool, comment: str = "") -> None:
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "threshold": threshold,
                "status": "OK" if bool(status) else "FAIL",
                "comment": comment,
            }
        )

    high_risk = int(pd.to_numeric(pd.Series([pick.get("high_risk_flag", 0)]), errors="coerce").fillna(0).iloc[0]) == 1
    mos_req = float(pd.to_numeric(pd.Series([pick.get("mos_req")]), errors="coerce").fillna(np.nan).iloc[0])
    mos = float(pd.to_numeric(pd.Series([pick.get("mos")]), errors="coerce").fillna(np.nan).iloc[0])
    add(
        "MoS",
        f"{mos:.1%}" if np.isfinite(mos) else "n/a",
        f">= {mos_req:.0%}" if np.isfinite(mos_req) else (f">= {mos_high:.0%}" if high_risk else f">= {mos_min:.0%}"),
        bool(np.isfinite(mos) and np.isfinite(mos_req) and mos >= mos_req),
        f"basis={pick.get('mos_basis', '')}",
    )

    price_age = pd.to_numeric(pd.Series([pick.get("stock_price_age_days")]), errors="coerce").iloc[0]
    add(
        "Aksjekurs ferskhet",
        f"{int(price_age)} dager" if np.isfinite(price_age) else "n/a",
        f"<= {max_price_age_days} dager",
        bool(_to_bool(pick.get("stock_price_stale")) is False),
        f"dato={pick.get('date', '')}",
    )

    idx_age = pd.to_numeric(pd.Series([pick.get("index_price_age_days")]), errors="coerce").iloc[0]
    add(
        "Indeks ferskhet",
        f"{int(idx_age)} dager" if np.isfinite(idx_age) else "n/a",
        f"<= {max_price_age_days} dager",
        bool(_to_bool(pick.get("index_price_stale")) is False),
        f"index={pick.get('relevant_index_symbol', '')}",
    )

    add(
        "Verdiskaping (ROIC-WACC 3y)",
        f"{float(pd.to_numeric(pd.Series([pick.get('roic_wacc_spread')]), errors='coerce').iloc[0]):.3%}" if np.isfinite(pd.to_numeric(pd.Series([pick.get("roic_wacc_spread")]), errors="coerce").iloc[0]) else "n/a",
        "> 0 i 3 år (konservativ bane)",
        _to_bool(pick.get("value_creation_ok")),
        "",
    )

    add(
        "Kvalitetsgate",
        f"weak_count={int(pd.to_numeric(pd.Series([pick.get('quality_weak_count')]), errors='coerce').fillna(-1).iloc[0])}",
        "< 2 svake indikatorer",
        _to_bool(pick.get("quality_gate_ok")),
        "",
    )

    add(
        "Teknisk gate",
        f"stock_ma200={_to_bool(pick.get('stock_ma200_ok'))}, index_ma200={_to_bool(pick.get('index_ma200_ok'))}, mad_ok={_to_bool(pick.get('stock_mad_ok'))}",
        "må bestå valgt teknisk regel",
        _to_bool(pick.get("technical_ok")),
        "",
    )

    avail_count = int(pd.to_numeric(pd.Series([pick.get("candidate_available_fields_count")]), errors="coerce").fillna(0).iloc[0])
    req_count = int(pd.to_numeric(pd.Series([pick.get("candidate_required_fields_count")]), errors="coerce").fillna(len(required_fields)).iloc[0])
    cov_ratio = pd.to_numeric(pd.Series([pick.get("candidate_data_coverage_ratio")]), errors="coerce").iloc[0]
    add(
        "Datatilstrekkelighet",
        f"{avail_count}/{req_count} ({(float(cov_ratio) if np.isfinite(cov_ratio) else 0.0):.0%})",
        f">= {min_count} felt og >= {min_ratio:.0%} dekning",
        _to_bool(pick.get("candidate_data_ok")),
        f"mangler={pick.get('candidate_data_missing_fields', '')}",
    )

    add(
        "Fundamental gate total",
        str(_to_bool(pick.get("fundamental_ok"))),
        "True",
        _to_bool(pick.get("fundamental_ok")),
        str(pick.get("reason_fundamental_fail", "")),
    )
    add(
        "Total eligibility",
        str(_to_bool(pick.get("eligible"))),
        "True",
        _to_bool(pick.get("eligible")),
        str(pick.get("reason_technical_fail", "")),
    )
    add(
        "Verdi-QA",
        f"alerts={int(pd.to_numeric(pd.Series([pick.get('value_qc_alert_count')]), errors='coerce').fillna(0).iloc[0])}, "
        f"unresolved={int(pd.to_numeric(pd.Series([pick.get('value_qc_unresolved_alert_count')]), errors='coerce').fillna(0).iloc[0])}",
        "unresolved=0",
        int(pd.to_numeric(pd.Series([pick.get("value_qc_unresolved_alert_count")]), errors="coerce").fillna(0).iloc[0]) == 0,
        str(pick.get("value_qc_unresolved_metrics", "")),
    )
    return pd.DataFrame(rows)


def _fetch_google_news_rss(
    query: str,
    timeout_sec: int = 20,
    hl: str = "en-US",
    gl: str = "US",
    ceid: str = "US:en",
) -> list[dict[str, str]]:
    q = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={urllib.parse.quote_plus(hl)}&gl={urllib.parse.quote_plus(gl)}&ceid={urllib.parse.quote_plus(ceid)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        xml_text = resp.read().decode("utf-8", errors="ignore")
    root = ET.fromstring(xml_text)
    out: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = ""
        src_node = item.find("{http://search.yahoo.com/mrss/}source")
        if src_node is not None and src_node.text:
            source = src_node.text.strip()
        if title and link:
            out.append({"title": title, "link": link, "pub_date": pub_date, "source": source})
    return out


def _run_media_red_flag_scan(
    asof: str,
    ticker: str,
    company: str,
    dec_cfg: dict,
) -> dict:
    mcfg = (dec_cfg.get("media_red_flags") or {})
    enabled = bool(mcfg.get("enabled", True))
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "headlines_checked": 0,
            "red_flag_count": 0,
            "red_flags": [],
            "headlines_sample": [],
        }

    days_back = int(mcfg.get("days_back", 30))
    max_headlines = int(mcfg.get("max_headlines", 30))
    terms_raw = mcfg.get("terms", DEFAULT_MEDIA_RED_FLAG_TERMS)
    terms = [str(x).strip().lower() for x in (terms_raw if isinstance(terms_raw, list) else DEFAULT_MEDIA_RED_FLAG_TERMS) if str(x).strip()]
    queries = []
    if company:
        queries.append(f'"{company}" aksje OR stock')
    if ticker:
        queries.append(f'"{ticker}" stock')
    if not queries:
        queries.append("stock warning")

    asof_dt = pd.to_datetime(asof, errors="coerce")
    seen: set[tuple[str, str]] = set()
    all_rows: list[dict[str, str]] = []
    try:
        locales = [("en-US", "US", "US:en"), ("sv", "SE", "SE:sv")]
        for q in queries:
            for hl, gl, ceid in locales:
                rows = _fetch_google_news_rss(q, hl=hl, gl=gl, ceid=ceid)
                for r in rows:
                    key = (r.get("title", ""), r.get("link", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    all_rows.append(r)
        all_rows = all_rows[:max_headlines]
    except (urllib.error.URLError, TimeoutError, ET.ParseError, ValueError) as exc:
        return {
            "enabled": True,
            "status": "error",
            "error": str(exc),
            "headlines_checked": 0,
            "red_flag_count": 0,
            "red_flags": [],
            "headlines_sample": all_rows[:10],
        }

    cutoff = None
    if pd.notna(asof_dt):
        cutoff = asof_dt.normalize() - pd.Timedelta(days=max(days_back, 1))

    red_flags: list[dict[str, str]] = []
    for row in all_rows:
        title = str(row.get("title", ""))
        ttl_l = title.lower()
        matched = [t for t in terms if t in ttl_l]
        if not matched:
            continue
        pub_raw = row.get("pub_date", "")
        keep = True
        if cutoff is not None and pub_raw:
            try:
                dt = parsedate_to_datetime(pub_raw)
                dt_naive = dt.replace(tzinfo=None)
                keep = pd.Timestamp(dt_naive).normalize() >= cutoff
            except Exception:
                keep = True
        if not keep:
            continue
        rec = dict(row)
        rec["matched_terms"] = ", ".join(matched)
        red_flags.append(rec)

    return {
        "enabled": True,
        "status": "ok",
        "headlines_checked": len(all_rows),
        "red_flag_count": len(red_flags),
        "red_flags": red_flags,
        "headlines_sample": all_rows[:10],
    }


def _media_assessment_from_scan(media_scan: dict) -> tuple[str, str]:
    status = str(media_scan.get("status", "unknown")).strip().lower()
    red_n = int(pd.to_numeric(pd.Series([media_scan.get("red_flag_count", 0)]), errors="coerce").fillna(0).iloc[0])

    if status == "error":
        return "Ukjent", "Media-scan feilet. Kjor manuell nyhetssjekk."
    if status == "disabled":
        return "Ikke vurdert", "Media-scan er deaktivert i config."
    if red_n >= 3:
        return "Hoy risiko", "Flere potensielt negative nyhetssignaler er funnet."
    if red_n >= 1:
        return "Moderat risiko", "Noen potensielt negative nyhetssignaler er funnet."
    return "Lav risiko", "Ingen tydelige red-flag treff i sjekkede overskrifter."


def _media_headlines_frame(media_scan: dict, max_rows: int = 10) -> pd.DataFrame:
    red_flags = media_scan.get("red_flags", []) or []
    sample = media_scan.get("headlines_sample", []) or []

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in red_flags:
        title = str(item.get("title", "")).strip()
        link = str(item.get("link", "")).strip()
        key = (title, link)
        if key in seen or (not title and not link):
            continue
        seen.add(key)
        rows.append(
            {
                "headline": title,
                "source": str(item.get("source", "")).strip(),
                "date": str(item.get("pub_date", "")).strip(),
                "classification": "RED_FLAG",
                "assessment": str(item.get("matched_terms", "")).strip(),
            }
        )
        if len(rows) >= max_rows:
            break

    if len(rows) < max_rows:
        for item in sample:
            title = str(item.get("title", "")).strip()
            link = str(item.get("link", "")).strip()
            key = (title, link)
            if key in seen or (not title and not link):
                continue
            seen.add(key)
            rows.append(
                {
                    "headline": title,
                    "source": str(item.get("source", "")).strip(),
                    "date": str(item.get("pub_date", "")).strip(),
                    "classification": "INFO",
                    "assessment": "Ingen red-flag term matchet",
                }
            )
            if len(rows) >= max_rows:
                break

    return pd.DataFrame(rows, columns=["headline", "source", "date", "classification", "assessment"])


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


def _write_top_candidates_report(run_dir: Path, asof: str, frame: pd.DataFrame, max_rows: int = 25) -> None:
    cols_pref = [
        "ticker",
        "company",
        "market_cap",
        "intrinsic_value",
        "mos",
        "quality_score",
        "adj_close",
        "above_ma200",
        "mad",
        "dq_fail_count",
        "dq_warn_count",
        "decision_reasons",
    ]
    cols = [c for c in cols_pref if c in frame.columns]
    short = frame[cols].head(int(max_rows)).copy() if cols else frame.head(int(max_rows)).copy()
    _atomic_write_csv(run_dir / "top_candidates.csv", short)

    md_lines = [
        f"# Top Candidates ({asof})",
        "",
        f"- Antall rader: {len(short)}",
        "- Full eksport: `decision.csv`",
        "",
        _md_table(short, max_rows=max_rows),
        "",
    ]
    _atomic_write_text(run_dir / "top_candidates.md", "\n".join(md_lines))


def _html_escape(x: object) -> str:
    return html.escape("" if x is None else str(x))


def _extract_first_number(text: object) -> Optional[float]:
    if text is None:
        return None
    s = str(text).replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_percent_value(text: object) -> Optional[float]:
    if text is None:
        return None
    s = str(text).replace(",", ".")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", s)
    if not m:
        return None
    try:
        return float(m.group(1)) / 100.0
    except Exception:
        return None


def _parse_bool_token(text: object) -> Optional[bool]:
    if text is None:
        return None
    s = str(text).strip().lower()
    if s in {"true", "ok", "pass", "ja", "yes"}:
        return True
    if s in {"false", "fail", "nei", "no"}:
        return False
    return None


def _metric_tone_and_comment(key: str, value: object, comment: object) -> tuple[str, str]:
    k = str(key).strip().lower()
    v = str(value).strip()
    c = str(comment).strip()
    v_pct = _extract_percent_value(v)
    c_pct = _extract_percent_value(c)
    v_num = _extract_first_number(v)
    c_num = _extract_first_number(c)
    vb = _parse_bool_token(v)

    if "mos" in k and v_pct is not None:
        if c_pct is not None:
            if v_pct >= c_pct:
                return "good", "Over krav"
            return "bad", "Under krav"
        if v_pct >= 0.30:
            return "good", "Sterk margin of safety"
        if v_pct >= 0.0:
            return "warn", "Moderat margin of safety"
        return "bad", "Negativ margin of safety"

    if "roic-wacc" in k and v_pct is not None:
        if v_pct > 0:
            return "good", "Verdiskaping positiv"
        return "bad", "Verdiskaping negativ"

    if "value qa unresolved" in k and v_num is not None:
        if int(round(v_num)) <= 0:
            return "good", "Ingen uforklarte avvik"
        return "bad", "Manuell verifisering anbefales"

    if "arsregnskap brukt" in k:
        s = v.lower()
        if s in {"ja", "yes"}:
            return "good", "Dekning fra arsrapport"
        if s in {"delvis", "partial"}:
            return "warn", "Delvis arsdekning"
        return "bad", "Manglende arsdekning"

    if "quality score" in k and v_num is not None:
        if v_num >= 0.50:
            return "good", "Hoy kvalitet"
        if v_num >= 0.0:
            return "warn", "Noytral kvalitet"
        return "bad", "Lav kvalitet"

    if "prisalder" in k and v_num is not None:
        thr = c_num if c_num is not None else 7.0
        if v_num <= thr:
            return "good", "Fersk kursdata"
        return "bad", "Kursdata er for gammel"

    if "above ma200" in k and vb is not None:
        if vb:
            return "good", "Trend over MA200"
        return "bad", "Under MA200"

    if "teknisk gate" in k and vb is not None:
        if vb:
            return "good", "Teknisk filter bestatt"
        return "bad", "Teknisk filter feilet"

    if k == "mad" and v_num is not None:
        thr = c_num if c_num is not None else -0.05
        if v_num >= thr:
            return "good", "Momentum innenfor krav"
        return "bad", "Momentum under terskel"

    if "relevant index" in k:
        if "true" in c.lower():
            return "good", "Markedstrend positiv"
        if "false" in c.lower():
            return "warn", "Markedstrend svak"
        return "neutral", "Ingen vurdering"

    return "neutral", "Ingen automatisk vurdering"


def _decorate_metric_table_for_html(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Nokkel", "Verdi", "Kommentar", "Status", "Vurdering"])
    out = df.copy()
    if "Nokkel" not in out.columns or "Verdi" not in out.columns:
        return out
    status_vals: list[str] = []
    verdicts: list[str] = []
    for _, row in out.iterrows():
        tone, verdict = _metric_tone_and_comment(row.get("Nokkel", ""), row.get("Verdi", ""), row.get("Kommentar", ""))
        status_vals.append(tone.upper())
        verdicts.append(verdict)
    out["Status"] = status_vals
    out["Vurdering"] = verdicts
    return out


def _svg_line_chart_from_demand(demand_df: pd.DataFrame) -> str:
    if demand_df is None or demand_df.empty:
        return "<p class='muted'>(for lite data til graf)</p>"
    if "demand_index" not in demand_df.columns:
        return "<p class='muted'>(mangler demand_index)</p>"

    d = demand_df.copy()
    d["demand_index"] = pd.to_numeric(d["demand_index"], errors="coerce")
    if "forecast_year" in d.columns:
        x_labels = d["forecast_year"].astype(str).tolist()
    elif "year_offset" in d.columns:
        x_labels = d["year_offset"].astype(str).tolist()
    else:
        x_labels = [str(i + 1) for i in range(len(d))]
    y_values = d["demand_index"].tolist()
    valid = [(x_labels[i], y_values[i]) for i in range(len(y_values)) if np.isfinite(_to_float(y_values[i]))]
    if len(valid) < 2:
        return "<p class='muted'>(for lite data til graf)</p>"

    labels = [str(v[0]) for v in valid]
    values = [float(v[1]) for v in valid]
    width = 820
    height = 260
    pad = 36
    xw = width - (2 * pad)
    yh = height - (2 * pad)
    ymin = float(min(values))
    ymax = float(max(values))
    if ymax <= ymin:
        ymax = ymin + 1.0

    pts: list[tuple[float, float]] = []
    for i, val in enumerate(values):
        x = pad + (xw * i / max(1, len(values) - 1))
        y = pad + yh * (1.0 - ((val - ymin) / (ymax - ymin)))
        pts.append((x, y))
    poly = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts])

    circles = []
    labels_html = []
    for i, (x, y) in enumerate(pts):
        circles.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' class='chart-dot' />")
        labels_html.append(
            f"<text x='{x:.1f}' y='{height - 12:.1f}' class='chart-label' text-anchor='middle'>{_html_escape(labels[i])}</text>"
        )
        labels_html.append(
            f"<text x='{x:.1f}' y='{max(12.0, y - 10.0):.1f}' class='chart-value' text-anchor='middle'>{values[i]:.1f}</text>"
        )

    y_ticks = []
    for k in range(5):
        ratio = k / 4.0
        val = ymin + (ymax - ymin) * (1.0 - ratio)
        y = pad + yh * ratio
        y_ticks.append(f"<line x1='{pad}' y1='{y:.1f}' x2='{width - pad}' y2='{y:.1f}' class='chart-grid' />")
        y_ticks.append(f"<text x='{pad - 8}' y='{y + 4:.1f}' class='chart-axis' text-anchor='end'>{val:.0f}</text>")

    return (
        f"<svg viewBox='0 0 {width} {height}' class='demand-chart' role='img' "
        f"aria-label='Base scenario demand index'>"
        + "".join(y_ticks)
        + f"<polyline points='{poly}' class='chart-line' />"
        + "".join(circles)
        + "".join(labels_html)
        + "</svg>"
    )


def _cell_tone_class(col: str, value: object, row: pd.Series) -> str:
    col_l = str(col).strip().lower()
    txt = str(value).strip()
    txt_l = txt.lower()

    if col_l == "status":
        if txt_l == "good":
            return "tone-good"
        if txt_l == "bad":
            return "tone-bad"
        if txt_l == "warn":
            return "tone-warn"
        return "tone-neutral"

    if col_l == "classification":
        if txt_l == "red_flag":
            return "tone-bad"
        if txt_l == "info":
            return "tone-neutral"

    if col_l == "assessment":
        if "ingen red-flag" in txt_l:
            return "tone-good"
        if "matchet term" in txt_l:
            return "tone-bad"
        return "tone-neutral"

    if col_l in {"verdi", "value"} and "nokkel" in [c.lower() for c in row.index.astype(str)]:
        key_col = [c for c in row.index if str(c).lower() == "nokkel"][0]
        comm_col = [c for c in row.index if str(c).lower() == "kommentar"]
        tone, _ = _metric_tone_and_comment(row.get(key_col, ""), value, row.get(comm_col[0], "") if comm_col else "")
        if tone == "good":
            return "tone-good"
        if tone == "bad":
            return "tone-bad"
        if tone == "warn":
            return "tone-warn"
        return "tone-neutral"

    if col_l == "vurdering":
        status_col = [c for c in row.index if str(c).lower() == "status"]
        if status_col:
            st = str(row.get(status_col[0], "")).strip().lower()
            if st == "good":
                return "tone-good"
            if st == "bad":
                return "tone-bad"
            if st == "warn":
                return "tone-warn"
    return ""


def _render_status_pill(text: str) -> str:
    t = str(text).strip().upper()
    if t == "GOOD":
        return "<span class='pill pill-good'>GRONN</span>"
    if t == "BAD":
        return "<span class='pill pill-bad'>ROD</span>"
    if t == "WARN":
        return "<span class='pill pill-warn'>GUL</span>"
    return "<span class='pill pill-neutral'>NOYTRAL</span>"


def _html_table_from_df(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p class='muted'>(no data)</p>"
    d = df.head(max_rows).copy()
    d = d.astype(object).where(d.notna(), "")
    cols = [str(c) for c in d.columns]
    head = "".join([f"<th>{_html_escape(c)}</th>" for c in cols])
    rows = []
    for i in range(len(d)):
        row = d.iloc[i]
        cells = []
        for c in cols:
            cls = _cell_tone_class(c, row[c], row)
            value_html = _render_status_pill(row[c]) if str(c).strip().lower() == "status" else _html_escape(row[c])
            cells.append(f"<td class='{cls}'>{value_html}</td>" if cls else f"<td>{value_html}</td>")
        tds = "".join(cells)
        rows.append(f"<tr>{tds}</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _metric_insights_html(df: pd.DataFrame, max_items: int = 4) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Ingen kommentarer.</p>"
    if "Status" not in df.columns or "Nokkel" not in df.columns:
        return "<p class='muted'>Ingen kommentarer.</p>"
    view = df.copy()
    view["Status"] = view["Status"].astype(str).str.upper()
    view = view[view["Status"].isin(["GOOD", "WARN", "BAD"])]
    if view.empty:
        return "<p class='muted'>Ingen avvik eller sterke signaler.</p>"
    lines = []
    for _, row in view.head(max_items).iterrows():
        tone = str(row.get("Status", "NEUTRAL")).lower()
        cls = "tone-neutral"
        if tone == "good":
            cls = "tone-good"
        elif tone == "warn":
            cls = "tone-warn"
        elif tone == "bad":
            cls = "tone-bad"
        key = _html_escape(row.get("Nokkel", ""))
        verdict = _html_escape(row.get("Vurdering", ""))
        lines.append(f"<li class='{cls}'><strong>{key}:</strong> {verdict}</li>")
    return "<ul class='insights'>" + "".join(lines) + "</ul>"


def _build_decision_report_html(
    asof: str,
    pick: pd.Series,
    fundamental_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    products_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    demand_chart_lines: list[str],
    media_df: pd.DataFrame,
    media_notes: list[str],
    top_candidates_rel: str = "top_candidates.md",
) -> str:
    ticker = str(pick.get("ticker", ""))
    company = str(pick.get("company", ""))
    header_sub = f"{ticker} | {company} | asof {asof}"
    fundamental_view = _decorate_metric_table_for_html(fundamental_df)
    stock_view = _decorate_metric_table_for_html(stock_df)
    media_view = media_df.copy()
    if media_view is not None and not media_view.empty and "classification" in media_view.columns:
        media_view["Status"] = (
            media_view["classification"]
            .astype(str)
            .str.upper()
            .map({"RED_FLAG": "BAD", "INFO": "NEUTRAL"})
            .fillna("NEUTRAL")
        )

    demand_ul = "".join([f"<li>{_html_escape(line)}</li>" for line in demand_chart_lines if str(line).strip()])
    media_ul = "".join([f"<li>{_html_escape(line)}</li>" for line in media_notes if str(line).strip()])
    demand_svg = _svg_line_chart_from_demand(demand_df)

    mos = _to_float(pick.get("mos"))
    mos_req = _to_float(pick.get("mos_req"))
    mos_tone = "neutral"
    if np.isfinite(mos):
        if np.isfinite(mos_req):
            mos_tone = "good" if mos >= mos_req else "bad"
        else:
            mos_tone = "good" if mos >= 0.30 else ("warn" if mos >= 0 else "bad")
    tech_ok = bool(_to_bool(pick.get("technical_ok")))
    dq_blocked = bool(_to_bool(pick.get("dq_blocked"))) or bool(_to_bool(pick.get("data_quality_fail")))
    unresolved_raw = pd.to_numeric(
        pd.Series(
            [
                pick.get(
                    "value_qc_unresolved_alert_count",
                    pick.get("value_qc_unresolved_count", pick.get("value_qc_unresolved", np.nan)),
                )
            ]
        ),
        errors="coerce",
    ).fillna(np.nan).iloc[0]
    if not np.isfinite(unresolved_raw):
        try:
            qa_rows = fundamental_view[
                fundamental_view.get("Nokkel", pd.Series("", index=fundamental_view.index))
                .astype(str)
                .str.strip()
                .eq("Value QA unresolved")
            ]
            if not qa_rows.empty:
                unresolved_raw = _extract_first_number(qa_rows.iloc[0].get("Verdi"))
        except Exception:
            unresolved_raw = np.nan
    unresolved = int(unresolved_raw) if np.isfinite(unresolved_raw) else 0
    red_flags = int((media_view.get("classification", pd.Series([], dtype=object)).astype(str).str.upper() == "RED_FLAG").sum())

    def _card(label: str, value: str, tone: str, detail: str) -> str:
        return (
            f"<article class='stat tone-{_html_escape(tone)}'>"
            f"<div class='stat-label'>{_html_escape(label)}</div>"
            f"<div class='stat-value'>{_html_escape(value)}</div>"
            f"<div class='stat-detail'>{_html_escape(detail)}</div>"
            f"</article>"
        )

    cards_html = "".join(
        [
            _card("MoS", _fmt_pct(mos, 1), mos_tone, f"Krav {_fmt_pct(mos_req, 0)}"),
            _card("Teknisk gate", "BESTATT" if tech_ok else "FEIL", "good" if tech_ok else "bad", "MA200/MAD + indeksregime"),
            _card("Data quality", "OK" if not dq_blocked else "BLOKKERT", "good" if not dq_blocked else "bad", "FAIL blokkerer kandidat"),
            _card("Value QA", f"{unresolved}", "good" if unresolved <= 0 else "bad", "Uforklarte avvik"),
            _card(
                "Nyhetsrisiko",
                f"{red_flags} red flags",
                "bad" if red_flags >= 2 else ("warn" if red_flags == 1 else "good"),
                "Fra headline-scan",
            ),
        ]
    )

    return f"""<!doctype html>
<html lang="no">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Decision Report {_html_escape(asof)}</title>
  <style>
    :root {{
      --bg-a: #f1f6ff;
      --bg-b: #fbfdff;
      --ink: #0f172a;
      --muted: #475569;
      --card: #ffffff;
      --line: #d8e3f2;
      --head: #edf3fb;
      --good: #166534;
      --good-bg: #ecfdf3;
      --bad: #b42318;
      --bad-bg: #fff1ef;
      --warn: #b54708;
      --warn-bg: #fff7e6;
      --accent: #0f5fd6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(1100px 520px at 90% -10%, #dbeafe 0%, rgba(219,234,254,0) 70%),
        linear-gradient(180deg, var(--bg-a), var(--bg-b));
      font-family: "Segoe UI", "Calibri", Arial, sans-serif;
      line-height: 1.45;
    }}
    .page {{ max-width: 1220px; margin: 24px auto; padding: 0 14px 24px 14px; }}
    .hero {{
      background: linear-gradient(135deg, #ffffff, #f9fbff);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }}
    h1 {{ margin: 0; font-size: 28px; letter-spacing: 0.2px; }}
    h2 {{ margin: 0 0 10px 0; font-size: 20px; }}
    h3 {{ margin: 12px 0 8px 0; font-size: 16px; }}
    .sub {{ color: var(--muted); margin-top: 4px; }}
    .hero-actions {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .btn {{
      display: inline-flex;
      align-items: center;
      border: 1px solid #c2d3ee;
      border-radius: 999px;
      padding: 7px 12px;
      text-decoration: none;
      color: #0b3d91;
      background: #eef4ff;
      font-size: 13px;
      font-weight: 600;
    }}
    .btn:hover {{ background: #e4edff; }}
    .stats-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .stat {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.4px; }}
    .stat-value {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
    .stat-detail {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
    .stat.tone-good {{ border-left: 5px solid var(--good); background: linear-gradient(90deg, var(--good-bg), #fff 36%); }}
    .stat.tone-bad {{ border-left: 5px solid var(--bad); background: linear-gradient(90deg, var(--bad-bg), #fff 36%); }}
    .stat.tone-warn {{ border-left: 5px solid var(--warn); background: linear-gradient(90deg, var(--warn-bg), #fff 36%); }}
    .stat.tone-neutral {{ border-left: 5px solid #64748b; }}
    .grid-2 {{ margin-top: 12px; display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.03);
    }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .section-spacer {{ margin-top: 12px; }}
    table {{
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      font-size: 13px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    th, td {{ padding: 8px 9px; text-align: left; vertical-align: top; border-bottom: 1px solid #e7edf6; }}
    th {{
      background: var(--head);
      color: #1f2937;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }}
    .pill-good {{ background: var(--good-bg); color: var(--good); }}
    .pill-bad {{ background: var(--bad-bg); color: var(--bad); }}
    .pill-warn {{ background: var(--warn-bg); color: var(--warn); }}
    .pill-neutral {{ background: #eef2f7; color: #475569; }}
    td.tone-good {{ background: #f1fcf6; color: #0f5132; font-weight: 600; }}
    td.tone-bad {{ background: #fff4f2; color: #8b1d18; font-weight: 600; }}
    td.tone-warn {{ background: #fff8eb; color: #8a4b08; font-weight: 600; }}
    td.tone-neutral {{ background: #f8fafc; color: #334155; }}
    .insights {{ margin: 10px 0 0 0; padding-left: 20px; }}
    .insights li {{ margin: 4px 0; padding: 4px 6px; border-radius: 8px; }}
    .demand-chart {{
      width: 100%;
      min-height: 220px;
      border: 1px solid #dbe6f7;
      border-radius: 12px;
      background: linear-gradient(180deg, #f8fbff, #ffffff);
      margin-bottom: 8px;
    }}
    .chart-grid {{ stroke: #e6edf7; stroke-width: 1; }}
    .chart-line {{ fill: none; stroke: var(--accent); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .chart-dot {{ fill: #ffffff; stroke: var(--accent); stroke-width: 2; }}
    .chart-label {{ font-size: 12px; fill: #475569; }}
    .chart-value {{ font-size: 11px; fill: #0f172a; font-weight: 700; }}
    @media (max-width: 760px) {{
      .hero {{ flex-direction: column; align-items: flex-start; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div>
        <h1>Resultatrapport</h1>
        <div class="sub">{_html_escape(header_sub)}</div>
      </div>
      <div class="hero-actions">
        <a class="btn" href="{_html_escape(top_candidates_rel)}">Top candidates</a>
        <a class="btn" href="decision.md">Tekstrapport</a>
        <a class="btn" href="decision.csv">Raw CSV</a>
      </div>
    </header>

    <section class="stats-grid">
      {cards_html}
    </section>

    <section class="grid-2">
      <article class="card">
        <h2>1) Fundamental analyse</h2>
        <p class="muted">Nokkeltall er hentet fra master_valued/valuation. Gronne felt er sterke signaler, rode felt er red flags.</p>
        {_html_table_from_df(fundamental_view, max_rows=30)}
        {_metric_insights_html(fundamental_view)}
      </article>
      <article class="card">
        <h2>2) Aksje analyse</h2>
        <p class="muted">Pris, trend og timing-gating. Teknisk analyse brukes kun for timing, ikke for intrinsic value.</p>
        {_html_table_from_df(stock_view, max_rows=30)}
        {_metric_insights_html(stock_view)}
      </article>
    </section>

    <section class="card section-spacer">
      <h2>3) Produkter og markedsforventning</h2>
      <p class="muted">Produktoversikt + enkel ettersporselsmodell (bear/base/bull). Grafen viser base-scenario.</p>
      {_html_table_from_df(products_df, max_rows=20)}
      <h3>Base scenario demand index</h3>
      {demand_svg}
      {_html_table_from_df(demand_df, max_rows=20)}
      <ul>{demand_ul}</ul>
    </section>

    <section class="card section-spacer">
      <h2>4) Nyheter og vurdering</h2>
      <p class="muted">Red flags i overskrifter markeres i rodt. Informasjon uten treff vises som noytral.</p>
      {_html_table_from_df(media_view, max_rows=20)}
      <ul>{media_ul}</ul>
    </section>
  </main>
</body>
</html>
"""


def _write_ma200_sanity_report(run_dir: Path, asof: str, df: pd.DataFrame) -> None:
    stock_price = pd.to_numeric(df.get("adj_close"), errors="coerce")
    stock_ma200 = pd.to_numeric(df.get("ma200"), errors="coerce")

    bench_key = df.get("relevant_index_key", pd.Series("", index=df.index)).astype(str).str.strip()
    bench_scope = bench_key.ne("")
    bench_price = pd.to_numeric(df.get("index_price"), errors="coerce")
    bench_ma200 = pd.to_numeric(df.get("index_ma200"), errors="coerce")

    def _row(scope: str, metric: str, numerator: int, denominator: int) -> dict:
        share = float(numerator) / float(denominator) if int(denominator) > 0 else float("nan")
        return {
            "scope": scope,
            "metric": metric,
            "numerator": int(numerator),
            "denominator": int(denominator),
            "share": share,
        }

    stock_total = int(len(df))
    stock_evaluable = stock_price.notna() & stock_ma200.notna()
    bench_total = int(bench_scope.sum())
    bench_ma200_in_scope = bench_scope & bench_ma200.isna()
    bench_evaluable = bench_scope & bench_price.notna() & bench_ma200.notna()

    rows = [
        _row("stock", "ma200_nan_rows", int(stock_ma200.isna().sum()), stock_total),
        _row("stock", "price_gt_ma200_rows", int((stock_evaluable & (stock_price > stock_ma200)).sum()), int(stock_evaluable.sum())),
        _row("benchmark", "ma200_nan_rows", int(bench_ma200_in_scope.sum()), bench_total),
        _row("benchmark", "price_gt_ma200_rows", int((bench_evaluable & (bench_price > bench_ma200)).sum()), int(bench_evaluable.sum())),
    ]
    report = pd.DataFrame(rows)
    _atomic_write_csv(run_dir / "ma200_sanity_report.csv", report)

    def _fmt(scope: str, metric: str) -> str:
        r = report[(report["scope"] == scope) & (report["metric"] == metric)].iloc[0]
        num = int(r["numerator"])
        den = int(r["denominator"])
        if den > 0:
            return f"{num}/{den} ({(float(num) / float(den)):.1%})"
        return f"{num}/{den} (n/a)"

    lines = [
        f"# MA200 Sanity Report ({asof})",
        "",
        "## Stock",
        f"- MA200=NaN: {_fmt('stock', 'ma200_nan_rows')}",
        f"- price>MA200 (blant evaluerbare rader): {_fmt('stock', 'price_gt_ma200_rows')}",
        "",
        "## Benchmark",
        f"- MA200=NaN (blant rader med relevant benchmark): {_fmt('benchmark', 'ma200_nan_rows')}",
        f"- price>MA200 (blant evaluerbare benchmark-rader): {_fmt('benchmark', 'price_gt_ma200_rows')}",
        "",
        "Formål: skille reelt regime (trend) fra datamangler/warmup.",
    ]
    _atomic_write_text(run_dir / "ma200_sanity_report.md", "\n".join(lines))


def _join_reasons(parts: list[str]) -> str:
    return "; ".join([p for p in parts if p])


def _field_source(field: str) -> str:
    if field.startswith("index_"):
        return "prices.parquet(index)"
    if field in {"adj_close", "ma21", "ma200", "mad", "date"}:
        return "master_valued/prices"
    return "master_valued"


def _pick_existing_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def _detect_group_key(df: pd.DataFrame) -> tuple[pd.Series, str, str]:
    sector_col = _pick_existing_col(df, ["sector", "gics_sector", "sector_name", "sector_code"])
    if sector_col:
        return df[sector_col].astype(str).str.strip(), sector_col, "SECTOR"
    industry_col = _pick_existing_col(df, ["industry_group", "industry", "gics_industry_group"])
    if industry_col:
        return df[industry_col].astype(str).str.strip(), industry_col, "INDUSTRY_FALLBACK"
    return pd.Series("ALL_MARKET", index=df.index), "", "NO_SECTOR_GROUP"


def _metadata_source_fields(df: pd.DataFrame) -> str:
    meta_cols = [
        "sector", "industry_group", "industry", "currency", "unit", "period_type", "report_period", "report_date", "source",
    ]
    present = [c for c in meta_cols if c in df.columns]
    return ",".join(present) if present else "MISSING_METADATA"


def _append_dq_event(events: list[dict], **kwargs) -> None:
    events.append(
        {
            "asof": kwargs.get("asof", ""),
            "date": kwargs.get("date", ""),
            "ticker": kwargs.get("ticker", ""),
            "ins_id": kwargs.get("ins_id", ""),
            "sector": kwargs.get("sector", ""),
            "rule_id": kwargs.get("rule_id", ""),
            "severity": kwargs.get("severity", "WARN"),
            "field": kwargs.get("field", ""),
            "value": kwargs.get("value", np.nan),
            "group_key": kwargs.get("group_key", ""),
            "group_n": int(kwargs.get("group_n", 0) or 0),
            "source_fields": kwargs.get("source_fields", "MISSING_METADATA"),
            "detail": kwargs.get("detail", ""),
            "row_index": kwargs.get("row_index", -1),
        }
    )


def _run_data_quality_checks(df: pd.DataFrame, dec_cfg: dict, asof: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dq_cfg = dec_cfg.get("data_quality", {}) or {}
    critical_fields = dq_cfg.get("critical_fields", DEFAULT_CANDIDATE_REQUIRED_FIELDS)
    if isinstance(critical_fields, str):
        critical_fields = [x.strip() for x in critical_fields.split(",") if x.strip()]
    critical_fields = [str(x).strip() for x in critical_fields if str(x).strip()]

    outlier_min_samples = int(dq_cfg.get("outlier_min_samples", 10))
    outlier_mad_threshold = float(dq_cfg.get("outlier_mad_threshold", 6.0))
    stale_days_threshold = int(dq_cfg.get("stale_fundamentals_days", 450))
    group_s, group_col, group_mode = _detect_group_key(df)
    source_fields = _metadata_source_fields(df)
    audit_rows: list[dict] = []

    px_col = _pick_existing_col(df, ["adj_close", "price"])
    sh_col = _pick_existing_col(df, ["shares_outstanding", "shares_out", "shares"])
    outlier_metrics = [
        ("fcf_yield", _pick_existing_col(df, ["fcf_yield"])),
        ("roic", _pick_existing_col(df, ["roic", "roic_dec", "roe"])),
        ("operating_margin", _pick_existing_col(df, ["operating_margin", "op_margin", "ebit_margin"])),
        ("quality_score", _pick_existing_col(df, ["quality_score"])),
    ]

    for i, row in df.iterrows():
        ticker = str(row.get("ticker", row.get("yahoo_ticker", "")))
        ins_id = row.get("ins_id", "")
        row_date = row.get("date", "")
        sector_val = str(row.get(group_col, "")) if group_col else ""

        for field in critical_fields:
            val = _to_float(row.get(field, np.nan))
            if (field not in df.columns) or (not np.isfinite(val)):
                _append_dq_event(
                    audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                    rule_id=f"DQ_CRITICAL_PRESENT_{field}", severity="FAIL", field=field, value=row.get(field, np.nan),
                    group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields, detail=f"missing_or_non_finite:{field}"
                )

        if px_col and np.isfinite(_to_float(row.get(px_col))) and _to_float(row.get(px_col)) <= 0:
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_PRICE_NON_POSITIVE", severity="FAIL", field=px_col, value=row.get(px_col),
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields, detail="price<=0")
        if "market_cap" in df.columns and np.isfinite(_to_float(row.get("market_cap"))) and _to_float(row.get("market_cap")) <= 0:
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_MARKET_CAP_NON_POSITIVE", severity="FAIL", field="market_cap", value=row.get("market_cap"),
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields, detail="market_cap<=0")
        if sh_col and np.isfinite(_to_float(row.get(sh_col))) and _to_float(row.get(sh_col)) <= 0:
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_SHARES_NON_POSITIVE", severity="FAIL", field=sh_col, value=row.get(sh_col),
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields, detail="shares<=0")

        # FCF/EBITDA implausibility check (FCF > 3× EBITDA = likely currency mismatch or data error)
        if row.get("fcf_implausible") is True or row.get("fcf_implausible") == 1:
            fe_ratio_val = _to_float(row.get("fcf_ebitda_ratio", np.nan))
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_FCF_IMPLAUSIBLE", severity="WARN", field="fcf_used_millions", value=fe_ratio_val,
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields,
                             detail=f"FCF/EBITDA={fe_ratio_val:.1f}x — sannsynlig valuta-mismatch eller datafeil")

        # ROIC implausibility check (> 100% after normalization = likely scale/data error)
        if row.get("roic_implausible") is True or row.get("roic_implausible") == 1:
            roic_val = _to_float(row.get("roic_implausible_value", row.get("roic_dec", np.nan)))
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_ROIC_IMPLAUSIBLE", severity="WARN", field="roic", value=roic_val,
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields,
                             detail=f"roic={roic_val:.2%} after normalization — likely scale error or bad data")

        intrinsic = _to_float(row.get("intrinsic_value", np.nan))
        mos = _to_float(row.get("mos", np.nan))
        mos_req = _to_float(row.get("mos_req", np.nan))
        if np.isfinite(mos_req) and not np.isfinite(intrinsic):
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_INTRINSIC_MISSING_WHEN_REQUIRED", severity="FAIL", field="intrinsic_value", value=row.get("intrinsic_value", np.nan),
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields, detail="intrinsic_missing_when_mos_required")
        px_val = _to_float(row.get(px_col, np.nan)) if px_col else float("nan")
        if np.isfinite(intrinsic) and np.isfinite(px_val) and not np.isfinite(mos):
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row_date, ticker=ticker, ins_id=ins_id, sector=sector_val,
                             rule_id="DQ_MOS_NAN_WITH_INPUTS", severity="FAIL", field="mos", value=row.get("mos", np.nan),
                             group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields, detail="mos_nan_when_intrinsic_and_price_exist")

    # stale fundamentals warning
    rep_col = _pick_existing_col(df, ["report_period", "report_date", "fundamental_date", "period_end_date"])
    asof_ts = pd.to_datetime(asof, errors="coerce")
    if rep_col and pd.notna(asof_ts):
        rep_ts = pd.to_datetime(df[rep_col], errors="coerce")
        ages = (asof_ts - rep_ts).dt.days
        stale_idx = ages[ages > stale_days_threshold].index
        for i in stale_idx:
            row = df.loc[i]
            _append_dq_event(audit_rows, row_index=i, asof=asof, date=row.get("date", ""), ticker=str(row.get("ticker", "")), ins_id=row.get("ins_id", ""),
                             sector=str(row.get(group_col, "")) if group_col else "", rule_id="DQ_STALE_FUNDAMENTALS", severity="WARN",
                             field=rep_col, value=row.get(rep_col), group_key=str(group_s.loc[i]), group_n=0, source_fields=source_fields,
                             detail=f"age_days={int(ages.loc[i])} threshold={stale_days_threshold}")

    # sector/industry-relative robust outlier warnings
    for metric_name, metric_col in outlier_metrics:
        if not metric_col:
            continue
        vals = pd.to_numeric(df[metric_col], errors="coerce")
        for grp, idx in group_s.groupby(group_s).groups.items():
            grp_vals = vals.loc[idx]
            finite = grp_vals[np.isfinite(grp_vals)]
            grp_n = int(finite.shape[0])
            if grp_n == 0:
                continue
            if grp_n < outlier_min_samples:
                for i in finite.index:
                    row = df.loc[i]
                    _append_dq_event(audit_rows, row_index=i, asof=asof, date=row.get("date", ""), ticker=str(row.get("ticker", "")), ins_id=row.get("ins_id", ""),
                                     sector=str(row.get(group_col, "")) if group_col else "", rule_id=f"DQ_OUTLIER_LOW_SAMPLE_{metric_name}", severity="WARN",
                                     field=metric_col, value=vals.loc[i], group_key=str(grp), group_n=grp_n, source_fields=source_fields,
                                     detail=f"LOW_SAMPLE_SIZE|min_n={outlier_min_samples}")
                continue
            median = float(finite.median())
            mad = float((finite - median).abs().median())
            if (not np.isfinite(mad)) or mad <= 0:
                continue
            robust_z = 0.6745 * (grp_vals - median).abs() / mad
            hit_idx = robust_z[robust_z >= outlier_mad_threshold].index
            for i in hit_idx:
                row = df.loc[i]
                detail = f"robust_z={float(robust_z.loc[i]):.3f} threshold={outlier_mad_threshold}"
                if group_mode == "NO_SECTOR_GROUP":
                    detail = f"{detail}|NO_SECTOR_GROUP"
                _append_dq_event(audit_rows, row_index=i, asof=asof, date=row.get("date", ""), ticker=str(row.get("ticker", "")), ins_id=row.get("ins_id", ""),
                                 sector=str(row.get(group_col, "")) if group_col else "", rule_id=f"DQ_OUTLIER_ROBUST_{metric_name}", severity="WARN",
                                 field=metric_col, value=vals.loc[i], group_key=str(grp), group_n=grp_n, source_fields=source_fields, detail=detail)

    audit = pd.DataFrame(audit_rows)
    flags = pd.DataFrame(index=df.index)
    flags["dq_fail_count"] = 0
    flags["dq_warn_count"] = 0
    flags["dq_blocked"] = False
    flags["dq_fail_reasons"] = ""
    flags["dq_warn_reasons"] = ""

    if audit.empty:
        return flags, audit

    fail_counts = audit[audit["severity"] == "FAIL"].groupby("row_index").size()
    warn_counts = audit[audit["severity"] == "WARN"].groupby("row_index").size()
    fail_reasons = audit[audit["severity"] == "FAIL"].groupby("row_index")["rule_id"].apply(lambda x: ";".join(sorted(set(x.astype(str)))))
    warn_reasons = audit[audit["severity"] == "WARN"].groupby("row_index")["rule_id"].apply(lambda x: ";".join(sorted(set(x.astype(str)))))

    flags["dq_fail_count"] = fail_counts.reindex(df.index).fillna(0).astype(int)
    flags["dq_warn_count"] = warn_counts.reindex(df.index).fillna(0).astype(int)
    flags["dq_blocked"] = flags["dq_fail_count"] > 0
    flags["dq_fail_reasons"] = fail_reasons.reindex(df.index).fillna("")
    flags["dq_warn_reasons"] = warn_reasons.reindex(df.index).fillna("")

    # backward-compatible aliases
    flags["data_quality_fail"] = flags["dq_blocked"]
    flags["data_quality_fail_count"] = flags["dq_fail_count"]
    flags["data_quality_warn_count"] = flags["dq_warn_count"]
    flags["data_quality_fail_reasons"] = flags["dq_fail_reasons"]
    flags["data_quality_warn_reasons"] = flags["dq_warn_reasons"]
    return flags, audit


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
    if int(x.notna().sum()) == 0:
        return x
    ax = x.abs()
    med = float(ax.median(skipna=True))
    q75 = float(ax.quantile(0.75))
    q95 = float(ax.quantile(0.95))
    share_gt_1 = float((ax > 1.0).mean())
    share_gt_2 = float((ax > 2.0).mean())

    # Heuristic:
    # - clear percent-point scale (median > 2)
    # - or mixed-scale series with substantial mass above 1-2.
    should_scale = (
        (np.isfinite(med) and med > 2.0) or
        (np.isfinite(q75) and q75 > 1.0 and share_gt_1 >= 0.20) or
        (np.isfinite(q95) and q95 > 2.0 and share_gt_2 >= 0.05)
    )
    if should_scale:
        x = x / 100.0
    return x


def _evaluate_numeric_rule(values: pd.Series, rule) -> tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(values, errors="coerce")
    available = s.notna()
    ok = pd.Series(False, index=s.index, dtype=bool)
    if bool(available.any()):
        ok.loc[available] = rule(s.loc[available]).astype(bool)
    return ok, available


def _baseline_quality_score(df: pd.DataFrame) -> pd.Series:
    comps: list[pd.Series] = []
    wts: list[float] = []
    if "roic" in df.columns and pd.to_numeric(df["roic"], errors="coerce").notna().any():
        comps.append(zscore(pd.to_numeric(df["roic"], errors="coerce")))
        wts.append(0.60)
    elif "roic_current" in df.columns and pd.to_numeric(df["roic_current"], errors="coerce").notna().any():
        comps.append(zscore(pd.to_numeric(df["roic_current"], errors="coerce")))
        wts.append(0.60)

    if "fcf_yield" in df.columns and pd.to_numeric(df["fcf_yield"], errors="coerce").notna().any():
        comps.append(zscore(pd.to_numeric(df["fcf_yield"], errors="coerce")))
        wts.append(0.40)

    if not comps:
        return pd.Series(0.0, index=df.index, dtype=float)
    w = np.array(wts, dtype=float)
    w = w / w.sum()
    return sum(w[i] * comps[i] for i in range(len(comps)))


def _dividend_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    def _apply(name: str, values: pd.Series, rule) -> None:
        ok, available = _evaluate_numeric_rule(values, rule)
        out[f"dividend_{name}_ok"] = ok.astype(int)
        out[f"dividend_{name}_available"] = available.astype(int)

    hist = _as_num_series(
        df,
        ["dividend_history_years", "dividend_streak_years", "dividend_years", "dividend_paid_years"],
    )
    _apply("dividend_history", hist, lambda s: s >= 5.0)

    div_growth = _to_decimal_rate(
        _as_num_series(
            df,
            ["dividend_growth", "dividend_growth_5y", "dividend_cagr_5y", "dps_growth_5y"],
        )
    )
    _apply("dividend_growth", div_growth, lambda s: s > 0.0)

    payout = _to_decimal_rate(_as_num_series(df, ["payout_ratio", "dividend_payout_ratio"]))
    _apply("payout_ratio", payout, lambda s: (s > 0.0) & (s <= 0.75))

    share_growth = _to_decimal_rate(
        _as_num_series(
            df,
            ["share_count_growth", "share_count_growth_5y", "shares_outstanding_growth_5y", "shares_dilution_rate"],
        )
    )
    _apply("share_count", share_growth, lambda s: s <= 0.02)

    market_value = _as_num_series(df, ["market_cap", "market_cap_current"])
    _apply("market_value", market_value, lambda s: s >= 1_000_000_000.0)

    fcf_margin = _to_decimal_rate(_as_num_series(df, ["fcf_margin", "free_cash_flow_margin", "fcf_margin_ttm"]))
    _apply("fcf_margin", fcf_margin, lambda s: s >= 0.05)

    ocf_margin = _to_decimal_rate(_as_num_series(df, ["ocf_margin", "operating_cash_flow_margin", "operating_cf_margin"]))
    _apply("ocf_margin", ocf_margin, lambda s: s >= 0.08)

    profit_margin = _to_decimal_rate(_as_num_series(df, ["profit_margin", "net_margin", "net_profit_margin"]))
    _apply("profit_margin", profit_margin, lambda s: s >= 0.05)

    roa = _to_decimal_rate(_as_num_series(df, ["roa", "return_on_assets"]))
    _apply("roa", roa, lambda s: s >= 0.05)

    div_yield = _to_decimal_rate(_as_num_series(df, ["dividend_yield", "yield", "dividend_yield_ttm"]))
    _apply("yield", div_yield, lambda s: s >= 0.02)

    pe = _as_num_series(df, ["p_e_current", "p_e", "pe", "pe_ratio"])
    _apply("pe", pe, lambda s: (s > 0.0) & (s <= 25.0))

    ok_cols = [f"dividend_{k}_ok" for k, _ in DIVIDEND_CRITERIA]
    available_cols = [f"dividend_{k}_available" for k, _ in DIVIDEND_CRITERIA]
    out["dividend_score"] = out[ok_cols].sum(axis=1)
    out["dividend_criteria_available_count"] = out[available_cols].sum(axis=1)
    out["dividend_criteria_missing_count"] = len(DIVIDEND_CRITERIA) - out["dividend_criteria_available_count"]

    reason_series = pd.Series("", index=df.index, dtype=object)
    for key, _ in DIVIDEND_CRITERIA:
        ok_col = f"dividend_{key}_ok"
        available_col = f"dividend_{key}_available"
        fail_mask = out[available_col].astype(bool) & ~out[ok_col].astype(bool)
        missing_mask = ~out[available_col].astype(bool)
        reason_series.loc[fail_mask] = reason_series.loc[fail_mask].map(
            lambda x, k=key: _join_reasons([x, f"{k}_fail"])
        )
        reason_series.loc[missing_mask] = reason_series.loc[missing_mask].map(
            lambda x, k=key: _join_reasons([x, f"{k}_missing"])
        )
    out["dividend_reason"] = reason_series
    return out


def _graham_strategy_score(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    def _apply(name: str, values: pd.Series, rule) -> None:
        ok, available = _evaluate_numeric_rule(values, rule)
        out[f"graham_{name}_ok"] = ok.astype(int)
        out[f"graham_{name}_available"] = available.astype(int)

    market_cap = _as_num_series(df, ["market_cap", "market_cap_current"])
    _apply("size", market_cap, lambda s: s >= 2_000_000_000.0)

    nd_ebitda = _as_num_series(df, ["nd_ebitda", "n_debt_ebitda_current"])
    net_debt = _as_num_series(df, ["net_debt", "net_debt_current", "net_debt_used"])
    mcap_pos = market_cap.where(market_cap > 0)
    fin_available = nd_ebitda.notna() | (net_debt.notna() & mcap_pos.notna())
    fin_ok = (
        (~nd_ebitda.notna() | (nd_ebitda <= 3.0)) &
        (~(net_debt.notna() & mcap_pos.notna()) | (net_debt <= mcap_pos))
    )
    out["graham_financial_strength_ok"] = (fin_available & fin_ok).astype(int)
    out["graham_financial_strength_available"] = fin_available.astype(int)

    profit_margin = _to_decimal_rate(_as_num_series(df, ["profit_margin", "net_margin", "net_profit_margin"]))
    roic = _to_decimal_rate(_as_num_series(df, ["roic", "roic_current"]))
    earn_available = profit_margin.notna() | roic.notna()
    earn_ok = (
        (~profit_margin.notna() | (profit_margin > 0.0)) &
        (~roic.notna() | (roic > 0.0))
    )
    out["graham_earning_stability_ok"] = (earn_available & earn_ok).astype(int)
    out["graham_earning_stability_available"] = earn_available.astype(int)

    div_hist = _as_num_series(df, ["dividend_history_years", "dividend_streak_years", "dividend_years", "dividend_paid_years"])
    _apply("dividend_history", div_hist, lambda s: s >= 10.0)

    profit_growth = _to_decimal_rate(
        _as_num_series(df, ["profit_growth_5y", "earnings_growth_5y", "eps_growth_5y", "net_income_growth_5y"])
    )
    _apply("profit_growth", profit_growth, lambda s: s > 0.0)

    pe = _as_num_series(df, ["p_e_current", "p_e", "pe", "pe_ratio"])
    _apply("moderate_pe", pe, lambda s: (s > 0.0) & (s <= 15.0))

    pb = _as_num_series(df, ["p_b_current", "p_b", "pb_ratio", "price_to_book"])
    ps = _as_num_series(df, ["p_s_current", "p_s", "ps", "price_to_sales"])
    eq_available = pb.notna() | ps.notna()
    eq_ok = (
        (pb.notna() & (pb > 0.0) & (pb <= 1.5)) |
        (~pb.notna() & ps.notna() & (ps > 0.0) & (ps <= 2.0))
    )
    out["graham_moderate_equity_price_ok"] = (eq_available & eq_ok).astype(int)
    out["graham_moderate_equity_price_available"] = eq_available.astype(int)

    ok_cols = [f"graham_{k}_ok" for k, _ in GRAHAM_CRITERIA]
    available_cols = [f"graham_{k}_available" for k, _ in GRAHAM_CRITERIA]
    out["graham_score"] = out[ok_cols].sum(axis=1)
    out["graham_criteria_available_count"] = out[available_cols].sum(axis=1)
    out["graham_criteria_missing_count"] = len(GRAHAM_CRITERIA) - out["graham_criteria_available_count"]

    reason_series = pd.Series("", index=df.index, dtype=object)
    for key, _ in GRAHAM_CRITERIA:
        ok_col = f"graham_{key}_ok"
        available_col = f"graham_{key}_available"
        fail_mask = out[available_col].astype(bool) & ~out[ok_col].astype(bool)
        missing_mask = ~out[available_col].astype(bool)
        reason_series.loc[fail_mask] = reason_series.loc[fail_mask].map(
            lambda x, k=key: _join_reasons([x, f"graham_{k}_fail"])
        )
        reason_series.loc[missing_mask] = reason_series.loc[missing_mask].map(
            lambda x, k=key: _join_reasons([x, f"graham_{k}_missing"])
        )
    out["graham_reason"] = reason_series
    return out


def _build_quality_block(df: pd.DataFrame, dec_cfg: dict) -> pd.DataFrame:
    strategy_raw = str(dec_cfg.get("quality_strategy", "baseline")).strip().lower()
    use_dividend = strategy_raw in {"dividend", "dividend_quality"}
    use_graham = strategy_raw in {"graham", "graham_strategy"}

    out = pd.DataFrame(index=df.index)
    if use_dividend:
        div = _dividend_quality_score(df)
        for c in div.columns:
            out[c] = div[c]
        out["quality_score"] = pd.to_numeric(out["dividend_score"], errors="coerce").fillna(0.0)
        out["quality_strategy"] = "dividend_quality"
    elif use_graham:
        graham = _graham_strategy_score(df)
        for c in graham.columns:
            out[c] = graham[c]
        out["quality_score"] = pd.to_numeric(out["graham_score"], errors="coerce").fillna(0.0)
        out["quality_strategy"] = "graham_strategy"
    else:
        out["quality_score"] = _baseline_quality_score(df)
        out["quality_strategy"] = "baseline"
    return out


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
    # Sanity check: ROIC > 100% after normalization is almost certainly a data/scale error
    implausible_roic = roic.notna() & (roic.abs() > 1.0)
    out["roic_implausible"] = implausible_roic
    out["roic_implausible_value"] = roic.where(implausible_roic)
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
    max_price_age_days = int(dec_cfg.get("max_price_age_days", 7))
    if max_price_age_days < 0:
        max_price_age_days = 0
    idx_price = pd.to_numeric(out.get("index_price"), errors="coerce")
    idx_ma200 = pd.to_numeric(out.get("index_ma200"), errors="coerce")
    idx_mad = pd.to_numeric(out.get("index_mad"), errors="coerce")
    idx_price_date = pd.to_datetime(out.get("index_price_date", pd.Series(pd.NaT, index=out.index)), errors="coerce")
    asof_dt = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_dt):
        asof_dt = pd.Timestamp(datetime.utcnow().date())
    idx_age_days = pd.to_numeric((asof_dt.normalize() - idx_price_date.dt.normalize()).dt.days, errors="coerce")
    idx_price_stale = idx_price_date.isna() | idx_age_days.gt(max_price_age_days)
    idx_has_price_ma = idx_price.notna() & idx_ma200.notna()
    idx_has_mad = idx_mad.notna()
    idx_above = idx_price > idx_ma200

    out["index_above_ma200"] = pd.Series(
        np.where(idx_has_price_ma, idx_above, pd.NA),
        index=out.index,
        dtype="boolean",
    )
    out["index_price_age_days"] = idx_age_days
    out["index_price_stale"] = idx_price_stale
    out["index_data_ok"] = out["relevant_index_key"].astype(str).ne("") & idx_has_price_ma & ~idx_price_stale
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
    asof_dt = pd.to_datetime(ctx.asof, errors="coerce")
    if pd.isna(asof_dt):
        raise SchemaError(f"decision: invalid asof '{ctx.asof}'")
    mos_min = float(dec_cfg.get("mos_min", 0.30))
    mos_high = float(dec_cfg.get("mos_high_uncertainty", 0.40))
    require_above_ma200 = bool(dec_cfg.get("require_above_ma200", True))
    mad_min = float(dec_cfg.get("mad_min", -0.05))
    min_fresh_price_coverage = float(dec_cfg.get("min_fresh_price_coverage", 0.50))
    max_price_age_days = int(dec_cfg.get("max_price_age_days", 7))
    if max_price_age_days < 0:
        max_price_age_days = 0
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
    stock_price_date = pd.to_datetime(df.get("date", pd.Series(pd.NaT, index=df.index)), errors="coerce")
    stock_price_age_days = pd.to_numeric((asof_dt.normalize() - stock_price_date.dt.normalize()).dt.days, errors="coerce")
    stock_price_stale = stock_price_date.isna() | stock_price_age_days.gt(max_price_age_days)
    fresh_stock_mask = stock_price.notna() & ~stock_price_stale
    fresh_stock_coverage = float(fresh_stock_mask.mean()) if len(df) > 0 else 0.0
    global_price_data_ok = bool(fresh_stock_coverage >= min_fresh_price_coverage)
    if not global_price_data_ok:
        log.info(
            "decision: fresh stock price coverage %.3f is below min_fresh_price_coverage %.3f",
            fresh_stock_coverage,
            min_fresh_price_coverage,
        )
    mad_s = _as_num_series(df, ["mad"])
    stock_has_price_ma = stock_price.notna() & stock_ma200.notna()
    stock_above_ma200 = stock_price > stock_ma200
    stock_data_ok = stock_has_price_ma & ~stock_price_stale

    above_ma200_series = pd.Series(
        np.where(stock_data_ok, stock_above_ma200, pd.NA),
        index=df.index,
        dtype="boolean",
    )
    df["above_ma200"] = above_ma200_series
    df["stock_data_ok"] = stock_data_ok
    df["stock_price_age_days"] = stock_price_age_days
    df["stock_price_stale"] = stock_price_stale
    df["fresh_stock_price_coverage"] = fresh_stock_coverage
    df["global_price_data_ok"] = global_price_data_ok

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
    tech_data_ready = df["stock_data_ok"].astype(bool) & df["index_data_ok"].astype(bool) & bool(global_price_data_ok)

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

    quality_block = _build_quality_block(df, dec_cfg)
    for c in quality_block.columns:
        df[c] = quality_block[c]

    quality_strategy = str(df.get("quality_strategy", pd.Series("baseline", index=df.index)).iloc[0]).strip().lower()
    use_dividend_strategy = quality_strategy == "dividend_quality"
    use_graham_strategy = quality_strategy == "graham_strategy"
    dividend_min_score = int(dec_cfg.get("dividend_min_score", 0)) if use_dividend_strategy else 0
    graham_min_score = int(dec_cfg.get("graham_min_score", 0)) if use_graham_strategy else 0
    if use_dividend_strategy and "dividend_score" in df.columns:
        df["dividend_min_score_ok"] = pd.to_numeric(df["dividend_score"], errors="coerce").fillna(0.0) >= float(dividend_min_score)
    else:
        df["dividend_min_score_ok"] = True
    if use_graham_strategy and "graham_score" in df.columns:
        df["graham_min_score_ok"] = pd.to_numeric(df["graham_score"], errors="coerce").fillna(0.0) >= float(graham_min_score)
    else:
        df["graham_min_score_ok"] = True

    value_gate = _value_creation_gate(df, dec_cfg)
    quality_gate = _quality_gate(df, dec_cfg)
    for c in value_gate.columns:
        df[c] = value_gate[c]
    for c in quality_gate.columns:
        df[c] = quality_gate[c]
    value_qc_block, value_qc_specs, value_qc_summary = _build_value_qc_flags(df, dec_cfg)
    for c in value_qc_block.columns:
        df[c] = value_qc_block[c]
    _atomic_write_csv(ctx.run_dir / "value_qc_summary.csv", value_qc_summary)
    sufficiency_block, candidate_required_fields, candidate_min_count, candidate_min_ratio = _candidate_data_sufficiency(
        df, dec_cfg
    )
    for c in sufficiency_block.columns:
        df[c] = sufficiency_block[c]
    dq_flags, dq_audit = _run_data_quality_checks(df, dec_cfg=dec_cfg, asof=ctx.asof)
    for c in dq_flags.columns:
        df[c] = dq_flags[c]
    dq_cols = ["asof", "date", "ticker", "ins_id", "sector", "rule_id", "severity", "field", "value", "group_key", "group_n", "source_fields", "detail", "row_index"]
    dq_audit_out = dq_audit.reindex(columns=dq_cols) if not dq_audit.empty else pd.DataFrame(columns=dq_cols)
    _atomic_write_csv(ctx.run_dir / "data_quality_audit.csv", dq_audit_out)

    # DQ v2: regel-ID-baserte sjekker (supplement til eksisterende, ikke erstatning)
    try:
        from src.data_quality.rules import run_dq_rules
        dq_flags_v2, dq_audit_v2 = run_dq_rules(df)
        for c in dq_flags_v2.columns:
            df[c] = dq_flags_v2[c]
        _atomic_write_csv(ctx.run_dir / "data_quality_audit_v2.csv", dq_audit_v2)
        log.info(f"decision: DQ v2 — {int(dq_flags_v2['dq_fail_v2'].sum())} tickers med FAIL")
    except Exception as _dq_v2_err:
        log.warning(f"decision: DQ v2 feilet (ikke-kritisk): {_dq_v2_err}")
        _atomic_write_csv(ctx.run_dir / "data_quality_audit_v2.csv", pd.DataFrame())

    df["fundamental_ok"] = (
        df["mos"].notna() &
        (df["mos"] >= df["mos_req"]) &
        df["value_creation_ok"].fillna(False) &
        df["quality_gate_ok"].fillna(False) &
        df["dividend_min_score_ok"].fillna(True) &
        df["graham_min_score_ok"].fillna(True) &
        df["candidate_data_ok"].fillna(False) &
        ~df["data_quality_fail"].fillna(False)
    )
    df["technical_ok"] = df["tech_ok"]

    df["ma200_ok"] = df["stock_ma200_ok"].astype(bool)
    df["reason_fundamental_fail"] = ""
    mos_fail = ~(df["mos"].notna() & (df["mos"] >= df["mos_req"]))
    vc_fail = ~df["value_creation_ok"].fillna(False)
    q_fail = ~df["quality_gate_ok"].fillna(False)
    div_fail = ~df["dividend_min_score_ok"].fillna(True)
    graham_fail = ~df["graham_min_score_ok"].fillna(True)
    suff_fail = ~df["candidate_data_ok"].fillna(False)
    dq_fail = df["data_quality_fail"].fillna(False)

    df.loc[mos_fail, "reason_fundamental_fail"] = "mos_below_required"
    df.loc[vc_fail, "reason_fundamental_fail"] = df.loc[vc_fail, "reason_fundamental_fail"].map(
        lambda x: _join_reasons([x, "value_creation_fail"])
    )
    df.loc[q_fail, "reason_fundamental_fail"] = df.loc[q_fail, "reason_fundamental_fail"].map(
        lambda x: _join_reasons([x, "quality_gate_fail"])
    )
    if use_dividend_strategy:
        df.loc[div_fail, "reason_fundamental_fail"] = df.loc[div_fail, "reason_fundamental_fail"].map(
            lambda x: _join_reasons([x, f"dividend_score_below_{dividend_min_score}"])
        )
        if "dividend_reason" in df.columns:
            df.loc[div_fail, "reason_fundamental_fail"] = df.loc[div_fail].apply(
                lambda r: _join_reasons([r["reason_fundamental_fail"], str(r.get("dividend_reason", ""))]),
                axis=1,
            )
    if use_graham_strategy:
        df.loc[graham_fail, "reason_fundamental_fail"] = df.loc[graham_fail, "reason_fundamental_fail"].map(
            lambda x: _join_reasons([x, f"graham_score_below_{graham_min_score}"])
        )
        if "graham_reason" in df.columns:
            df.loc[graham_fail, "reason_fundamental_fail"] = df.loc[graham_fail].apply(
                lambda r: _join_reasons([r["reason_fundamental_fail"], str(r.get("graham_reason", ""))]),
                axis=1,
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
    if "candidate_data_missing_fields" in df.columns:
        df.loc[suff_fail, "reason_fundamental_fail"] = df.loc[suff_fail].apply(
            lambda r: _join_reasons(
                [
                    r["reason_fundamental_fail"],
                    REASON_DATA_NOT_SUFFICIENT,
                    f"missing_fields={str(r.get('candidate_data_missing_fields', ''))}",
                ]
            ),
            axis=1,
        )
    if "data_quality_fail_reasons" in df.columns:
        df.loc[dq_fail, "reason_fundamental_fail"] = df.loc[dq_fail].apply(
            lambda r: _join_reasons(
                [
                    r["reason_fundamental_fail"],
                    REASON_DATA_INVALID,
                    str(r.get("data_quality_fail_reasons", "")),
                ]
            ),
            axis=1,
        )

    df["reason_technical_fail"] = ""
    missing_stock_ma200 = stock_ma200.isna()
    missing_stock_tech = stock_price.isna() & ~missing_stock_ma200
    stale_stock_price = df["stock_price_stale"].astype(bool)
    stale_index_price = df.get("index_price_stale", pd.Series(False, index=df.index)).astype(bool)
    stock_below_ma200 = df["stock_data_ok"].astype(bool) & ~df["stock_ma200_ok"].astype(bool)
    bad_mad = df["stock_data_ok"].astype(bool) & mad_s.notna() & ~df["stock_mad_ok"].astype(bool)

    df.loc[missing_stock_ma200, "reason_technical_fail"] = df.loc[missing_stock_ma200, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, REASON_DATA_MISSING_MA200])
    )
    df.loc[missing_stock_tech, "reason_technical_fail"] = df.loc[missing_stock_tech, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, "missing_stock_technical_data"])
    )
    df.loc[stale_stock_price, "reason_technical_fail"] = df.loc[stale_stock_price, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, f"stale_price_data_gt_{max_price_age_days}d"])
    )
    df.loc[stale_index_price, "reason_technical_fail"] = df.loc[stale_index_price, "reason_technical_fail"].map(
        lambda x: _join_reasons([x, f"stale_benchmark_price_gt_{max_price_age_days}d"])
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
    if not global_price_data_ok:
        df["reason_technical_fail"] = df["reason_technical_fail"].map(
            lambda x: _join_reasons(
                [x, f"fresh_price_coverage_below_{min_fresh_price_coverage:.2f}"]
            )
        )
    df.loc[df["technical_ok"].astype(bool), "reason_technical_fail"] = ""
    df["decision_reasons"] = df.apply(
        lambda r: _join_reasons(
            [
                str(r.get("reason_fundamental_fail", "")),
                str(r.get("reason_technical_fail", "")),
                str(r.get("dq_fail_reasons", "")),
                str(r.get("dq_warn_reasons", "")),
            ]
        ),
        axis=1,
    )

    df["eligible"] = df["technical_ok"] & df["fundamental_ok"]
    df["value_qc_unresolved_alert_count"] = 0
    df["value_qc_unresolved_metrics"] = ""
    _write_ma200_sanity_report(ctx.run_dir, ctx.asof, df)

    # decision_reasons.json — maskinlesbar per-ticker beslutningslogg
    try:
        _dr_records = []
        for _, _r in df.iterrows():
            _dr_records.append({
                "ticker": str(_r.get("ticker", "")),
                "fundamental_ok": bool(_r.get("fundamental_ok", False)),
                "technical_ok": bool(_r.get("technical_ok", False)),
                "eligible": bool(_r.get("eligible", False)),
                "decision_reasons": str(_r.get("decision_reasons", "")),
                "dq_fail_rules": str(_r.get("dq_fail_rules", "")),
                "dq_warn_rules": str(_r.get("dq_warn_rules", "")),
            })
        _atomic_write_text(
            ctx.run_dir / "decision_reasons.json",
            json.dumps({"asof": ctx.asof, "run_id": ctx.run_id, "decisions": _dr_records}, indent=2, ensure_ascii=False),
        )
    except Exception as _dr_err:
        log.warning(f"decision: decision_reasons.json feilet (ikke-kritisk): {_dr_err}")

    strategy_screen_cols: list[str] = []
    if use_dividend_strategy:
        strategy_screen_cols.extend(
            [
                "quality_strategy",
                "dividend_score",
                "dividend_criteria_available_count",
                "dividend_criteria_missing_count",
                "dividend_min_score_ok",
                "dividend_reason",
            ]
        )
        for key, _ in DIVIDEND_CRITERIA:
            strategy_screen_cols.append(f"dividend_{key}_ok")
            strategy_screen_cols.append(f"dividend_{key}_available")
    if use_graham_strategy:
        strategy_screen_cols.extend(
            [
                "quality_strategy",
                "graham_score",
                "graham_criteria_available_count",
                "graham_criteria_missing_count",
                "graham_min_score_ok",
                "graham_reason",
            ]
        )
        for key, _ in GRAHAM_CRITERIA:
            strategy_screen_cols.append(f"graham_{key}_ok")
            strategy_screen_cols.append(f"graham_{key}_available")

    screen_cols = [c for c in [
        "ticker", "company", "market_cap", "intrinsic_value", "mos", "mos_req",
        "fundamental_ok", "technical_ok", "reason_fundamental_fail", "reason_technical_fail", "decision_reasons",
        "dq_blocked", "dq_fail_count", "dq_warn_count", "dq_fail_reasons", "dq_warn_reasons",
        "dq_blocked", "dq_fail_count", "dq_warn_count", "dq_fail_reasons", "dq_warn_reasons",
        "data_quality_fail", "data_quality_fail_count", "data_quality_warn_count", "data_quality_fail_reasons", "data_quality_warn_reasons",
        "value_creation_ok", "roic_wacc_spread", "roic_wacc_spread_y3", "value_creation_reason",
        "quality_gate_ok", "quality_weak_count", "quality_gate_reason",
        "value_qc_alert_count", "value_qc_has_alerts", "value_qc_alert_metrics",
        "candidate_data_ok", "candidate_available_fields_count", "candidate_required_fields_count", "candidate_data_coverage_ratio", "candidate_data_missing_fields",
        "date", "adj_close", "stock_price_age_days", "stock_price_stale", "fresh_stock_price_coverage", "global_price_data_ok",
        "stock_ma200_ok", "stock_mad_ok", "ma200_ok",
        "relevant_index_symbol", "relevant_index_key", "index_price_date", "index_price_age_days", "index_price_stale", "index_price", "index_ma200", "index_mad", "index_above_ma200",
        "index_data_ok", "index_ma200_ok", "index_mad_ok", "index_tech_ok",
        "mad", "ma21", "ma200", "above_ma200", "quality_score"
    ] + strategy_screen_cols if c in df.columns]
    _atomic_write_csv(ctx.run_dir / "screen_basic.csv", df[screen_cols])

    dq_lines = [f"# Data Quality Report ({ctx.asof})", ""]
    by_ticker_out = pd.DataFrame(columns=["ticker", "dq_fail_count", "dq_warn_count", "top_fail_reasons", "top_warn_reasons"])
    if dq_audit.empty:
        dq_lines.append("Ingen datakvalitetsbrudd registrert.")
    else:
        by_ticker = dq_audit.groupby(["ticker", "severity"]).size().unstack(fill_value=0).reset_index()
        for sev in ["FAIL", "WARN"]:
            if sev not in by_ticker.columns:
                by_ticker[sev] = 0
        by_ticker = by_ticker.sort_values(["FAIL", "WARN"], ascending=[False, False])
        fail_reason = dq_audit[dq_audit["severity"] == "FAIL"].groupby("ticker")["rule_id"].apply(lambda x: ";".join(x.value_counts().head(3).index.tolist()))
        warn_reason = dq_audit[dq_audit["severity"] == "WARN"].groupby("ticker")["rule_id"].apply(lambda x: ";".join(x.value_counts().head(3).index.tolist()))
        by_ticker_out = by_ticker.rename(columns={"FAIL": "dq_fail_count", "WARN": "dq_warn_count"})
        by_ticker_out["top_fail_reasons"] = by_ticker_out["ticker"].map(fail_reason).fillna("")
        by_ticker_out["top_warn_reasons"] = by_ticker_out["ticker"].map(warn_reason).fillna("")

        blocked = df[df["dq_blocked"].fillna(False)]["ticker"].astype(str).tolist()
        warned = df[df["dq_warn_count"].fillna(0).astype(int) > 0]["ticker"].astype(str).tolist()
        top20 = dq_audit.groupby(["rule_id", "severity", "field"]).size().reset_index(name="count").sort_values("count", ascending=False).head(20)
        dq_lines.extend([
            "## FAIL/WARN per ticker",
            _md_table(by_ticker_out.head(200), max_rows=200),
            "",
            "## Topp 20 regelbrudd",
            _md_table(top20, max_rows=20),
            "",
            "## Blokkerte tickere",
            ", ".join(blocked) if blocked else "_(ingen)_",
            "",
            "## Tickers med WARN",
            ", ".join(warned) if warned else "_(ingen)_",
        ])
    _atomic_write_csv(ctx.run_dir / "data_quality_by_ticker.csv", by_ticker_out)
    _atomic_write_text(ctx.run_dir / "data_quality_report.md", "\n".join(dq_lines))

    eligible = df[df["eligible"]].copy()
    if use_dividend_strategy and "dividend_score" in eligible.columns:
        eligible = eligible.sort_values(
            by=["dividend_score", "intrinsic_value", "market_cap"],
            ascending=[False, False, False],
            na_position="last",
        )
    elif use_graham_strategy and "graham_score" in eligible.columns:
        eligible = eligible.sort_values(
            by=["graham_score", "intrinsic_value", "market_cap"],
            ascending=[False, False, False],
            na_position="last",
        )
    else:
        eligible = eligible.sort_values(by=["quality_score", "mos", "market_cap"], ascending=[False, False, False], na_position="last")

    out_cols = [c for c in [
        "ticker", "company",
        "market_cap", "intrinsic_value", "mos", "mos_req", "mos_basis",
        "quality_strategy", "quality_score", "dividend_score", "dividend_criteria_available_count", "dividend_criteria_missing_count",
        "dividend_min_score_ok", "dividend_reason",
        "graham_score", "graham_criteria_available_count", "graham_criteria_missing_count",
        "graham_min_score_ok", "graham_reason",
        "beta", "coe_used", "wacc_used",
        "value_creation_ok", "roic_wacc_spread", "roic_wacc_spread_y1", "roic_wacc_spread_y2", "roic_wacc_spread_y3", "value_creation_reason",
        "quality_gate_ok", "quality_weak_count", "quality_gate_reason",
        "value_qc_alert_count", "value_qc_has_alerts", "value_qc_alert_metrics", "value_qc_unresolved_alert_count", "value_qc_unresolved_metrics",
        "candidate_data_ok", "candidate_available_fields_count", "candidate_required_fields_count", "candidate_data_coverage_ratio", "candidate_data_missing_fields",
        "date", "adj_close", "stock_price_age_days", "stock_price_stale", "fresh_stock_price_coverage", "global_price_data_ok",
        "above_ma200", "mad", "ma21", "ma200", "stock_ma200_ok", "stock_mad_ok",
        "relevant_index_symbol", "relevant_index_key", "index_price_date", "index_price_age_days", "index_price_stale", "index_price", "index_ma200", "index_mad", "index_above_ma200",
        "index_data_ok", "ma200_ok", "index_ma200_ok", "index_mad_ok", "index_tech_ok", "high_risk_flag",
        "fundamental_ok", "technical_ok", "reason_fundamental_fail", "reason_technical_fail", "decision_reasons",
        "dq_blocked", "dq_fail_count", "dq_warn_count", "dq_fail_reasons", "dq_warn_reasons",
        "data_quality_fail", "data_quality_fail_count", "data_quality_warn_count", "data_quality_fail_reasons", "data_quality_warn_reasons",
        "model", "reason",
    ] if c in df.columns]

    out_csv = ctx.run_dir / "decision.csv"
    out_md = ctx.run_dir / "decision.md"

    if eligible.empty:
        if use_dividend_strategy and "dividend_score" in df.columns:
            diag = df.sort_values(by=["dividend_score", "mos"], ascending=[False, False], na_position="last")
        elif use_graham_strategy and "graham_score" in df.columns:
            diag = df.sort_values(by=["graham_score", "mos"], ascending=[False, False], na_position="last")
        else:
            diag = df.sort_values(by=["quality_score", "mos"], ascending=[False, False], na_position="last")
        _atomic_write_csv(out_csv, diag[out_cols].head(top_n))
        _write_top_candidates_report(ctx.run_dir, ctx.asof, diag[out_cols], max_rows=top_n)
        _atomic_write_csv(ctx.run_dir / "shortlist.csv", pd.DataFrame(columns=out_cols))
        md = [
            f"# Decision ({ctx.asof})",
            "",
            "**Anbefaling:** CASH (ingen kandidater bestod filter).",
            "",
            "## 1) Fundamental analyse",
            f"- MoS-regel: min {mos_min:.0%} (hoy risiko {mos_high:.0%})",
            "- Verdiskaping: ROIC > WACC i konservativ 3-ars bane",
            "- Kvalitetsgate: >=2 svake indikatorer forkaster kandidat",
            f"- Datatilstrekkelighet: minst {candidate_min_count}/{len(candidate_required_fields)} felt og >= {candidate_min_ratio:.0%} dekning",
            "",
            "## 2) Aksjeanalyse",
            f"- Teknisk regel: {tech_rule_text}",
            f"- Maks alder aksjekurs: {max_price_age_days} dager",
            f"- Fersk prisdekning i universet: {fresh_stock_coverage:.1%} (krav {min_fresh_price_coverage:.0%})",
            "",
            ("## Klar beskjed" if (not global_price_data_ok) else "## Klar beskjed"),
            (
                f"- CASH fordi oppdaterte kurser er utilstrekkelige: fersk dekning {fresh_stock_coverage:.1%} < krav {min_fresh_price_coverage:.0%}."
                if not global_price_data_ok
                else "- Ingen kandidat passerte alle krav (inkludert datatilstrekkelighet)."
            ),
            "",
            "## 3) Produkter og markedsforventning",
            "- Ikke relevant for CASH i denne kjoringen (ingen valgt kandidat).",
            "",
            "## 4) Nyheter og vurdering",
            "- Ikke vurdert i detalj fordi ingen kandidat ble valgt.",
            "",
            "## Videre lesing",
            "- Top candidates (egen side): [top_candidates.md](top_candidates.md)",
            "- Full eksport: `decision.csv`",
        ]
        _atomic_write_csv(ctx.run_dir / "decision_schema.csv", pd.DataFrame(columns=["parameter", "value", "threshold", "status", "comment"]))
        _atomic_write_csv(
            ctx.run_dir / "candidate_value_qc.csv",
            pd.DataFrame(columns=["metric", "value", "lower", "upper", "is_alert", "resolved", "source_unique_values", "source_snapshot", "formula_check", "note"]),
        )
        _atomic_write_text(
            ctx.run_dir / "media_red_flags.json",
            json.dumps(
                {
                    "enabled": False,
                    "status": "not_run_no_candidate",
                    "headlines_checked": 0,
                    "red_flag_count": 0,
                    "red_flags": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _atomic_write_csv(ctx.run_dir / "candidate_products.csv", _empty_candidate_products())
        _atomic_write_csv(ctx.run_dir / "candidate_product_demand_forecast.csv", _empty_candidate_product_demand_forecast())
        _atomic_write_csv(ctx.run_dir / "candidate_product_demand_summary.csv", _empty_candidate_product_demand_summary())
        _atomic_write_csv(ctx.run_dir / "candidate_market_position.csv", _empty_candidate_market_position())
        _atomic_write_text(out_md, "\n".join(md))
        log.info(f"decision: wrote {out_csv}")
        log.info(f"decision: wrote {out_md}")
        return 0

    pick = eligible.iloc[0].copy()
    candidate_qc = _analyze_candidate_value_qc(pick, value_qc_specs)
    _atomic_write_csv(ctx.run_dir / "candidate_value_qc.csv", candidate_qc)
    pick_alerts = int(candidate_qc["is_alert"].astype(bool).sum()) if not candidate_qc.empty else 0
    pick_unresolved = int((candidate_qc["is_alert"].astype(bool) & ~candidate_qc["resolved"].astype(bool)).sum()) if not candidate_qc.empty else 0
    pick_unresolved_metrics = ", ".join(
        candidate_qc.loc[
            candidate_qc["is_alert"].astype(bool) & ~candidate_qc["resolved"].astype(bool),
            "metric",
        ].astype(str).tolist()
    )
    pick["value_qc_alert_count"] = pick_alerts
    pick["value_qc_unresolved_alert_count"] = pick_unresolved
    pick["value_qc_unresolved_metrics"] = pick_unresolved_metrics

    if pick.name in df.index:
        df.loc[pick.name, "value_qc_alert_count"] = pick_alerts
        df.loc[pick.name, "value_qc_unresolved_alert_count"] = pick_unresolved
        df.loc[pick.name, "value_qc_unresolved_metrics"] = pick_unresolved_metrics
    if pick.name in eligible.index:
        eligible.loc[pick.name, "value_qc_alert_count"] = pick_alerts
        eligible.loc[pick.name, "value_qc_unresolved_alert_count"] = pick_unresolved
        eligible.loc[pick.name, "value_qc_unresolved_metrics"] = pick_unresolved_metrics

    _atomic_write_csv(out_csv, eligible[out_cols].head(top_n))
    _write_top_candidates_report(ctx.run_dir, ctx.asof, eligible[out_cols], max_rows=top_n)
    _atomic_write_csv(ctx.run_dir / "shortlist.csv", eligible[out_cols].head(top_n))
    candidate_products = _extract_candidate_products(pick)
    market_inputs = _estimate_market_forecast_from_prices(prices_df=prices_df, pick=pick, asof=ctx.asof, lookback_years=5)
    candidate_market_position = _build_candidate_market_position(pick=pick, universe_df=df)
    candidate_demand_forecast, candidate_demand_summary = _build_candidate_product_demand_forecast(
        pick=pick,
        universe_df=df,
        asof=ctx.asof,
        products=candidate_products,
        market_inputs=market_inputs,
        horizon_years=3,
    )
    _atomic_write_csv(ctx.run_dir / "candidate_products.csv", candidate_products)
    _atomic_write_csv(ctx.run_dir / "candidate_product_demand_forecast.csv", candidate_demand_forecast)
    _atomic_write_csv(ctx.run_dir / "candidate_product_demand_summary.csv", candidate_demand_summary)
    _atomic_write_csv(ctx.run_dir / "candidate_market_position.csv", candidate_market_position)

    annual_status, annual_detail = _annual_source_status(pick)
    fundamental_rows = [
        {"Nokkel": "Market cap", "Verdi": _fmt_num(pick.get("market_cap"), 4), "Kommentar": "Fra master_valued"},
        {"Nokkel": "Intrinsic value", "Verdi": _fmt_num(pick.get("intrinsic_value"), 4), "Kommentar": "Fra valuation (fundamental modell)"},
        {"Nokkel": "MoS", "Verdi": _fmt_pct(pick.get("mos"), 1), "Kommentar": f"Krav { _fmt_pct(pick.get('mos_req'), 0) }"},
        {"Nokkel": "WACC used", "Verdi": _fmt_pct(pick.get("wacc_used"), 2), "Kommentar": "Kapitalkostnad"},
        {"Nokkel": "COE used", "Verdi": _fmt_pct(pick.get("coe_used"), 2), "Kommentar": "Egenkapitalkostnad"},
        {"Nokkel": "ROIC-WACC spread", "Verdi": _fmt_pct(pick.get("roic_wacc_spread"), 2), "Kommentar": "Verdiskaping > 0 er positivt"},
        {"Nokkel": "Quality score", "Verdi": _fmt_num(pick.get("quality_score"), 4), "Kommentar": "Tverrsnittsmodell"},
        {"Nokkel": "Value QA unresolved", "Verdi": str(int(pick_unresolved)), "Kommentar": "Uforklarte avvik som bor sjekkes manuelt"},
        {"Nokkel": "Arsregnskap brukt", "Verdi": annual_status, "Kommentar": annual_detail},
    ]
    fundamental_df = pd.DataFrame(fundamental_rows)

    px_date = pd.to_datetime(pick.get("date"), errors="coerce")
    px_date_txt = str(px_date.date()) if pd.notna(px_date) else "ukjent"
    price_age = pd.to_numeric(pd.Series([pick.get("stock_price_age_days")]), errors="coerce").fillna(np.nan).iloc[0]
    stock_rows = [
        {"Nokkel": "Siste tilgjengelige pris", "Verdi": _fmt_num(pick.get("adj_close"), 5), "Kommentar": f"Dato {px_date_txt}"},
        {"Nokkel": "Prisalder", "Verdi": (f"{int(price_age)} dager" if np.isfinite(price_age) else "ukjent"), "Kommentar": f"Krav <= {max_price_age_days} dager"},
        {"Nokkel": "Above MA200", "Verdi": str(bool(_to_bool(pick.get("above_ma200")))), "Kommentar": f"MA200={_fmt_num(pick.get('ma200'), 4)}"},
        {"Nokkel": "MAD", "Verdi": _fmt_num(pick.get("mad"), 4), "Kommentar": f"Terskel {mad_min:.3f}"},
        {"Nokkel": "Relevant index", "Verdi": str(pick.get("relevant_index_symbol", "")), "Kommentar": f"Index > MA200: {bool(_to_bool(pick.get('index_ma200_ok')))}"},
        {"Nokkel": "Teknisk gate", "Verdi": str(bool(_to_bool(pick.get("technical_ok")))), "Kommentar": tech_rule_text},
    ]
    stock_df = pd.DataFrame(stock_rows)
    stock_spark = _price_sparkline_from_prices(prices_df, pick, ctx.asof)

    product_display_cols = [c for c in ["product_label", "source_field", "value_share", "detail"] if c in candidate_products.columns]
    product_display = candidate_products[product_display_cols].copy() if product_display_cols else candidate_products.copy()
    if "value_share" in product_display.columns:
        product_display["value_share"] = product_display["value_share"].map(lambda v: _fmt_pct(v, 1) if np.isfinite(_to_float(v)) else "n/a")
    demand_base = _base_demand_series(candidate_demand_forecast)
    demand_lines = _render_ascii_bar_chart(demand_base)

    md = []
    md.append(f"# Decision ({ctx.asof})")
    md.append("")
    md.append(f"**Anbefaling:** Kandidat = `{pick['ticker']}` (beste blant de som bestod filter).")
    md.append("")
    md.append("## Oversikt")
    md.append("- Top candidates (egen side): [top_candidates.md](top_candidates.md)")
    md.append("- Full eksport: `decision.csv`")
    md.append("- Grafisk rapport: `decision_report.html`")
    md.append("")
    md.append("## 1) Fundamental analyse")
    md.append(_md_table(fundamental_df, max_rows=40))
    if np.isfinite(_to_float(pick.get("candidate_data_coverage_ratio"))):
        md.append(
            f"- Kommentar: Datadekning {int(pick.get('candidate_available_fields_count', 0))}/{int(pick.get('candidate_required_fields_count', len(candidate_required_fields)))} "
            f"({float(pick.get('candidate_data_coverage_ratio')):.0%})."
        )
    md.append(
        "- Kommentar: Verdsettelse er fundamental (kontantstrom, netto gjeld, WACC/COE). Pris/teknisk brukes kun i timing/gating."
    )
    md.append("")
    md.append("## 2) Aksje analyse")
    md.append(_md_table(stock_df, max_rows=30))
    md.append(f"- Prisgraf (ASCII, siste observasjoner): `{stock_spark}`")
    md.append(
        f"- Kommentar: Siste tilgjengelige kurs brukes som 'dagens pris' i denne rapporten (per {px_date_txt})."
    )
    md.append("")
    md.append("## Beslutningskommentar")
    md.append(f"- MoS { _fmt_pct(pick.get('mos'), 1) } mot krav { _fmt_pct(pick.get('mos_req'), 0) }.")
    md.append(f"- Teknisk regel: {tech_rule_text}")
    md.append("- Break-case: Hvis aksjen faller under MA200 eller MoS under krav ved neste kjoring => CASH.")
    md.append("")
    md.append("## Beslutningsskjema")
    schema_df = _decision_schema_rows(
        pick=pick,
        mos_min=mos_min,
        mos_high=mos_high,
        mad_min=mad_min,
        max_price_age_days=max_price_age_days,
        required_fields=candidate_required_fields,
        min_count=candidate_min_count,
        min_ratio=candidate_min_ratio,
    )
    _atomic_write_csv(ctx.run_dir / "decision_schema.csv", schema_df)
    md.append(_md_table(schema_df, max_rows=50))
    md.append("")
    md.append("## Kvalitetssikring Av Verdier")
    md.append(f"- Antall avvik i kandidattall: {pick_alerts}")
    md.append(f"- Uforklarte avvik etter kildeanalyse: {pick_unresolved}")
    if pick_unresolved > 0:
        md.append(
            f"- VIKTIG: Uvanlige verdier uten robust forklaring for metrikker: {pick_unresolved_metrics}. "
            "Disse bør manuelt verifiseres før endelig kjøpsbeslutning."
        )
    else:
        md.append("- Ingen uforklarte avvik i nøkkelverdiene for valgt kandidat.")
    md.append("")
    media_scan = _run_media_red_flag_scan(
        asof=ctx.asof,
        ticker=str(pick.get("ticker", "")),
        company=str(pick.get("company", "")),
        dec_cfg=dec_cfg,
    )
    _atomic_write_text(ctx.run_dir / "media_red_flags.json", json.dumps(media_scan, ensure_ascii=False, indent=2))
    media_risk, media_comment = _media_assessment_from_scan(media_scan)
    media_df = _media_headlines_frame(media_scan, max_rows=10)
    media_notes = [
        f"Status: {media_scan.get('status', 'unknown')}",
        f"Headline-sjekk: {int(media_scan.get('headlines_checked', 0))}",
        f"Potensielle red flags: {int(media_scan.get('red_flag_count', 0))}",
        f"Vurdering: {media_risk}",
        f"Kommentar: {media_comment}",
    ]

    md.extend(
        _render_candidate_product_demand_md(
            products=candidate_products,
            demand_forecast=candidate_demand_forecast,
            demand_summary=candidate_demand_summary,
            market_position=candidate_market_position,
        )
    )
    md.append("## 4) Nyheter og vurdering")
    md.extend([f"- {line}" for line in media_notes])
    if media_scan.get("status") == "error":
        md.append(f"- Feil under mediesjekk: {media_scan.get('error', '')}")
    md.append("")
    md.append(_md_table(media_df, max_rows=10))
    md.append("")

    report_html = _build_decision_report_html(
        asof=ctx.asof,
        pick=pick,
        fundamental_df=fundamental_df,
        stock_df=stock_df,
        products_df=product_display,
        demand_df=demand_base,
        demand_chart_lines=demand_lines,
        media_df=media_df,
        media_notes=media_notes,
        top_candidates_rel="top_candidates.md",
    )
    _atomic_write_text(ctx.run_dir / "decision_report.html", report_html)
    _atomic_write_text(out_md, "\n".join(md))
    log.info(f"decision: wrote {out_csv}")
    log.info(f"decision: wrote {out_md}")
    return 0
