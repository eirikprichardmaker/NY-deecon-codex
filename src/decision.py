from __future__ import annotations

import json
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
        return {"enabled": False, "status": "disabled", "headlines_checked": 0, "red_flag_count": 0, "red_flags": []}

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
    }


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
            f"- Quality-strategi: {quality_strategy}",
            (f"- Dividend minimum score: {dividend_min_score}/11" if use_dividend_strategy else "- Dividend minimum score: ikke aktiv"),
            (f"- Graham minimum score: {graham_min_score}/7" if use_graham_strategy else "- Graham minimum score: ikke aktiv"),
            f"- {tech_rule_text}",
            f"- Aksjekurs må være maks {max_price_age_days} dager gammel relativt til asof.",
            f"- Min andel ferske aksjekurser i universet: {min_fresh_price_coverage:.0%} (nå: {fresh_stock_coverage:.1%})",
            f"- Datatilstrekkelighet for kandidat: minst {candidate_min_count}/{len(candidate_required_fields)} felt og >= {candidate_min_ratio:.0%} dekning",
            "- Verdsetting: bygger primært på fundamentale størrelser (FCF/OCF/CAPEX, netto gjeld, WACC/COE), ikke pris/teknisk.",
            "",
            ("## Klar beskjed" if (not global_price_data_ok) else "## Klar beskjed"),
            (
                f"- CASH fordi oppdaterte kurser er utilstrekkelige: fersk dekning {fresh_stock_coverage:.1%} < krav {min_fresh_price_coverage:.0%}."
                if not global_price_data_ok
                else "- Ingen kandidat passerte alle krav (inkludert datatilstrekkelighet)."
            ),
            "",
            "## Topp (diagnostikk – før filter)",
            _md_table(diag[out_cols], max_rows=10),
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
    md.append(f"- Quality strategy: {quality_strategy}")
    if np.isfinite(pick.get("beta", np.nan)): md.append(f"- Beta: {float(pick['beta']):.2f}")
    if np.isfinite(pick.get("quality_score", np.nan)): md.append(f"- Quality score: {float(pick['quality_score']):.3f}")
    if use_dividend_strategy and np.isfinite(pick.get("dividend_score", np.nan)):
        md.append(f"- Dividend quality score: {int(pick['dividend_score'])}/11")
        md.append(f"- Dividend criteria available: {int(pick.get('dividend_criteria_available_count', 0))}/11")
    if use_graham_strategy and np.isfinite(pick.get("graham_score", np.nan)):
        md.append(f"- Graham score: {int(pick['graham_score'])}/7")
        md.append(f"- Graham criteria available: {int(pick.get('graham_criteria_available_count', 0))}/7")
    if np.isfinite(pick.get("fresh_stock_price_coverage", np.nan)):
        md.append(
            f"- Fersk prisdekning i universet: {float(pick.get('fresh_stock_price_coverage')):.1%} (krav >= {min_fresh_price_coverage:.0%})"
        )
    if np.isfinite(pick.get("candidate_data_coverage_ratio", np.nan)):
        md.append(
            f"- Datadekning for kandidat: {int(pick.get('candidate_available_fields_count', 0))}/{int(pick.get('candidate_required_fields_count', len(candidate_required_fields)))} "
            f"({float(pick.get('candidate_data_coverage_ratio')):.0%})"
        )
    if np.isfinite(pick.get("value_qc_alert_count", np.nan)):
        md.append(
            f"- Verdi-QA avvik: {int(pick.get('value_qc_alert_count', 0))} (uløste: {int(pick.get('value_qc_unresolved_alert_count', 0))})"
        )
    md.append("")
    md.append("## Årsaker (5-10 punkter)")
    md.append(f"- Krav MoS >= {mos_min:.0%} (høy risiko: {mos_high:.0%})")
    md.append("- Verdiskaping: ROIC-WACC må være positiv i 3-års konservativ bane")
    md.append("- Kvalitetsgate: minst 2 svekkede indikatorer forkaster kandidat")
    if use_dividend_strategy:
        md.append("- Dividend strategy aktiv: 11 kriterier for utbyttekvalitet, lønnsomhet og verdsettelse")
        md.append(f"- Minst {dividend_min_score} av 11 kriterier kreves")
        md.append("- Kriterier: Dividend History, Dividend Growth, Payout ratio, Increase of the number of shares, Market value, Free cash flow margin, Operating cash flow margin, Profit Margin, Return on assets, Yield, P/E")
    if use_graham_strategy:
        md.append("- Graham strategy aktiv: defensive kriterier fra The Intelligent Investor")
        md.append(f"- Minst {graham_min_score} av 7 kriterier kreves")
        md.append("- Kriterier: Size, Financial strength, Earning stability, Dividend History, Profit Growth, Moderate P/E ratio, Moderate Equity Price")
    md.append("- Verdsettingen er fundamentaldrevet: DCF basert på kontantstrøm, netto gjeld og kapitalkostnad. Pris/teknisk brukes ikke i intrinsic-verdi.")
    md.append(f"- {tech_rule_text}")
    md.append(f"- Aksjekurs må være maks {max_price_age_days} dager gammel relativt til asof.")
    md.append(f"- Valgt ticker har MoS {float(pick['mos']):.1%} og quality_score {float(pick.get('quality_score', 0.0)):.3f}")
    if np.isfinite(pick.get("roic_wacc_spread", np.nan)):
        md.append(f"- ROIC-WACC spread (normalisert): {float(pick.get('roic_wacc_spread')):.3%}")
    if np.isfinite(pick.get("quality_weak_count", np.nan)):
        md.append(f"- Svekkede kvalitetsindikatorer: {int(pick.get('quality_weak_count'))}")
    if "ma200" in pick.index and np.isfinite(pick.get("ma200", np.nan)):
        md.append(f"- Teknisk: pris over MA200={bool(pick.get('above_ma200', False))}, MA200={float(pick.get('ma200')):.3g}")
    if "adj_close" in pick.index and np.isfinite(pick.get("adj_close", np.nan)):
        pick_px_date = pd.to_datetime(pick.get("date"), errors="coerce")
        pick_px_date_txt = str(pick_px_date.date()) if pd.notna(pick_px_date) else "ukjent"
        if np.isfinite(pick.get("stock_price_age_days", np.nan)):
            pick_age_txt = f"{int(pick.get('stock_price_age_days'))} dager"
        else:
            pick_age_txt = "ukjent alder"
        md.append(f"- Siste aksjekurs (adj_close): {float(pick.get('adj_close')):.4g} (dato {pick_px_date_txt}, {pick_age_txt})")
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
    md.append("## Media Red Flag")
    md.append(f"- Status: {media_scan.get('status', 'unknown')}")
    md.append(f"- Headline-sjekk: {int(media_scan.get('headlines_checked', 0))}")
    md.append(f"- Potensielle red flags: {int(media_scan.get('red_flag_count', 0))}")
    if media_scan.get("status") == "error":
        md.append(f"- Feil under mediesjekk: {media_scan.get('error', '')}")
    else:
        red_flags = media_scan.get("red_flags", []) or []
        if red_flags:
            md.append("- Treff:")
            for item in red_flags[:5]:
                md.append(
                    f"  - {item.get('title', '')} | terms={item.get('matched_terms', '')} | source={item.get('source', '')}"
                )
        else:
            md.append("- Ingen tydelige red-flag nøkkelord i sjekkede overskrifter.")
    md.append("")
    md.append("## Topp 10 (eligible)")
    md.append(_md_table(eligible[out_cols], max_rows=10))

    _atomic_write_text(out_md, "\n".join(md))
    log.info(f"decision: wrote {out_csv}")
    log.info(f"decision: wrote {out_md}")
    return 0
