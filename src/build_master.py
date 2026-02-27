from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.config import resolve_paths
from src.common.io import read_parquet


MARKET_BY_SUFFIX = {
    "OL": "NO",
    "ST": "SE",
    "CO": "DK",
    "HE": "FI",
}

MARKET_BY_COUNTRY = {
    "NO": "NO",
    "NORWAY": "NO",
    "SE": "SE",
    "SWEDEN": "SE",
    "DK": "DK",
    "DENMARK": "DK",
    "FI": "FI",
    "FINLAND": "FI",
}

DIVIDEND_FEATURE_COLS = [
    "dividend_history_years",
    "dividend_paid_years",
    "dividend_growth_5y",
    "dividend_cagr_5y",
    "profit_growth_5y",
    "payout_ratio",
    "share_count_growth_5y",
    "fcf_margin",
    "ocf_margin",
    "profit_margin",
    "roa",
    "dividend_yield",
]

DEFAULT_FINANCIALS_CORE_FIELDS = [
    "revenue_total",
    "cost_of_goods_sold",
    "gross_profit",
    "sga_expense",
    "depreciation_expense",
    "amortization_expense",
    "operating_income_ebit",
    "net_finance_income_expense",
    "profit_before_tax",
    "income_tax_expense",
    "net_income",
    "net_income_attributable_to_parent",
    "cash_and_cash_equivalents",
    "accounts_receivable",
    "inventory",
    "property_plant_equipment",
    "right_of_use_assets",
    "intangible_assets",
    "goodwill",
    "deferred_tax_assets",
    "total_assets",
    "accounts_payable",
    "short_term_debt",
    "long_term_debt",
    "lease_liabilities_current",
    "lease_liabilities_noncurrent",
    "total_liabilities",
    "equity_attributable_to_parent",
    "total_equity",
    "cash_flow_from_operations",
    "interest_paid_cash",
    "income_taxes_paid_cash",
    "capex_purchase_ppe",
    "cash_flow_from_investing",
    "cash_flow_from_financing",
    "dividends_paid_total",
    "net_change_in_cash",
    "fx_effect_on_cash",
    "shares_outstanding_basic",
    "shares_outstanding_diluted",
]

REPORTS_Y_TO_CANONICAL = {
    "revenues": "revenue_total",
    "gross_income": "gross_profit",
    "operating_income": "operating_income_ebit",
    "profit_before_tax": "profit_before_tax",
    "profit_to_equity_holders": "net_income_attributable_to_parent",
    "cash_and_equivalents": "cash_and_cash_equivalents",
    "intangible_assets": "intangible_assets",
    "tangible_assets": "property_plant_equipment",
    "total_assets": "total_assets",
    "total_equity": "total_equity",
    "cash_flow_from_operating_activities": "cash_flow_from_operations",
    "cash_flow_from_investing_activities": "cash_flow_from_investing",
    "cash_flow_from_financing_activities": "cash_flow_from_financing",
    "cash_flow_for_the_year": "net_change_in_cash",
    "number_of_shares": "shares_outstanding_basic",
}

DEFAULT_REPORTS_Y_GUARD_FIELDS = [
    "revenue_total",
    "gross_profit",
    "operating_income_ebit",
    "profit_before_tax",
    "net_income_attributable_to_parent",
    "cash_and_cash_equivalents",
    "property_plant_equipment",
    "intangible_assets",
    "total_assets",
    "total_equity",
    "total_liabilities",
    "cash_flow_from_operations",
    "cash_flow_from_investing",
    "cash_flow_from_financing",
    "net_change_in_cash",
]

FINANCIALS_GUARD_COLS = [
    "financials_guard_anomaly",
    "reason_financials_guard",
    "financials_guard_changed_fields",
    "financials_guard_fields",
    "financials_guard_prev_snapshot",
    "financials_guard_current_snapshot",
]


def _quality_write(run_dir, lines: list[str]) -> None:
    path = run_dir / "quality.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    s = s.lstrip("^")
    if "." in s:
        s = s.split(".", 1)[0]
    return s.strip()


def _suffix_from_symbol(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    if "." not in s:
        return ""
    return s.rsplit(".", 1)[-1].strip()


def _ensure_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Ensure SMA-based MA columns exist and recompute derived technical flags."""
    prices = prices.copy()
    need = ["ma21", "ma200"]
    missing_any = any(c not in prices.columns for c in need)
    all_nan = any((c in prices.columns and pd.to_numeric(prices[c], errors="coerce").notna().mean() == 0.0) for c in need)

    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    if missing_any or all_nan:
        prices = prices.sort_values(["ticker_norm", "date"])
        g = prices.groupby("ticker_norm", group_keys=False)
        prices["ma21"] = g["adj_close"].transform(lambda s: s.rolling(21, min_periods=21).mean())
        prices["ma200"] = g["adj_close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
    else:
        prices["ma21"] = pd.to_numeric(prices["ma21"], errors="coerce")
        prices["ma200"] = pd.to_numeric(prices["ma200"], errors="coerce")

    prices["mad"] = (prices["ma21"] - prices["ma200"]) / prices["ma200"]
    prices["above_ma200"] = prices["adj_close"] > prices["ma200"]
    return prices


def _load_ticker_mapping(ctx) -> pd.DataFrame:
    candidates = [
        ctx.project_root / "config" / "tickers.csv",
        ctx.project_root / "config" / "tickers_with_insid_clean.csv",
        ctx.project_root / "config" / "tickers_with_insid.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = {str(c).strip().lower(): c for c in df.columns}
            t_col = cols.get("ticker")
            y_col = cols.get("yahoo_ticker") or cols.get("yahoo")
            if t_col and y_col:
                out = df[[t_col, y_col]].rename(columns={t_col: "ticker", y_col: "yahoo_ticker"}).copy()
                out["ticker_norm"] = out["ticker"].map(_norm_ticker)
                out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
                out = out[(out["ticker_norm"] != "") & (out["yahoo_ticker"].replace("", np.nan).notna())]
                return out[["ticker_norm", "yahoo_ticker"]].drop_duplicates("ticker_norm", keep="first")
    return pd.DataFrame(columns=["ticker_norm", "yahoo_ticker"])


def _load_insid_mapping(ctx) -> pd.DataFrame:
    candidates = [
        ctx.project_root / "config" / "tickers_with_insid_clean.csv",
        ctx.project_root / "config" / "tickers_with_insid.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = {str(c).strip().lower(): c for c in df.columns}
        y_col = cols.get("yahoo_ticker")
        i_col = cols.get("ins_id") or cols.get("insid")
        if not y_col or not i_col:
            continue

        t_col = cols.get("ticker") or cols.get("ticker_x") or cols.get("ticker_y")
        c_col = cols.get("country")
        keep = [y_col, i_col]
        if t_col:
            keep.append(t_col)
        if c_col:
            keep.append(c_col)

        out = df[keep].copy().rename(columns={y_col: "yahoo_ticker", i_col: "ins_id"})
        if t_col:
            out = out.rename(columns={t_col: "ticker"})
        else:
            out["ticker"] = ""
        if c_col:
            out = out.rename(columns={c_col: "country"})
        else:
            out["country"] = ""

        out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
        out["ticker_norm"] = out["ticker"].map(_norm_ticker)
        out["ins_id"] = pd.to_numeric(out["ins_id"], errors="coerce")
        out = out[out["ins_id"].notna()].copy()
        out["ins_id"] = out["ins_id"].astype(int)
        out["market"] = out["country"].map(lambda x: MARKET_BY_COUNTRY.get(str(x).strip().upper(), ""))
        out["yahoo_key"] = out["yahoo_ticker"].str.upper()
        out = out[(out["yahoo_ticker"] != "") | (out["ticker_norm"] != "")]
        return out[["yahoo_ticker", "yahoo_key", "ticker_norm", "ins_id", "market"]].drop_duplicates(
            subset=["yahoo_key", "ticker_norm"], keep="first"
        )

    return pd.DataFrame(columns=["yahoo_ticker", "yahoo_key", "ticker_norm", "ins_id", "market"])


def _normalize_prices_with_mapping(prices: pd.DataFrame, mapping: pd.DataFrame, log) -> pd.DataFrame:
    prices = prices.copy()
    if "ticker" in prices.columns:
        prices["ticker_norm"] = prices["ticker"].map(_norm_ticker)
    else:
        prices["ticker_norm"] = ""

    if "yahoo_ticker" in prices.columns:
        prices["yahoo_ticker"] = prices["yahoo_ticker"].astype(str).str.strip()
    else:
        prices["yahoo_ticker"] = ""

    missing_yahoo = prices["yahoo_ticker"].eq("") | prices["yahoo_ticker"].isna() | prices["yahoo_ticker"].eq("nan")
    if missing_yahoo.any() and "ticker_norm" in prices.columns:
        inferable = missing_yahoo & prices["ticker_norm"].ne("")
        prices.loc[inferable, "yahoo_ticker"] = prices.loc[inferable, "ticker_norm"]
        log.info(f"MASTER: inferred yahoo_ticker from ticker for {int(inferable.sum())} rows")

    if not mapping.empty:
        prices = prices.merge(mapping, on="ticker_norm", how="left", suffixes=("", "_map"))
        needs_map = prices["yahoo_ticker"].eq("") | prices["yahoo_ticker"].isna() | prices["yahoo_ticker"].eq("nan")
        prices.loc[needs_map, "yahoo_ticker"] = prices.loc[needs_map, "yahoo_ticker_map"]
        prices = prices.drop(columns=[c for c in ["yahoo_ticker_map"] if c in prices.columns])
        log.info(f"MASTER: mapped yahoo_ticker from config for {int(needs_map.sum())} candidate rows")

    prices["missing_price_reason"] = np.where(
        prices["yahoo_ticker"].isna() | prices["yahoo_ticker"].eq("") | prices["yahoo_ticker"].eq("nan"),
        "missing_yahoo_ticker",
        "",
    )
    return prices


def _coalesce_master_yahoo_ticker(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    candidates = [c for c in ["yahoo_ticker", "yahoo_ticker_x", "yahoo_ticker_y"] if c in out.columns]
    if not candidates:
        out["yahoo_ticker"] = ""
        return out

    merged = pd.Series("", index=out.index, dtype=object)
    for c in candidates:
        s = out[c].astype(str).str.strip()
        valid = s.ne("") & s.ne("nan")
        merged = merged.where(merged.str.len() > 0, s.where(valid, ""))

    out["yahoo_ticker"] = merged
    drop_cols = [c for c in ["yahoo_ticker_x", "yahoo_ticker_y"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _latest_raw_snapshot_dir(raw_dir: Path, asof: str) -> Path | None:
    if not raw_dir.exists():
        return None
    asof_dt = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_dt):
        return None

    rows: list[tuple[pd.Timestamp, Path]] = []
    for d in raw_dir.iterdir():
        if not d.is_dir():
            continue
        ts = pd.to_datetime(d.name, format="%Y-%m-%d", errors="coerce")
        if pd.isna(ts):
            continue
        rows.append((ts.normalize(), d))

    if not rows:
        return None

    eligible = [r for r in rows if r[0] <= asof_dt.normalize()]
    if not eligible:
        return None

    eligible.sort(key=lambda x: x[0], reverse=True)
    return eligible[0][1]


def _latest_raw_snapshot_dir_with_data(raw_dir: Path, asof: str, required_any_subdirs: list[str]) -> Path | None:
    if not raw_dir.exists():
        return None
    asof_dt = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_dt):
        return None

    rows: list[tuple[pd.Timestamp, Path]] = []
    for d in raw_dir.iterdir():
        if not d.is_dir():
            continue
        ts = pd.to_datetime(d.name, format="%Y-%m-%d", errors="coerce")
        if pd.isna(ts) or ts.normalize() > asof_dt.normalize():
            continue
        ok = False
        for sub in required_any_subdirs:
            if (d / sub).exists():
                ok = True
                break
        if ok:
            rows.append((ts.normalize(), d))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[0][1]


def _previous_raw_snapshot_dir_with_data(raw_dir: Path, before_date: str, required_any_subdirs: list[str]) -> Path | None:
    if not raw_dir.exists():
        return None
    before_dt = pd.to_datetime(before_date, format="%Y-%m-%d", errors="coerce")
    if pd.isna(before_dt):
        return None

    rows: list[tuple[pd.Timestamp, Path]] = []
    for d in raw_dir.iterdir():
        if not d.is_dir():
            continue
        ts = pd.to_datetime(d.name, format="%Y-%m-%d", errors="coerce")
        if pd.isna(ts) or ts.normalize() >= before_dt.normalize():
            continue
        ok = False
        for sub in required_any_subdirs:
            if (d / sub).exists():
                ok = True
                break
        if ok:
            rows.append((ts.normalize(), d))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[0][1]


def _apply_reports_y_anomaly_guard(
    *,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    candidate_fields: list[str],
    guard_cfg: dict[str, Any],
    current_snapshot_name: str,
    previous_snapshot_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if current.empty:
        return (
            current,
            pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS),
            {"compared_rows": 0, "flagged_rows": 0, "excluded_rows": 0},
        )
    if previous.empty:
        return (
            current,
            pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS),
            {"compared_rows": 0, "flagged_rows": 0, "excluded_rows": 0},
        )

    ratio_threshold = float(guard_cfg.get("ratio_threshold", 100.0))
    min_changed_fields = int(guard_cfg.get("min_changed_fields", 5))
    min_abs_delta = float(guard_cfg.get("min_abs_delta", 1.0))
    near_zero_threshold = float(guard_cfg.get("near_zero_threshold", 1e-9))
    min_abs_level = float(guard_cfg.get("min_abs_level", 1.0))
    compare_fields = [f for f in candidate_fields if f in set(current.columns) and f in set(previous.columns)]
    if not compare_fields:
        return (
            current,
            pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS),
            {"compared_rows": 0, "flagged_rows": 0, "excluded_rows": 0},
        )

    ccols = ["yahoo_ticker", "financials_period_end"] + compare_fields
    pcols = ["yahoo_ticker", "financials_period_end"] + compare_fields
    cur = current[ccols].copy()
    prev = previous[pcols].copy()
    joined = cur.merge(prev, on=["yahoo_ticker", "financials_period_end"], how="left", suffixes=("_cur", "_prev"), validate="one_to_one")
    if joined.empty:
        return (
            current,
            pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS),
            {"compared_rows": 0, "flagged_rows": 0, "excluded_rows": 0},
        )

    jump_flags = pd.DataFrame(index=joined.index)
    has_prev = pd.Series(False, index=joined.index, dtype="boolean")
    for col in compare_fields:
        s_cur = pd.to_numeric(joined[f"{col}_cur"], errors="coerce")
        s_prev = pd.to_numeric(joined[f"{col}_prev"], errors="coerce")
        both = s_cur.notna() & s_prev.notna()
        has_prev = has_prev | s_prev.notna()
        abs_cur = s_cur.abs()
        abs_prev = s_prev.abs()
        abs_delta = (s_cur - s_prev).abs()

        ratio_up = abs_cur / abs_prev.replace(0.0, np.nan)
        ratio_down = abs_prev / abs_cur.replace(0.0, np.nan)
        ratio_jump = ratio_up.ge(ratio_threshold) | ratio_down.ge(ratio_threshold)
        zero_jump = ((abs_prev <= near_zero_threshold) & (abs_cur >= min_abs_level)) | (
            (abs_cur <= near_zero_threshold) & (abs_prev >= min_abs_level)
        )
        significant = abs_delta.ge(min_abs_delta) & (np.maximum(abs_cur, abs_prev) >= min_abs_level)
        jump_flags[col] = both & significant & (ratio_jump | zero_jump)

    if jump_flags.empty:
        return (
            current,
            pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS),
            {"compared_rows": int(has_prev.sum()), "flagged_rows": 0, "excluded_rows": 0},
        )

    changed_count = jump_flags.sum(axis=1).astype(int)
    anomaly_mask = has_prev & changed_count.ge(min_changed_fields)
    if not bool(anomaly_mask.any()):
        return (
            current,
            pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS),
            {"compared_rows": int(has_prev.sum()), "flagged_rows": 0, "excluded_rows": 0},
        )

    fields_series = jump_flags.apply(lambda r: ",".join([c for c, v in r.items() if bool(v)]), axis=1)
    reason = f"reports_y_scale_jump_vs_prev_snapshot:{previous_snapshot_name}->{current_snapshot_name}"
    flagged = joined.loc[anomaly_mask, ["yahoo_ticker"]].copy()
    flagged["financials_guard_anomaly"] = True
    flagged["reason_financials_guard"] = reason
    flagged["financials_guard_changed_fields"] = changed_count.loc[anomaly_mask].astype(float)
    flagged["financials_guard_fields"] = fields_series.loc[anomaly_mask]
    flagged["financials_guard_prev_snapshot"] = previous_snapshot_name
    flagged["financials_guard_current_snapshot"] = current_snapshot_name
    flagged = flagged.drop_duplicates(subset=["yahoo_ticker"], keep="first")

    kept = current[~current["yahoo_ticker"].isin(set(flagged["yahoo_ticker"]))].copy()
    return (
        kept,
        flagged,
        {
            "compared_rows": int(has_prev.sum()),
            "flagged_rows": int(len(flagged)),
            "excluded_rows": int(len(flagged)),
        },
    )


def _load_financials_tier_fields(project_root: Path, cfg: dict) -> tuple[list[str], list[str]]:
    fin_cfg = cfg.get("financials_enrichment", {}) or {}
    rel = str(fin_cfg.get("tiers_config", "config/financials_field_tiers.yaml"))
    path = Path(rel)
    if not path.is_absolute():
        path = (project_root / rel).resolve()
    if not path.exists():
        return DEFAULT_FINANCIALS_CORE_FIELDS, []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    core = [str(x).strip() for x in payload.get("core_model_input_fields", []) if str(x).strip()]
    enrich = [str(x).strip() for x in payload.get("enrichment_optional_fields", []) if str(x).strip()]
    if not core:
        core = DEFAULT_FINANCIALS_CORE_FIELDS
    return core, enrich


def _build_financials_feature_snapshot(
    *,
    project_root: Path,
    raw_root: Path,
    processed_root: Path,
    asof: str,
    ticker_map: pd.DataFrame,
    fields: list[str],
) -> pd.DataFrame:
    fin_dir = _latest_raw_snapshot_dir(processed_root, asof)
    if fin_dir is None:
        return pd.DataFrame(columns=["yahoo_ticker", "financials_period_end"] + fields)
    wide_path = fin_dir / "financials_wide.parquet"
    if not wide_path.exists():
        return pd.DataFrame(columns=["yahoo_ticker", "financials_period_end"] + fields)

    wide = pd.read_parquet(wide_path).copy()
    if wide.empty or "company_id" not in wide.columns or "period_end" not in wide.columns:
        return pd.DataFrame(columns=["yahoo_ticker", "financials_period_end"] + fields)

    docs_path = raw_root / fin_dir.name / "download_index.parquet"
    if docs_path.exists():
        docs = pd.read_parquet(docs_path)
    else:
        docs = pd.DataFrame(columns=["company_id", "ticker", "year"])

    if not docs.empty and "company_id" in docs.columns and "ticker" in docs.columns:
        for c in ["company_id", "ticker"]:
            docs[c] = docs[c].astype(str).str.strip()
        if "year" in docs.columns:
            docs["year"] = pd.to_numeric(docs["year"], errors="coerce")
            docs = docs.sort_values(["company_id", "year"], kind="mergesort")
        docs_map = docs[["company_id", "ticker"]].drop_duplicates(subset=["company_id"], keep="last")
    else:
        docs_map = pd.DataFrame({"company_id": wide["company_id"].astype(str).unique(), "ticker": ""})

    tmap = ticker_map.copy() if ticker_map is not None else pd.DataFrame(columns=["ticker_norm", "yahoo_ticker"])
    tmap["ticker_norm"] = tmap.get("ticker_norm", pd.Series(dtype=object)).astype(str).str.strip()
    tmap["yahoo_ticker"] = tmap.get("yahoo_ticker", pd.Series(dtype=object)).astype(str).str.strip()
    norm_to_yahoo = {
        str(r.ticker_norm): str(r.yahoo_ticker)
        for r in tmap.itertuples(index=False)
        if str(getattr(r, "ticker_norm", "")).strip() and str(getattr(r, "yahoo_ticker", "")).strip()
    }

    docs_map["ticker"] = docs_map["ticker"].astype(str).str.strip().str.upper()
    docs_map["ticker_norm"] = docs_map["ticker"].map(_norm_ticker)
    docs_map["yahoo_ticker"] = np.where(
        docs_map["ticker"].str.contains(r"\."),
        docs_map["ticker"],
        docs_map["ticker_norm"].map(lambda x: norm_to_yahoo.get(str(x), "")),
    )
    docs_map["yahoo_ticker"] = docs_map["yahoo_ticker"].astype(str).str.strip().str.upper()

    wide["company_id"] = wide["company_id"].astype(str).str.strip()
    wide["period_end_dt"] = pd.to_datetime(wide["period_end"], errors="coerce")
    wide = wide.merge(docs_map[["company_id", "yahoo_ticker"]], on="company_id", how="left")
    wide = wide[wide["yahoo_ticker"].astype(str).str.strip().ne("")].copy()
    if wide.empty:
        return pd.DataFrame(columns=["yahoo_ticker", "financials_period_end"] + fields)

    wide = wide.sort_values(["yahoo_ticker", "period_end_dt"], kind="mergesort").drop_duplicates(subset=["yahoo_ticker"], keep="last")
    if wide["yahoo_ticker"].duplicated().any():
        dups = wide[wide["yahoo_ticker"].duplicated(keep=False)]["yahoo_ticker"].tolist()
        raise ValueError(f"financials snapshot has duplicate yahoo_ticker rows: {dups[:10]}")

    keep_cols = ["yahoo_ticker", "period_end"] + [c for c in fields if c in wide.columns]
    out = wide[keep_cols].copy().rename(columns={"period_end": "financials_period_end"})
    out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip().str.upper()
    out["financials_source"] = "financials_agent"
    return out


def _build_reports_y_feature_snapshot(
    *,
    master: pd.DataFrame,
    raw_snapshot_dir: Path | None,
    asof_dt: pd.Timestamp,
    ins_map: pd.DataFrame,
    fields: list[str],
) -> pd.DataFrame:
    cols = ["yahoo_ticker", "financials_period_end", "financials_source"] + fields
    if raw_snapshot_dir is None or ins_map is None or ins_map.empty or master.empty:
        return pd.DataFrame(columns=cols)

    work = master[[c for c in ["ticker", "yahoo_ticker"] if c in master.columns]].copy()
    if "ticker" not in work.columns:
        work["ticker"] = ""
    work["ticker_norm"] = work["ticker"].map(_norm_ticker)
    work["yahoo_ticker"] = work["yahoo_ticker"].astype(str).str.strip().str.upper()
    work["yahoo_key"] = work["yahoo_ticker"]

    m = work.merge(ins_map, on="yahoo_key", how="left", suffixes=("", "_map"))
    missing_ins = m["ins_id"].isna() & m["ticker_norm"].ne("")
    if missing_ins.any():
        by_ticker = ins_map.dropna(subset=["ticker_norm"]).drop_duplicates(subset=["ticker_norm"], keep="first")
        by_ticker = by_ticker[["ticker_norm", "ins_id", "market"]].rename(columns={"ins_id": "ins_id_t", "market": "market_t"})
        m = m.merge(by_ticker, on="ticker_norm", how="left")
        m.loc[missing_ins, "ins_id"] = m.loc[missing_ins, "ins_id_t"]
        m.loc[missing_ins, "market"] = m.loc[missing_ins, "market_t"]
        m = m.drop(columns=[c for c in ["ins_id_t", "market_t"] if c in m.columns])

    market_from_suffix = m["yahoo_ticker"].map(lambda x: MARKET_BY_SUFFIX.get(_suffix_from_symbol(x), ""))
    m["market"] = m["market"].astype(str)
    m.loc[m["market"].eq("") | m["market"].eq("nan"), "market"] = market_from_suffix

    m["ins_id"] = pd.to_numeric(m["ins_id"], errors="coerce")
    m = m[m["ins_id"].notna() & m["market"].astype(str).ne("")].copy()
    if m.empty:
        return pd.DataFrame(columns=cols)

    m["ins_id"] = m["ins_id"].astype(int)
    m = m.drop_duplicates(subset=["yahoo_ticker", "ins_id", "market"], keep="first")

    rows: list[dict[str, Any]] = []
    for _, r in m.iterrows():
        market = str(r["market"]).upper()
        ins_id = int(r["ins_id"])
        p_y = raw_snapshot_dir / "reports_y" / f"market={market}" / f"ins_id={ins_id}.parquet"
        if not p_y.exists():
            continue
        rep_y = pd.read_parquet(p_y)
        latest = _pick_latest_row(rep_y, asof_dt=asof_dt)
        if latest is None:
            continue
        latest_d = _canon_cols(pd.DataFrame([latest])).iloc[0]
        row: dict[str, Any] = {
            "yahoo_ticker": str(r["yahoo_ticker"]).strip().upper(),
            "financials_source": "reports_y",
        }
        per_end = latest_d.get("report_end_date", latest_d.get("report_date", None))
        per_end_dt = pd.to_datetime(per_end, errors="coerce")
        if pd.notna(per_end_dt):
            row["financials_period_end"] = per_end_dt.date().isoformat()
        else:
            yr = pd.to_numeric(pd.Series([latest_d.get("year")]), errors="coerce").iloc[0]
            row["financials_period_end"] = f"{int(yr)}-12-31" if pd.notna(yr) else None

        # Direct canonical mappings from reports_y
        for src_col, dst_col in REPORTS_Y_TO_CANONICAL.items():
            if dst_col not in fields:
                continue
            if src_col in latest_d.index:
                row[dst_col] = pd.to_numeric(pd.Series([latest_d.get(src_col)]), errors="coerce").iloc[0]

        # Additional deterministic derivations from reports_y
        if "total_liabilities" in fields:
            total_liab = pd.to_numeric(pd.Series([latest_d.get("total_liabilities_and_equity")]), errors="coerce").iloc[0]
            total_eq = pd.to_numeric(pd.Series([latest_d.get("total_equity")]), errors="coerce").iloc[0]
            if pd.notna(total_liab) and pd.notna(total_eq):
                row["total_liabilities"] = float(total_liab - total_eq)
            else:
                ncl = pd.to_numeric(pd.Series([latest_d.get("non_current_liabilities")]), errors="coerce").iloc[0]
                cl = pd.to_numeric(pd.Series([latest_d.get("current_liabilities")]), errors="coerce").iloc[0]
                if pd.notna(ncl) and pd.notna(cl):
                    row["total_liabilities"] = float(ncl + cl)

        if "shares_outstanding_diluted" in fields and pd.notna(row.get("shares_outstanding_basic")):
            row["shares_outstanding_diluted"] = row.get("shares_outstanding_basic")

        cur = str(latest_d.get("currency", "")).strip().upper()
        if "reporting_currency" in fields and cur:
            row["reporting_currency"] = cur
        if "fiscal_year" in fields:
            yr = pd.to_numeric(pd.Series([latest_d.get("year")]), errors="coerce").iloc[0]
            row["fiscal_year"] = yr if pd.notna(yr) else np.nan
        if "reporting_period_type" in fields:
            row["reporting_period_type"] = "annual"

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    out = out.sort_values(["yahoo_ticker", "financials_period_end"], kind="mergesort").drop_duplicates(subset=["yahoo_ticker"], keep="last")
    if out["yahoo_ticker"].duplicated().any():
        dups = out[out["yahoo_ticker"].duplicated(keep=False)]["yahoo_ticker"].tolist()
        raise ValueError(f"reports_y snapshot has duplicate yahoo_ticker rows: {dups[:10]}")
    keep = [c for c in cols if c in out.columns]
    return out[keep]


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def canon(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    out.columns = [canon(c) for c in out.columns]
    return out


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pick_latest_row(df: pd.DataFrame, asof_dt: pd.Timestamp) -> pd.Series | None:
    if df is None or df.empty:
        return None

    d = _canon_cols(df)

    date_col = "report_date" if "report_date" in d.columns else ("report_end_date" if "report_end_date" in d.columns else None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d[d[date_col].notna() & (d[date_col] <= asof_dt)]

    if d.empty:
        return None

    if "year" in d.columns:
        d["year"] = _to_num(d["year"])
    if "period" in d.columns:
        d["period"] = _to_num(d["period"])

    sort_cols = [c for c in ["year", "period"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols)
    return d.iloc[-1]


def _annual_frame(df: pd.DataFrame, asof_dt: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = _canon_cols(df)
    if "year" not in d.columns:
        return pd.DataFrame()

    date_col = "report_date" if "report_date" in d.columns else ("report_end_date" if "report_end_date" in d.columns else None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d[d[date_col].notna() & (d[date_col] <= asof_dt)]

    d["year"] = _to_num(d["year"])
    d = d[d["year"].notna()].copy()
    if d.empty:
        return d

    if "period" in d.columns:
        d["period"] = _to_num(d["period"])
    else:
        d["period"] = 0

    d = d.sort_values(["year", "period"]).drop_duplicates(subset=["year"], keep="last")
    return d


def _cagr_from_points(latest_year: float, latest_value: float, older_year: float, older_value: float) -> float:
    if not np.isfinite(latest_year) or not np.isfinite(older_year):
        return float("nan")
    if not np.isfinite(latest_value) or not np.isfinite(older_value):
        return float("nan")
    if latest_value <= 0 or older_value <= 0:
        return float("nan")
    years = float(latest_year - older_year)
    if years <= 0:
        return float("nan")
    return float((latest_value / older_value) ** (1.0 / years) - 1.0)


def _calc_growth_5y(annual: pd.DataFrame, value_col: str) -> float:
    if annual.empty or value_col not in annual.columns:
        return float("nan")

    d = annual[["year", value_col]].copy()
    d[value_col] = _to_num(d[value_col])
    d = d[d[value_col].notna() & (d[value_col] > 0)].sort_values("year")
    if len(d) < 2:
        return float("nan")

    latest = d.iloc[-1]
    target_year = float(latest["year"]) - 5.0
    older = d[d["year"] <= target_year]
    if older.empty:
        return float("nan")

    old = older.iloc[-1]
    return _cagr_from_points(float(latest["year"]), float(latest[value_col]), float(old["year"]), float(old[value_col]))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if den == 0:
        return float("nan")
    return float(num / den)


def _derive_tier23_financial_metrics(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    out = master.copy()
    idx = out.index

    def num(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(np.nan, index=idx, dtype="float64")

    def sum_any(parts: list[pd.Series]) -> pd.Series:
        if not parts:
            return pd.Series(np.nan, index=idx, dtype="float64")
        total = pd.Series(0.0, index=idx, dtype="float64")
        mask = pd.Series(False, index=idx, dtype="boolean")
        for p in parts:
            pp = pd.to_numeric(p, errors="coerce")
            total = total + pp.fillna(0.0)
            mask = mask | pp.notna()
        return total.where(mask, np.nan)

    def first_valid(candidates: list[tuple[str, pd.Series]]) -> tuple[pd.Series, pd.Series]:
        value = pd.Series(np.nan, index=idx, dtype="float64")
        method = pd.Series("", index=idx, dtype="object")
        for name, s in candidates:
            ss = pd.to_numeric(s, errors="coerce")
            use = value.isna() & ss.notna()
            value = value.where(~use, ss)
            method = method.where(~use, name)
        return value, method

    coverage: dict[str, float] = {}

    def set_metric(name: str, value: pd.Series, method: pd.Series) -> None:
        v = pd.to_numeric(value, errors="coerce")
        out[name] = v
        out[f"{name}_derived_from"] = method.where(v.notna(), "")
        coverage[name] = float(v.notna().mean())

    revenue = num("revenue_total")
    ebit = num("operating_income_ebit")
    pbt = num("profit_before_tax")
    tax = num("income_tax_expense")
    ni_parent = num("net_income_attributable_to_parent")
    ni = num("net_income")
    cfo = num("cash_flow_from_operations")
    gross = num("gross_profit")
    finance_costs = num("finance_costs")
    interest_paid = num("interest_paid_cash")
    total_assets = num("total_assets")
    total_equity = num("total_equity")
    total_liabilities = num("total_liabilities")

    # Tier-2
    etr = (tax / pbt).where(pbt.notna() & pbt.ne(0.0), np.nan)
    set_metric("t2_effective_tax_rate", etr, pd.Series("income_tax_expense/profit_before_tax", index=idx))

    dep = num("depreciation_expense").abs()
    amort = num("amortization_expense").abs()
    ebitda_reported = num("ebitda")
    ebitda_fallback = (ebit + dep.fillna(0.0) + amort.fillna(0.0)).where(ebit.notna() & (dep.notna() | amort.notna()), np.nan)
    ebitda_used, ebitda_used_m = first_valid(
        [
            ("reported:ebitda", ebitda_reported),
            ("fallback:operating_income_ebit+abs(depreciation_expense)+abs(amortization_expense)", ebitda_fallback),
        ]
    )
    set_metric("t2_ebitda_used", ebitda_used, ebitda_used_m)

    gross_margin = (gross / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)
    set_metric("t2_gross_margin", gross_margin, pd.Series("gross_profit/revenue_total", index=idx))

    ebit_margin = (ebit / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)
    set_metric("t2_ebit_margin", ebit_margin, pd.Series("operating_income_ebit/revenue_total", index=idx))

    ebitda_margin = (ebitda_used / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)
    set_metric("t2_ebitda_margin", ebitda_margin, ebitda_used_m.where(ebitda_margin.notna(), "") + "/revenue_total")

    net_margin_parent, net_margin_parent_m = first_valid(
        [
            ("net_income_attributable_to_parent/revenue_total", (ni_parent / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)),
            ("fallback:net_income/revenue_total", (ni / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)),
        ]
    )
    set_metric("t2_net_margin", net_margin_parent, net_margin_parent_m)

    cfo_margin = (cfo / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)
    set_metric("t2_cfo_margin", cfo_margin, pd.Series("cash_flow_from_operations/revenue_total", index=idx))

    capex_ppe = num("capex_purchase_ppe").abs()
    capex_int = num("capex_purchase_intangibles").abs()
    capex_total = sum_any([capex_ppe, capex_int])
    fcf_strict = (cfo - capex_total).where(cfo.notna() & capex_total.notna(), np.nan)
    fcf_used, fcf_used_m = first_valid(
        [
            ("cash_flow_from_operations-abs(capex_purchase_ppe)-abs(capex_purchase_intangibles)", fcf_strict),
            ("fallback:cash_flow_from_operations", cfo),
        ]
    )
    set_metric("t2_fcf_simple", fcf_used, fcf_used_m)
    fcf_margin = (fcf_used / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)
    set_metric("t2_fcf_margin", fcf_margin, fcf_used_m.where(fcf_margin.notna(), "") + "/revenue_total")

    debt_total = sum_any([num("short_term_debt"), num("long_term_debt")])
    lease_total = sum_any([num("lease_liabilities_current"), num("lease_liabilities_noncurrent")])
    liquidity_total = sum_any([num("cash_and_cash_equivalents"), num("short_term_investments")])
    net_debt_excl = (debt_total - liquidity_total).where(debt_total.notna() & liquidity_total.notna(), np.nan)
    set_metric("t2_net_debt_excl_leases", net_debt_excl, pd.Series("(short_term_debt+long_term_debt)-(cash_and_cash_equivalents+short_term_investments)", index=idx))
    net_debt_incl, net_debt_incl_m = first_valid(
        [
            ("(short_term_debt+long_term_debt+lease_liabilities_current+lease_liabilities_noncurrent)-(cash_and_cash_equivalents+short_term_investments)", (debt_total + lease_total - liquidity_total).where(debt_total.notna() & lease_total.notna() & liquidity_total.notna(), np.nan)),
            ("fallback:t2_net_debt_excl_leases", net_debt_excl),
        ]
    )
    set_metric("t2_net_debt_incl_leases", net_debt_incl, net_debt_incl_m)

    nd_ebitda_excl = (net_debt_excl / ebitda_used).where(ebitda_used.notna() & ebitda_used.ne(0.0), np.nan)
    set_metric("t2_nd_ebitda_excl_leases", nd_ebitda_excl, pd.Series("t2_net_debt_excl_leases/t2_ebitda_used", index=idx))
    nd_ebitda_incl = (net_debt_incl / ebitda_used).where(ebitda_used.notna() & ebitda_used.ne(0.0), np.nan)
    set_metric("t2_nd_ebitda_incl_leases", nd_ebitda_incl, pd.Series("t2_net_debt_incl_leases/t2_ebitda_used", index=idx))

    interest_cov, interest_cov_m = first_valid(
        [
            ("operating_income_ebit/abs(finance_costs)", (ebit / finance_costs.abs()).where(ebit.notna() & finance_costs.abs().gt(0.0), np.nan)),
            ("fallback:operating_income_ebit/abs(interest_paid_cash)", (ebit / interest_paid.abs()).where(ebit.notna() & interest_paid.abs().gt(0.0), np.nan)),
        ]
    )
    set_metric("t2_interest_coverage", interest_cov, interest_cov_m)

    # Tier-3
    ar = num("accounts_receivable")
    inv = num("inventory")
    ap = num("accounts_payable")
    wc = (ar + inv - ap).where(ar.notna() & inv.notna() & ap.notna(), np.nan)
    set_metric("t3_working_capital", wc, pd.Series("accounts_receivable+inventory-accounts_payable", index=idx))
    wc_rev = (wc / revenue).where(revenue.notna() & revenue.ne(0.0), np.nan)
    set_metric("t3_working_capital_to_revenue", wc_rev, pd.Series("t3_working_capital/revenue_total", index=idx))

    roa_used, roa_m = first_valid(
        [
            ("net_income_attributable_to_parent/total_assets", (ni_parent / total_assets).where(total_assets.notna() & total_assets.ne(0.0), np.nan)),
            ("fallback:net_income/total_assets", (ni / total_assets).where(total_assets.notna() & total_assets.ne(0.0), np.nan)),
        ]
    )
    set_metric("t3_roa", roa_used, roa_m)

    lev = (total_liabilities / total_equity).where(total_equity.notna() & total_equity.ne(0.0), np.nan)
    set_metric("t3_liabilities_to_equity", lev, pd.Series("total_liabilities/total_equity", index=idx))

    cash_conv, cash_conv_m = first_valid(
        [
            ("cash_flow_from_operations/net_income_attributable_to_parent", (cfo / ni_parent).where(ni_parent.notna() & ni_parent.ne(0.0), np.nan)),
            ("fallback:cash_flow_from_operations/net_income", (cfo / ni).where(ni.notna() & ni.ne(0.0), np.nan)),
        ]
    )
    set_metric("t3_cash_conversion", cash_conv, cash_conv_m)

    return out, coverage


def _derive_dividend_features(reports_y: pd.DataFrame, reports_r12: pd.DataFrame, asof_dt: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {k: float("nan") for k in DIVIDEND_FEATURE_COLS}

    annual = _annual_frame(reports_y, asof_dt)
    if not annual.empty:
        div = _to_num(annual.get("dividend", pd.Series(np.nan, index=annual.index)))
        hist_mask = div > 0
        out["dividend_history_years"] = float(int(hist_mask.sum()))
        out["dividend_paid_years"] = out["dividend_history_years"]
        out["dividend_growth_5y"] = _calc_growth_5y(annual, "dividend")
        out["dividend_cagr_5y"] = out["dividend_growth_5y"]
        out["profit_growth_5y"] = _calc_growth_5y(annual, "profit_to_equity_holders")
        out["share_count_growth_5y"] = _calc_growth_5y(annual, "number_of_shares")

        latest_a = annual.iloc[-1]
        dps = float(_to_num(pd.Series([latest_a.get("dividend")])).iloc[0])
        eps = float(_to_num(pd.Series([latest_a.get("earnings_per_share")])).iloc[0])
        px = float(_to_num(pd.Series([latest_a.get("stock_price_average")])).iloc[0])

        if np.isfinite(dps) and dps >= 0 and np.isfinite(eps) and eps > 0:
            out["payout_ratio"] = _safe_ratio(dps, eps)
        if np.isfinite(dps) and dps >= 0 and np.isfinite(px) and px > 0:
            out["dividend_yield"] = _safe_ratio(dps, px)

    latest_r12 = _pick_latest_row(reports_r12, asof_dt)
    if latest_r12 is not None:
        rev = float(_to_num(pd.Series([latest_r12.get("revenues", latest_r12.get("net_sales"))])).iloc[0])
        profit = float(_to_num(pd.Series([latest_r12.get("profit_to_equity_holders")])).iloc[0])
        ocf = float(_to_num(pd.Series([latest_r12.get("cash_flow_from_operating_activities")])).iloc[0])
        fcf = float(_to_num(pd.Series([latest_r12.get("free_cash_flow")])).iloc[0])
        assets = float(_to_num(pd.Series([latest_r12.get("total_assets")])).iloc[0])

        out["fcf_margin"] = _safe_ratio(fcf, rev)
        out["ocf_margin"] = _safe_ratio(ocf, rev)
        out["profit_margin"] = _safe_ratio(profit, rev)
        out["roa"] = _safe_ratio(profit, assets)

        if not np.isfinite(out["payout_ratio"]):
            dps = float(_to_num(pd.Series([latest_r12.get("dividend")])).iloc[0])
            eps = float(_to_num(pd.Series([latest_r12.get("earnings_per_share")])).iloc[0])
            if np.isfinite(dps) and dps >= 0 and np.isfinite(eps) and eps > 0:
                out["payout_ratio"] = _safe_ratio(dps, eps)

        if not np.isfinite(out["dividend_yield"]):
            dps = float(_to_num(pd.Series([latest_r12.get("dividend")])).iloc[0])
            px = float(_to_num(pd.Series([latest_r12.get("stock_price_average")])).iloc[0])
            if np.isfinite(dps) and dps >= 0 and np.isfinite(px) and px > 0:
                out["dividend_yield"] = _safe_ratio(dps, px)

    return out


def _build_dividend_feature_snapshot(
    master: pd.DataFrame,
    raw_snapshot_dir: Path | None,
    asof_dt: pd.Timestamp,
    ins_map: pd.DataFrame,
) -> pd.DataFrame:
    if raw_snapshot_dir is None or not raw_snapshot_dir.exists() or ins_map.empty:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    if "yahoo_ticker" not in master.columns:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    work = master[[c for c in ["ticker", "yahoo_ticker"] if c in master.columns]].copy()
    if "ticker" not in work.columns:
        work["ticker"] = ""
    work["ticker_norm"] = work["ticker"].map(_norm_ticker)
    work["yahoo_ticker"] = work["yahoo_ticker"].astype(str).str.strip()
    work["yahoo_key"] = work["yahoo_ticker"].str.upper()

    m = work.merge(ins_map, on="yahoo_key", how="left", suffixes=("", "_map"))
    missing_ins = m["ins_id"].isna() & m["ticker_norm"].ne("")
    if missing_ins.any():
        by_ticker = ins_map.dropna(subset=["ticker_norm"]).drop_duplicates(subset=["ticker_norm"], keep="first")
        by_ticker = by_ticker[["ticker_norm", "ins_id", "market"]].rename(
            columns={"ins_id": "ins_id_t", "market": "market_t"}
        )
        m = m.merge(by_ticker, on="ticker_norm", how="left")
        m.loc[missing_ins, "ins_id"] = m.loc[missing_ins, "ins_id_t"]
        m.loc[missing_ins, "market"] = m.loc[missing_ins, "market_t"]
        m = m.drop(columns=[c for c in ["ins_id_t", "market_t"] if c in m.columns])

    market_from_suffix = m["yahoo_ticker"].map(lambda x: MARKET_BY_SUFFIX.get(_suffix_from_symbol(x), ""))
    m["market"] = m["market"].astype(str)
    m.loc[m["market"].eq("") | m["market"].eq("nan"), "market"] = market_from_suffix

    m["ins_id"] = pd.to_numeric(m["ins_id"], errors="coerce")
    m = m[m["ins_id"].notna() & m["market"].astype(str).ne("")].copy()
    if m.empty:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    m["ins_id"] = m["ins_id"].astype(int)
    m = m.drop_duplicates(subset=["yahoo_ticker", "ins_id", "market"], keep="first")

    rows: list[dict[str, float | str]] = []
    for _, r in m.iterrows():
        market = str(r["market"]).upper()
        ins_id = int(r["ins_id"])
        p_y = raw_snapshot_dir / "reports_y" / f"market={market}" / f"ins_id={ins_id}.parquet"
        p_r12 = raw_snapshot_dir / "reports_r12" / f"market={market}" / f"ins_id={ins_id}.parquet"
        if (not p_y.exists()) and (not p_r12.exists()):
            continue

        rep_y = pd.read_parquet(p_y) if p_y.exists() else pd.DataFrame()
        rep_r12 = pd.read_parquet(p_r12) if p_r12.exists() else pd.DataFrame()
        feat = _derive_dividend_features(rep_y, rep_r12, asof_dt=asof_dt)

        row: dict[str, float | str] = {"yahoo_ticker": str(r["yahoo_ticker"])}
        row.update(feat)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    out = pd.DataFrame(rows).drop_duplicates(subset=["yahoo_ticker"], keep="first")
    return out


def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    raw_dir = paths["raw_dir"]
    processed_dir = paths["processed_dir"]
    financials_info = {
        "enabled": bool((ctx.cfg.get("financials_enrichment", {}) or {}).get("enabled", True)),
        "snapshot_date": None,
        "matched_rows": 0,
        "matched_agent_rows": 0,
        "matched_reports_y_rows": 0,
        "core_coverage": {},
        "enrich_coverage": {},
        "guard_enabled": False,
        "guard_current_snapshot": None,
        "guard_previous_snapshot": None,
        "guard_compared_rows": 0,
        "guard_flagged_rows": 0,
        "guard_excluded_rows": 0,
    }
    tier23_coverage: dict[str, float] = {}

    # 1) Base master: prefer raw/master_input if present (som hos deg)
    master_input = raw_dir / "master_input.parquet"
    if master_input.exists():
        master = read_parquet(master_input)
        log.info("MASTER: base from master_input")
    else:
        # fallback: fundamentals snapshot
        fpath = processed_dir / "fundamentals.parquet"
        master = read_parquet(fpath)
        log.info("MASTER: base from fundamentals.parquet")

    if "ticker" not in master.columns:
        raise ValueError("MASTER: missing 'ticker' in base dataset")

    master = master.copy()
    master["ticker_norm"] = master["ticker"].map(_norm_ticker)

    # 2) Prices snapshot (siste dato <= asof) ALWAYS merged in
    ppath = processed_dir / "prices.parquet"
    if not ppath.exists():
        raise ValueError(f"MASTER: missing {ppath} (run transform_prices first)")

    prices = read_parquet(ppath).copy()
    if "ticker" not in prices.columns or "date" not in prices.columns:
        raise ValueError("MASTER: prices.parquet must have ticker,date")

    mapping = _load_ticker_mapping(ctx)
    prices = _normalize_prices_with_mapping(prices, mapping, log)

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker_norm"] = prices["ticker"].map(_norm_ticker)

    # unngå å blande inn indekser i master (tickere som starter med ^)
    # (etter norm er ^OSEAX -> OSEAX, så filtrer før norm hvis felt finnes)
    if prices["ticker"].astype(str).str.startswith("^").any():
        prices = prices[~prices["ticker"].astype(str).str.startswith("^")].copy()

    asof_dt = pd.to_datetime(ctx.asof)
    prices = prices[prices["date"] <= asof_dt].copy()

    # compute MA/MAD if needed
    prices = _ensure_price_features(prices)

    snap = (
        prices.sort_values(["ticker_norm", "date"])
        .groupby("ticker_norm", as_index=False)
        .tail(1)
        .copy()
    )

    keep_cols = [c for c in ["ticker_norm", "yahoo_ticker", "date", "adj_close", "volume", "ma21", "ma200", "mad", "above_ma200", "missing_price_reason"] if c in snap.columns]
    snap = snap[keep_cols]

    # 3) Drop any old price cols in master (prevents "all NaN" from master_input)
    drop_old = [c for c in ["date", "adj_close", "volume", "ma21", "ma200", "mad", "above_ma200", "missing_price", "missing_technical_data", "reason_technical_fail"] if c in master.columns]
    if drop_old:
        master = master.drop(columns=drop_old)

    master = master.merge(snap, on="ticker_norm", how="left")
    master = _coalesce_master_yahoo_ticker(master)

    # 3b) Optional: enrich master with financials-agent snapshot (as-of, latest period per ticker)
    fin_cfg = ctx.cfg.get("financials_enrichment", {}) or {}
    if bool(fin_cfg.get("enabled", True)):
        guard_flags = pd.DataFrame(columns=["yahoo_ticker"] + FINANCIALS_GUARD_COLS)
        core_fields, enrich_fields = _load_financials_tier_fields(ctx.project_root, ctx.cfg)
        requested_fields = core_fields + [f for f in enrich_fields if f not in set(core_fields)]
        guard_cfg = fin_cfg.get("anomaly_guard", {}) or {}
        financials_info["guard_enabled"] = bool(guard_cfg.get("enabled", True))
        fin_snap = _build_financials_feature_snapshot(
            project_root=ctx.project_root,
            raw_root=raw_dir,
            processed_root=processed_dir,
            asof=ctx.asof,
            ticker_map=mapping,
            fields=requested_fields,
        )
        raw_snapshot_dir = _latest_raw_snapshot_dir_with_data(raw_dir, ctx.asof, ["reports_y"])
        ins_map = _load_insid_mapping(ctx)
        bridge_enabled = bool(fin_cfg.get("include_reports_y_bridge", True))
        rep_snap = _build_reports_y_feature_snapshot(
            master=master,
            raw_snapshot_dir=raw_snapshot_dir if bridge_enabled else None,
            asof_dt=asof_dt,
            ins_map=ins_map if bridge_enabled else pd.DataFrame(),
            fields=requested_fields,
        )
        if bridge_enabled and financials_info["guard_enabled"] and raw_snapshot_dir is not None and not rep_snap.empty:
            prev_raw_snapshot_dir = _previous_raw_snapshot_dir_with_data(raw_dir, raw_snapshot_dir.name, ["reports_y"])
            if prev_raw_snapshot_dir is not None:
                rep_snap_prev = _build_reports_y_feature_snapshot(
                    master=master,
                    raw_snapshot_dir=prev_raw_snapshot_dir,
                    asof_dt=asof_dt,
                    ins_map=ins_map,
                    fields=requested_fields,
                )
                guard_fields = [f for f in DEFAULT_REPORTS_Y_GUARD_FIELDS if f in requested_fields]
                rep_snap, guard_flags, guard_stats = _apply_reports_y_anomaly_guard(
                    current=rep_snap,
                    previous=rep_snap_prev,
                    candidate_fields=guard_fields,
                    guard_cfg=guard_cfg,
                    current_snapshot_name=raw_snapshot_dir.name,
                    previous_snapshot_name=prev_raw_snapshot_dir.name,
                )
                financials_info["guard_current_snapshot"] = raw_snapshot_dir.name
                financials_info["guard_previous_snapshot"] = prev_raw_snapshot_dir.name
                financials_info["guard_compared_rows"] = int(guard_stats.get("compared_rows", 0))
                financials_info["guard_flagged_rows"] = int(guard_stats.get("flagged_rows", 0))
                financials_info["guard_excluded_rows"] = int(guard_stats.get("excluded_rows", 0))
                if financials_info["guard_excluded_rows"] > 0:
                    log.info(
                        "MASTER: reports_y anomaly guard excluded=%s rows (compared=%s, prev=%s, curr=%s)",
                        financials_info["guard_excluded_rows"],
                        financials_info["guard_compared_rows"],
                        financials_info["guard_previous_snapshot"],
                        financials_info["guard_current_snapshot"],
                    )
            else:
                log.info("MASTER: reports_y anomaly guard skipped (no previous reports_y snapshot).")

        if not fin_snap.empty and fin_snap["yahoo_ticker"].duplicated().any():
            dups = fin_snap[fin_snap["yahoo_ticker"].duplicated(keep=False)]["yahoo_ticker"].tolist()
            raise ValueError(f"Financials enrichment duplicate yahoo_ticker rows (agent): {dups[:10]}")
        if not rep_snap.empty and rep_snap["yahoo_ticker"].duplicated().any():
            dups = rep_snap[rep_snap["yahoo_ticker"].duplicated(keep=False)]["yahoo_ticker"].tolist()
            raise ValueError(f"Financials enrichment duplicate yahoo_ticker rows (reports_y): {dups[:10]}")

        combined = pd.DataFrame(columns=["yahoo_ticker", "financials_period_end", "financials_source"] + requested_fields)
        if not rep_snap.empty and not fin_snap.empty:
            joined = rep_snap.merge(fin_snap, on="yahoo_ticker", how="outer", suffixes=("_ry", "_fa"), validate="one_to_one")
            combined["yahoo_ticker"] = joined["yahoo_ticker"]
            combined["financials_period_end"] = joined["financials_period_end_fa"].where(
                joined["financials_period_end_fa"].notna(),
                joined["financials_period_end_ry"],
            )
            combined["financials_source"] = np.where(
                joined["financials_source_fa"].notna(),
                joined["financials_source_fa"],
                joined["financials_source_ry"],
            )
            for c in requested_fields:
                c_fa = f"{c}_fa"
                c_ry = f"{c}_ry"
                if c_fa in joined.columns and c_ry in joined.columns:
                    combined[c] = joined[c_fa].where(joined[c_fa].notna(), joined[c_ry])
                elif c_fa in joined.columns:
                    combined[c] = joined[c_fa]
                elif c_ry in joined.columns:
                    combined[c] = joined[c_ry]
        elif not fin_snap.empty:
            combined = fin_snap.copy()
        elif not rep_snap.empty:
            combined = rep_snap.copy()

        if not combined.empty:
            combined["yahoo_ticker"] = combined["yahoo_ticker"].astype(str).str.strip().str.upper()
            master["yahoo_ticker"] = master["yahoo_ticker"].astype(str).str.strip().str.upper()
            merged = master.merge(combined, on="yahoo_ticker", how="left", suffixes=("", "_fin"), validate="many_to_one")

            if "financials_period_end_fin" in merged.columns:
                if "financials_period_end" in merged.columns:
                    merged["financials_period_end"] = merged["financials_period_end"].where(
                        merged["financials_period_end"].notna(),
                        merged["financials_period_end_fin"],
                    )
                    merged = merged.drop(columns=["financials_period_end_fin"])
                else:
                    merged = merged.rename(columns={"financials_period_end_fin": "financials_period_end"})

            if "financials_source_fin" in merged.columns:
                if "financials_source" in merged.columns:
                    merged["financials_source"] = merged["financials_source"].where(
                        merged["financials_source"].notna(),
                        merged["financials_source_fin"],
                    )
                    merged = merged.drop(columns=["financials_source_fin"])
                else:
                    merged = merged.rename(columns={"financials_source_fin": "financials_source"})

            for c in requested_fields:
                fc = f"{c}_fin"
                if fc not in merged.columns:
                    continue
                if c in merged.columns:
                    merged[c] = merged[c].where(merged[c].notna(), merged[fc])
                else:
                    merged[c] = merged[fc]
            merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_fin")])
            master = merged

            financials_info["snapshot_date"] = str(pd.to_datetime(master.get("financials_period_end"), errors="coerce").max().date()) if "financials_period_end" in master.columns else None
            financials_info["matched_rows"] = int(master["financials_period_end"].notna().sum()) if "financials_period_end" in master.columns else 0
            financials_info["matched_agent_rows"] = int((master.get("financials_source", pd.Series(dtype=object)) == "financials_agent").sum())
            financials_info["matched_reports_y_rows"] = int((master.get("financials_source", pd.Series(dtype=object)) == "reports_y").sum())
            for col in core_fields:
                if col in master.columns:
                    financials_info["core_coverage"][col] = float(pd.to_numeric(master[col], errors="coerce").notna().mean())
            for col in enrich_fields:
                if col in master.columns:
                    financials_info["enrich_coverage"][col] = float(pd.to_numeric(master[col], errors="coerce").notna().mean())
            log.info(
                "MASTER: financials enrichment matched=%s rows (agent=%s,reports_y=%s); core fields tracked=%s, enrichment fields tracked=%s",
                financials_info["matched_rows"],
                financials_info["matched_agent_rows"],
                financials_info["matched_reports_y_rows"],
                len(financials_info["core_coverage"]),
                len(financials_info["enrich_coverage"]),
            )
        else:
            log.info("MASTER: financials enrichment enabled but no financials snapshot found <= asof.")

        if not guard_flags.empty:
            guard_flags["yahoo_ticker"] = guard_flags["yahoo_ticker"].astype(str).str.strip().str.upper()
            master["yahoo_ticker"] = master["yahoo_ticker"].astype(str).str.strip().str.upper()
            master = master.merge(guard_flags, on="yahoo_ticker", how="left", validate="many_to_one")
        for col in FINANCIALS_GUARD_COLS:
            if col not in master.columns:
                if col == "financials_guard_anomaly":
                    master[col] = False
                elif col == "financials_guard_changed_fields":
                    master[col] = np.nan
                else:
                    master[col] = ""
        master["financials_guard_anomaly"] = np.where(master["financials_guard_anomaly"].isna(), False, master["financials_guard_anomaly"]).astype(bool)
        master["reason_financials_guard"] = master["reason_financials_guard"].fillna("").astype(str)

    # 4) Enrich dividend-quality features from raw reports snapshot (as-of)
    raw_snapshot_dir = _latest_raw_snapshot_dir_with_data(raw_dir, ctx.asof, ["reports_y", "reports_r12"])
    ins_map = _load_insid_mapping(ctx)
    div_snapshot = _build_dividend_feature_snapshot(master=master, raw_snapshot_dir=raw_snapshot_dir, asof_dt=asof_dt, ins_map=ins_map)
    if not div_snapshot.empty:
        for col in DIVIDEND_FEATURE_COLS:
            if col not in div_snapshot.columns:
                div_snapshot[col] = np.nan

        merged = master.merge(
            div_snapshot[["yahoo_ticker"] + DIVIDEND_FEATURE_COLS],
            on="yahoo_ticker",
            how="left",
            suffixes=("", "_derived"),
        )
        for c in DIVIDEND_FEATURE_COLS:
            dc = f"{c}_derived"
            if dc not in merged.columns:
                continue
            if c in merged.columns:
                merged[c] = merged[c].where(merged[c].notna(), merged[dc])
            else:
                merged[c] = merged[dc]
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_derived")])
        master = merged

    master, tier23_coverage = _derive_tier23_financial_metrics(master)
    if tier23_coverage:
        nonzero = sum(1 for v in tier23_coverage.values() if v > 0.0)
        log.info("MASTER: tier2/tier3 derived metrics computed=%s, with_nonzero_coverage=%s", len(tier23_coverage), nonzero)

    price_s = pd.to_numeric(master.get("adj_close"), errors="coerce")
    ma200_s = pd.to_numeric(master.get("ma200"), errors="coerce")
    mad_s = pd.to_numeric(master.get("mad"), errors="coerce")

    master["above_ma200"] = pd.Series(
        np.where(price_s.notna() & ma200_s.notna(), price_s > ma200_s, pd.NA),
        index=master.index,
        dtype="boolean",
    )
    master["ma200_ok"] = price_s.notna() & ma200_s.notna() & (price_s > ma200_s)
    master["index_ma200_ok"] = True
    master["missing_price"] = price_s.isna()
    master["missing_technical_data"] = (~master["missing_price"]) & (ma200_s.isna() | mad_s.isna())
    master["reason_technical_fail"] = ""
    master.loc[master["missing_price"], "reason_technical_fail"] = "missing_price"
    master.loc[master["missing_technical_data"], "reason_technical_fail"] = "missing_technical_data"
    master.loc[~master["missing_price"] & ~master["missing_technical_data"] & ~master["ma200_ok"], "reason_technical_fail"] = "below_ma200"
    if "mad" in master.columns:
        master.loc[~master["missing_price"] & ~master["missing_technical_data"] & master["reason_technical_fail"].eq("") & mad_s.lt(0), "reason_technical_fail"] = "negative_mad"
    master = master.drop(columns=["ticker_norm"])

    # Coverage log
    cov = float(pd.to_numeric(master["adj_close"], errors="coerce").notna().mean()) if "adj_close" in master.columns else 0.0
    log.info(f"MASTER: price coverage adj_close={cov:.3f}")

    dividend_coverage: dict[str, float] = {}
    for col in DIVIDEND_FEATURE_COLS:
        if col in master.columns:
            dividend_coverage[col] = float(pd.to_numeric(master[col], errors="coerce").notna().mean())

    if raw_snapshot_dir is not None:
        log.info(f"MASTER: dividend feature source snapshot={raw_snapshot_dir.name}")
    if dividend_coverage:
        top_cov = ", ".join([f"{k}={v:.1%}" for k, v in dividend_coverage.items()])
        log.info(f"MASTER: dividend feature coverage {top_cov}")

    # Data quality summary for explainability
    missing_count = int(master["missing_price"].sum()) if "missing_price" in master.columns else 0
    no_yahoo = int((master.get("yahoo_ticker").astype(str).str.strip() == "").sum()) if "yahoo_ticker" in master.columns else len(master)
    quality_lines = [
        f"# Data quality ({ctx.asof})",
        "",
        f"- Universe rows: {len(master)}",
        f"- Price coverage (adj_close): {cov:.1%}",
        f"- Rows missing price: {missing_count}",
        f"- Rows without yahoo_ticker in master: {no_yahoo}",
        f"- Dividend snapshot used (<= asof): {raw_snapshot_dir.name if raw_snapshot_dir is not None else 'none'}",
        "",
        "## Technical fail reasons (top)",
    ]
    reason_counts = master.get("reason_technical_fail", pd.Series(dtype=str)).value_counts(dropna=False)
    if len(reason_counts) == 0:
        quality_lines.append("- none")
    else:
        for reason, count in reason_counts.head(10).items():
            r = reason if isinstance(reason, str) and reason else "ok"
            quality_lines.append(f"- {r}: {int(count)}")

    quality_lines.append("")
    quality_lines.append("## Dividend Feature Coverage")
    if dividend_coverage:
        for col, share in dividend_coverage.items():
            quality_lines.append(f"- {col}: {share:.1%}")
    else:
        quality_lines.append("- none")

    quality_lines.append("")
    quality_lines.append("## Financials Enrichment Coverage")
    if not financials_info["enabled"]:
        quality_lines.append("- disabled")
    elif not financials_info["core_coverage"] and not financials_info["enrich_coverage"]:
        quality_lines.append("- no snapshot merged")
    else:
        quality_lines.append(f"- matched rows in master: {financials_info['matched_rows']}")
        quality_lines.append(f"- matched from financials_agent: {financials_info['matched_agent_rows']}")
        quality_lines.append(f"- matched from reports_y bridge: {financials_info['matched_reports_y_rows']}")
        if financials_info["snapshot_date"]:
            quality_lines.append(f"- latest financials_period_end: {financials_info['snapshot_date']}")
        if financials_info["core_coverage"]:
            quality_lines.append("- core fields:")
            for col, share in financials_info["core_coverage"].items():
                quality_lines.append(f"  - {col}: {share:.1%}")
        if financials_info["enrich_coverage"]:
            quality_lines.append("- enrichment fields:")
            for col, share in financials_info["enrich_coverage"].items():
                quality_lines.append(f"  - {col}: {share:.1%}")

    quality_lines.append("")
    quality_lines.append("## Financials Anomaly Guard")
    if not financials_info["enabled"]:
        quality_lines.append("- skipped (financials enrichment disabled)")
    elif not financials_info["guard_enabled"]:
        quality_lines.append("- disabled")
    else:
        quality_lines.append(f"- compared rows: {financials_info['guard_compared_rows']}")
        quality_lines.append(f"- flagged rows: {financials_info['guard_flagged_rows']}")
        quality_lines.append(f"- excluded rows from reports_y enrichment: {financials_info['guard_excluded_rows']}")
        if financials_info["guard_previous_snapshot"] and financials_info["guard_current_snapshot"]:
            quality_lines.append(
                f"- compared snapshots: {financials_info['guard_previous_snapshot']} -> {financials_info['guard_current_snapshot']}"
            )
        reason_counts = master.get("reason_financials_guard", pd.Series(dtype=str))
        reason_counts = reason_counts[reason_counts.astype(str).str.strip().ne("")]
        if len(reason_counts) == 0:
            quality_lines.append("- reasons: none")
        else:
            quality_lines.append("- reasons:")
            for reason, count in reason_counts.value_counts().head(5).items():
                quality_lines.append(f"  - {reason}: {int(count)}")

    quality_lines.append("")
    quality_lines.append("## Financials Tier2/Tier3 Derived Coverage")
    if not tier23_coverage:
        quality_lines.append("- none")
    else:
        for col, share in tier23_coverage.items():
            quality_lines.append(f"- {col}: {share:.1%}")

    _quality_write(ctx.run_dir, quality_lines)

    out = processed_dir / "master.parquet"
    master.to_parquet(out, index=False)
    log.info(f"MASTER: wrote {out} (from master_input + prices snapshot)")
    return 0
