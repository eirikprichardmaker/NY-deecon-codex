from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


PRICE_ALIASES = ["adj_close", "close", "price", "last"]
MARKET_CAP_ALIASES = ["market_cap_current", "market_cap"]
SHARES_ALIASES = ["shares_outstanding"]
INTRINSIC_ALIASES = ["intrinsic_value", "intrinsic_equity"]
MKT_ALIASES = ["market", "country_code"]
DEFAULT_WARN_METRICS = ["fcf_yield", "roic", "roe"]
SUFFIX_TO_MARKET = {"OL": "NO", "ST": "SE", "CO": "DK", "HE": "FI"}


@dataclass(frozen=True)
class DQConfig:
    dq_min_group_n: int = 8
    dq_outlier_method: str = "IQR"
    dq_iqr_multiplier: float = 3.0
    dq_warn_metrics: tuple[str, ...] = ("fcf_yield", "roic", "roe")


def dq_config_from_cfg(cfg: dict | None) -> DQConfig:
    raw = (cfg or {}).get("data_quality", {}) or {}
    if "decision" in (cfg or {}):
        d = (cfg or {}).get("decision", {}) or {}
        raw = {**raw, **(d.get("data_quality", {}) or {})}

    metrics_raw = raw.get("dq_warn_metrics", DEFAULT_WARN_METRICS)
    if isinstance(metrics_raw, str):
        metrics = tuple([x.strip() for x in metrics_raw.split(",") if x.strip()])
    else:
        metrics = tuple(str(x).strip() for x in metrics_raw if str(x).strip())
    if not metrics:
        metrics = tuple(DEFAULT_WARN_METRICS)

    return DQConfig(
        dq_min_group_n=int(raw.get("dq_min_group_n", 8)),
        dq_outlier_method=str(raw.get("dq_outlier_method", "IQR")).upper(),
        dq_iqr_multiplier=float(raw.get("dq_iqr_multiplier", 3.0)),
        dq_warn_metrics=metrics,
    )


def _pick_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _derive_market(df: pd.DataFrame) -> pd.Series:
    market_col = _pick_col(df, MKT_ALIASES)
    if market_col:
        m = df[market_col].astype(str).str.strip().replace({"": "ALL", "nan": "ALL", "None": "ALL"})
        return m.str.upper()

    ticker_col = _pick_col(df, ["yahoo_ticker", "ticker"])
    if not ticker_col:
        return pd.Series("ALL", index=df.index)

    suffix = (
        df[ticker_col]
        .astype(str)
        .str.upper()
        .str.extract(r"\.([A-Z0-9]+)$", expand=False)
        .fillna("")
    )
    return suffix.map(lambda x: SUFFIX_TO_MARKET.get(x, "ALL"))


def _assign_size_bucket(df: pd.DataFrame, market: pd.Series) -> tuple[pd.Series, pd.Series, str]:
    mcap_col = _pick_col(df, MARKET_CAP_ALIASES)
    mcap = _to_num(df[mcap_col]) if mcap_col else pd.Series(np.nan, index=df.index)
    bucket = pd.Series("U", index=df.index, dtype=object)
    group_n = pd.Series(0, index=df.index, dtype=int)
    method = "quantile_q5"

    for mk, idx in market.groupby(market).groups.items():
        vals = mcap.loc[idx]
        valid = vals[np.isfinite(vals)]
        group_n.loc[idx] = int(valid.shape[0])
        if valid.shape[0] >= 5:
            ranks = vals.rank(method="first", pct=True)
            qbin = np.ceil(ranks * 5).clip(1, 5)
            bucket.loc[idx] = qbin.map(lambda x: f"Q{int(x)}" if np.isfinite(x) else "U")
        else:
            method = "rank_split_3"
            ranks = vals.rank(method="first", pct=True)
            bucket.loc[idx] = np.select(
                [ranks <= 1 / 3, ranks <= 2 / 3, ranks > 2 / 3],
                ["S", "M", "L"],
                default="U",
            )

    return bucket.astype(str), group_n.astype(int), method


def _robust_bounds(s: pd.Series, method: str, iqr_multiplier: float) -> tuple[float, float] | None:
    x = _to_num(s)
    x = x[np.isfinite(x)]
    if x.empty:
        return None
    method = method.upper()
    if method == "MAD":
        med = float(x.median())
        mad = float((x - med).abs().median())
        if not np.isfinite(mad) or mad <= 0:
            return None
        scale = 1.4826 * mad
        return med - iqr_multiplier * scale, med + iqr_multiplier * scale

    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return q1, q3
    return q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr


def apply_data_quality(
    df: pd.DataFrame,
    cfg: DQConfig,
    asof: str,
    require_intrinsic: bool = True,
    context: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = context or {}
    d = df.copy()
    market = _derive_market(d)
    size_bucket, group_n, bucket_method = _assign_size_bucket(d, market)
    group_key = market.astype(str) + ":" + size_bucket.astype(str)

    events: list[dict[str, Any]] = []
    price_col = _pick_col(d, PRICE_ALIASES)
    mcap_col = _pick_col(d, MARKET_CAP_ALIASES)
    shares_col = _pick_col(d, SHARES_ALIASES)
    intrinsic_col = _pick_col(d, INTRINSIC_ALIASES)
    mos_col = _pick_col(d, ["mos"])

    def add_event(i: Any, rule_id: str, severity: str, field: str, value: Any, reason: str) -> None:
        row = d.loc[i]
        payload = {
            "row_index": i,
            "ticker": str(row.get("ticker", row.get("yahoo_ticker", ""))),
            "asof": asof,
            "rule_id": rule_id,
            "severity": severity,
            "field": field,
            "value": value,
            "reason": reason,
            "group_key": str(group_key.loc[i]),
            "group_n": int(group_n.loc[i]),
            "size_bucket": str(size_bucket.loc[i]),
            "bucket_method": bucket_method,
        }
        payload.update(context)
        events.append(payload)

    # FAIL rules
    if price_col:
        price = _to_num(d[price_col])
        for i in price.index[price <= 0]:
            add_event(i, "DQ_PRICE_NON_POSITIVE", "FAIL", price_col, price.loc[i], f"{price_col}<=0")

    if mcap_col:
        mcap = _to_num(d[mcap_col])
        for i in mcap.index[mcap <= 0]:
            add_event(i, "DQ_MARKET_CAP_NON_POSITIVE", "FAIL", mcap_col, mcap.loc[i], f"{mcap_col}<=0")

    if shares_col:
        shares = _to_num(d[shares_col])
        for i in shares.index[shares <= 0]:
            add_event(i, "DQ_SHARES_NON_POSITIVE", "FAIL", shares_col, shares.loc[i], f"{shares_col}<=0")

    if require_intrinsic:
        if intrinsic_col:
            intrinsic = _to_num(d[intrinsic_col])
            for i in intrinsic.index[~np.isfinite(intrinsic)]:
                add_event(i, "DQ_INTRINSIC_MISSING", "FAIL", intrinsic_col, d.loc[i, intrinsic_col], "intrinsic_missing")

        if mos_col and price_col and intrinsic_col:
            mos = _to_num(d[mos_col])
            intrinsic = _to_num(d[intrinsic_col])
            price = _to_num(d[price_col])
            broken = mos.isna() & np.isfinite(intrinsic) & np.isfinite(price)
            for i in mos.index[broken]:
                add_event(i, "DQ_MOS_COMPUTE_MISSING", "FAIL", mos_col, d.loc[i, mos_col], "mos_nan_with_intrinsic_and_price")

    # WARN rules
    for metric in cfg.dq_warn_metrics:
        if metric not in d.columns:
            continue
        vals = _to_num(d[metric])
        for gk, idx in group_key.groupby(group_key).groups.items():
            gvals = vals.loc[idx]
            gn = int(np.isfinite(gvals).sum())
            if gn < cfg.dq_min_group_n:
                for i in idx:
                    add_event(i, f"LOW_SAMPLE_SIZE_{metric}", "WARN", metric, d.loc[i, metric], f"group_n={gn}<min={cfg.dq_min_group_n};group={gk}")
                continue

            bounds = _robust_bounds(gvals, cfg.dq_outlier_method, cfg.dq_iqr_multiplier)
            if bounds is None:
                continue
            lo, hi = bounds
            mask = (gvals < lo) | (gvals > hi)
            for i in gvals.index[mask.fillna(False)]:
                add_event(i, f"DQ_OUTLIER_{cfg.dq_outlier_method}_{metric}", "WARN", metric, gvals.loc[i], f"outside=[{lo:.6g},{hi:.6g}];group={gk}")

    audit = pd.DataFrame(events)
    flags = pd.DataFrame(index=d.index)

    if audit.empty:
        flags["dq_blocked"] = False
        flags["dq_fail_count"] = 0
        flags["dq_warn_count"] = 0
        flags["dq_fail_reasons"] = ""
        flags["dq_warn_reasons"] = ""
        flags["dq_group_key"] = group_key
        flags["dq_group_n"] = group_n
        flags["dq_bucket_method"] = bucket_method
    else:
        fail = audit[audit["severity"] == "FAIL"].groupby("row_index")
        warn = audit[audit["severity"] == "WARN"].groupby("row_index")

        flags["dq_fail_count"] = fail.size().reindex(d.index).fillna(0).astype(int)
        flags["dq_warn_count"] = warn.size().reindex(d.index).fillna(0).astype(int)
        flags["dq_blocked"] = flags["dq_fail_count"] > 0
        flags["dq_fail_reasons"] = fail["rule_id"].apply(lambda s: ";".join(sorted(set(s.astype(str))))).reindex(d.index).fillna("")
        flags["dq_warn_reasons"] = warn["rule_id"].apply(lambda s: ";".join(sorted(set(s.astype(str))))).reindex(d.index).fillna("")
        flags["dq_group_key"] = group_key
        flags["dq_group_n"] = group_n
        flags["dq_bucket_method"] = bucket_method

    # compatibility aliases
    flags["data_quality_fail"] = flags["dq_blocked"].astype(bool)
    flags["data_quality_fail_count"] = flags["dq_fail_count"].astype(int)
    flags["data_quality_warn_count"] = flags["dq_warn_count"].astype(int)
    flags["data_quality_fail_reasons"] = flags["dq_fail_reasons"]
    flags["data_quality_warn_reasons"] = flags["dq_warn_reasons"]

    return flags, audit
