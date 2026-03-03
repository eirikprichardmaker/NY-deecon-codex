from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _run_data_quality_checks


def _base_row(ticker: str = "AAA.OL") -> dict:
    return {
        "ticker": ticker,
        "ins_id": "1001",
        "sector": "Industrials",
        "date": "2026-02-16",
        "intrinsic_value": 100.0,
        "market_cap": 50.0,
        "mos": 1.0,
        "mos_req": 0.2,
        "quality_score": 0.3,
        "roic_wacc_spread": 0.02,
        "adj_close": 10.0,
        "ma200": 9.0,
        "mad": 0.1,
        "index_price": 120.0,
        "index_ma200": 110.0,
        "fcf_yield": 0.1,
        "roic": 0.12,
        "operating_margin": 0.15,
    }


def test_missing_critical_field_is_fail_and_blocked():
    row = _base_row()
    row["adj_close"] = None
    df = pd.DataFrame([row])

    flags, audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")

    assert bool(flags.loc[0, "dq_blocked"]) is True
    assert int(flags.loc[0, "dq_fail_count"]) >= 1
    assert "DQ_CRITICAL_PRESENT_adj_close" in str(flags.loc[0, "dq_fail_reasons"])
    assert (audit["severity"] == "FAIL").any()


def test_impossible_negative_values_are_fail():
    row = _base_row()
    row["market_cap"] = -1.0
    row["shares_outstanding"] = 0
    df = pd.DataFrame([row])

    flags, audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")

    assert bool(flags.loc[0, "dq_blocked"]) is True
    assert {"DQ_MARKET_CAP_NON_POSITIVE", "DQ_SHARES_NON_POSITIVE"}.issubset(set(audit["rule_id"].tolist()))


def test_sector_outlier_is_warn_only_not_blocked():
    rows = []
    for i in range(11):
        row = _base_row(f"I{i}.OL")
        row["fcf_yield"] = 0.08 + i * 0.001
        rows.append(row)
    outlier = _base_row("OUT.OL")
    outlier["fcf_yield"] = 5.0
    rows.append(outlier)
    df = pd.DataFrame(rows)

    flags, audit = _run_data_quality_checks(
        df,
        {"data_quality": {"outlier_min_samples": 10, "outlier_mad_threshold": 4.0}},
        asof="2026-02-16",
    )

    out_idx = df.index[df["ticker"] == "OUT.OL"][0]
    assert bool(flags.loc[out_idx, "dq_blocked"]) is False
    assert int(flags.loc[out_idx, "dq_warn_count"]) >= 1
    assert (audit["rule_id"] == "DQ_OUTLIER_ROBUST_fcf_yield").any()


def test_low_sample_size_sector_warns_and_logs_group_n():
    rows = []
    for i in range(3):
        row = _base_row(f"L{i}.OL")
        row["sector"] = "TinySector"
        row["fcf_yield"] = 0.1 + i * 0.01
        rows.append(row)
    df = pd.DataFrame(rows)

    flags, audit = _run_data_quality_checks(
        df,
        {"data_quality": {"outlier_min_samples": 5}},
        asof="2026-02-16",
    )

    assert (flags["dq_warn_count"] > 0).all()
    low_sample = audit[audit["rule_id"] == "DQ_OUTLIER_LOW_SAMPLE_fcf_yield"]
    assert not low_sample.empty
    assert (low_sample["group_n"] == 3).all()
