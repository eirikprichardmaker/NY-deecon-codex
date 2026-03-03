from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import REASON_DATA_INVALID, _run_data_quality_checks


def test_data_quality_fail_on_missing_critical_field():
    df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "intrinsic_value": [100.0],
            "market_cap": [50.0],
            "mos": [1.0],
            "mos_req": [0.2],
            "quality_score": [0.3],
            "roic_wacc_spread": [0.02],
            "adj_close": [None],
            "ma200": [90.0],
            "mad": [0.1],
            "index_price": [120.0],
            "index_ma200": [110.0],
        }
    )

    flags, audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")

    assert bool(flags.loc[0, "data_quality_fail"]) is True
    assert "missing_or_non_finite:adj_close" in str(flags.loc[0, "data_quality_fail_reasons"])
    assert (audit["severity"] == "FAIL").any()


def test_data_quality_fail_on_hard_range_violation():
    df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "intrinsic_value": [100.0],
            "market_cap": [-5.0],
            "mos": [1.0],
            "mos_req": [0.2],
            "quality_score": [0.3],
            "roic_wacc_spread": [0.02],
            "adj_close": [10.0],
            "ma200": [9.0],
            "mad": [0.1],
            "index_price": [120.0],
            "index_ma200": [110.0],
        }
    )

    flags, audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")

    assert bool(flags.loc[0, "data_quality_fail"]) is True
    assert any(audit["rule_id"].astype(str).str.contains("DQ_RANGE_MIN_market_cap"))


def test_data_quality_outlier_warn_and_fail_thresholds():
    values = [100.0] * 19 + [200.0, 500.0]
    df = pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(len(values))],
            "intrinsic_value": [1000.0] * len(values),
            "market_cap": [1e9] * len(values),
            "mos": [0.5] * len(values),
            "mos_req": [0.2] * len(values),
            "quality_score": [0.3] * len(values),
            "roic_wacc_spread": [0.02] * len(values),
            "adj_close": values,
            "ma200": [100.0] * len(values),
            "mad": [0.01] * len(values),
            "index_price": [1000.0] * len(values),
            "index_ma200": [900.0] * len(values),
        }
    )

    cfg_warn = {"data_quality": {"outlier_min_samples": 10, "outlier_z_warn": 2.0, "outlier_z_fail": 10.0}}
    _, audit_warn = _run_data_quality_checks(df, cfg_warn, asof="2026-02-16")
    warning = audit_warn[(audit_warn["field"] == "adj_close") & (audit_warn["severity"] == "WARN")]

    cfg_fail = {"data_quality": {"outlier_min_samples": 10, "outlier_z_warn": 2.0, "outlier_z_fail": 3.0}}
    flags, audit_fail = _run_data_quality_checks(df, cfg_fail, asof="2026-02-16")
    severe = audit_fail[(audit_fail["field"] == "adj_close") & (audit_fail["severity"] == "FAIL")]

    assert not severe.empty
    assert not warning.empty
    assert REASON_DATA_INVALID == "DATA_INVALID"
    assert int(flags["data_quality_fail_count"].sum()) >= 1
