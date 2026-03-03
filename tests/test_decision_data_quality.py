from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _run_data_quality_checks


def test_missing_intrinsic_is_fail_and_blocked():
    df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "adj_close": [100.0],
            "market_cap": [1_000_000_000.0],
            "intrinsic_value": [None],
            "mos": [None],
            "fcf_yield": [0.05],
            "roic": [0.10],
        }
    )

    flags, audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")

    assert bool(flags.loc[0, "dq_blocked"]) is True
    assert "DQ_INTRINSIC_MISSING" in str(flags.loc[0, "dq_fail_reasons"])
    assert (audit["severity"] == "FAIL").any()


def test_missing_intrinsic_columns_is_fail_and_blocked():
    df = pd.DataFrame(
        {
            "ticker": ["AAA.OL", "BBB.OL"],
            "adj_close": [100.0, 101.0],
            "market_cap": [1_000_000_000.0, 2_000_000_000.0],
            "mos": [None, None],
            "fcf_yield": [0.05, 0.04],
            "roic": [0.10, 0.11],
        }
    )

    flags, audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")

    assert flags["dq_blocked"].all()
    assert (audit["rule_id"] == "DQ_INTRINSIC_MISSING").sum() == len(df)
    assert (audit["reason"] == "intrinsic_column_missing").all()


def test_non_positive_price_and_market_cap_are_fail():
    df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "adj_close": [0.0],
            "market_cap": [-5.0],
            "intrinsic_value": [100.0],
            "mos": [0.3],
            "fcf_yield": [0.05],
            "roic": [0.10],
        }
    )

    flags, _audit = _run_data_quality_checks(df, {"data_quality": {}}, asof="2026-02-16")
    reasons = str(flags.loc[0, "dq_fail_reasons"])

    assert bool(flags.loc[0, "dq_blocked"]) is True
    assert "DQ_PRICE_NON_POSITIVE" in reasons
    assert "DQ_MARKET_CAP_NON_POSITIVE" in reasons


def test_extreme_outlier_warn_only():
    vals = [0.1] * 49 + [50.0]
    mcap = [1_000_000_000.0 + (i * 1_000_000.0) for i in range(50)]
    df = pd.DataFrame(
        {
            "ticker": [f"T{i}.OL" for i in range(50)],
            "adj_close": [100.0] * 50,
            "market_cap": mcap,
            "intrinsic_value": [1_200_000_000.0] * 50,
            "mos": [0.2] * 50,
            "fcf_yield": vals,
            "roic": [0.12] * 50,
            "market": ["NO"] * 50,
        }
    )

    flags, audit = _run_data_quality_checks(
        df,
        {"data_quality": {"dq_min_group_n": 5, "dq_iqr_multiplier": 1.5, "dq_warn_metrics": ["fcf_yield"]}},
        asof="2026-02-16",
    )

    assert (audit["rule_id"].astype(str).str.contains("DQ_OUTLIER_IQR_fcf_yield")).any()
    assert bool(flags.loc[49, "dq_blocked"]) is False
    assert int(flags.loc[49, "dq_warn_count"]) >= 1


def test_low_sample_size_warn_logged():
    df = pd.DataFrame(
        {
            "ticker": ["A.OL", "B.OL"],
            "adj_close": [100.0, 110.0],
            "market_cap": [1_000_000_000.0, 1_200_000_000.0],
            "intrinsic_value": [1_200_000_000.0, 1_300_000_000.0],
            "mos": [0.2, 0.2],
            "fcf_yield": [0.1, 0.2],
            "roic": [0.1, 0.11],
            "market": ["NO", "NO"],
        }
    )

    flags, audit = _run_data_quality_checks(
        df,
        {"data_quality": {"dq_min_group_n": 10, "dq_warn_metrics": ["fcf_yield"]}},
        asof="2026-02-16",
    )

    assert (audit["rule_id"] == "LOW_SAMPLE_SIZE_fcf_yield").any()
    assert int(flags["dq_warn_count"].sum()) >= 2
