from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import wft


def _base_month_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-01-31")],
            "ticker": ["AAA.OL", "BBB.OL"],
            "k": ["AAA", "BBB"],
            "market": ["NO", "NO"],
            "mos": [0.55, 0.60],
            "high_risk_flag": [False, False],
            "value_creation_ok_base": [True, True],
            "quality_weak_count": [0, 0],
            "adj_close": [100.0, 110.0],
            "ma200": [90.0, 100.0],
            "mad": [0.02, 0.03],
            "index_price": [1000.0, 1000.0],
            "index_ma200": [900.0, 900.0],
            "index_mad": [0.03, 0.03],
            "roic": [0.1, 0.11],
            "fcf_yield": [0.05, 0.06],
            "market_cap": [1_000_000_000.0, 1_100_000_000.0],
            "intrinsic_value": [1_500_000_000.0, 1_600_000_000.0],
        }
    )


def test_wft_dq_blocked_candidate_excluded():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = _base_month_df()
    month_df.loc[0, "adj_close"] = 0.0

    out = wft._apply_filters(month_df, params)

    assert bool(out.loc[out["k"] == "AAA", "dq_blocked"].iloc[0]) is True
    assert bool(out.loc[out["k"] == "AAA", "eligible"].iloc[0]) is False


def test_wft_warn_does_not_exclude_candidate():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = _base_month_df()
    month_df.loc[1, "fcf_yield"] = 9.0

    out = wft._apply_filters(month_df, params)

    assert bool(out.loc[out["k"] == "BBB", "dq_blocked"].iloc[0]) is False
    assert int(out.loc[out["k"] == "BBB", "dq_warn_count"].iloc[0]) >= 1
    assert bool(out.loc[out["k"] == "BBB", "eligible"].iloc[0]) is True
