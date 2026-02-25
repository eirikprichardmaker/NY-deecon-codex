from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _build_quality_block, _dividend_quality_score


def test_dividend_quality_score_all_criteria_met():
    df = pd.DataFrame(
        {
            "dividend_streak_years": [12],
            "dividend_growth_5y": [0.04],
            "payout_ratio": [0.55],
            "share_count_growth_5y": [-0.01],
            "market_cap": [5_000_000_000.0],
            "fcf_margin": [0.12],
            "ocf_margin": [0.14],
            "profit_margin": [0.10],
            "roa": [0.08],
            "dividend_yield": [0.03],
            "p_e_current": [18.0],
        }
    )

    out = _dividend_quality_score(df)

    assert int(out.loc[0, "dividend_score"]) == 11
    assert int(out.loc[0, "dividend_criteria_available_count"]) == 11
    assert int(out.loc[0, "dividend_criteria_missing_count"]) == 0
    assert str(out.loc[0, "dividend_reason"]) == ""


def test_dividend_quality_score_handles_missing_fields_with_reasons():
    df = pd.DataFrame(
        {
            "market_cap": [500_000_000.0],
            "p_e_current": [40.0],
        }
    )

    out = _dividend_quality_score(df)

    assert int(out.loc[0, "dividend_score"]) == 0
    assert int(out.loc[0, "dividend_criteria_available_count"]) == 2
    assert int(out.loc[0, "dividend_criteria_missing_count"]) == 9
    reasons = str(out.loc[0, "dividend_reason"])
    assert "market_value_fail" in reasons
    assert "pe_fail" in reasons
    assert "yield_missing" in reasons


def test_build_quality_block_uses_dividend_strategy_when_requested():
    df = pd.DataFrame(
        {
            "market_cap": [5_000_000_000.0],
            "p_e_current": [15.0],
        }
    )

    out = _build_quality_block(df, {"quality_strategy": "dividend_quality"})

    assert str(out.loc[0, "quality_strategy"]) == "dividend_quality"
    assert float(out.loc[0, "quality_score"]) == float(out.loc[0, "dividend_score"])
