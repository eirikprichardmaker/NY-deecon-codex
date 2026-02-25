from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _build_quality_block, _graham_strategy_score


def test_graham_strategy_score_all_criteria_met():
    df = pd.DataFrame(
        {
            "market_cap": [5_000_000_000.0],
            "nd_ebitda": [1.5],
            "net_debt": [1_000_000_000.0],
            "profit_margin": [0.08],
            "roic_current": [0.12],
            "dividend_history_years": [12.0],
            "profit_growth_5y": [0.05],
            "p_e_current": [12.0],
            "p_b_current": [1.2],
        }
    )

    out = _graham_strategy_score(df)

    assert int(out.loc[0, "graham_score"]) == 7
    assert int(out.loc[0, "graham_criteria_available_count"]) == 7
    assert int(out.loc[0, "graham_criteria_missing_count"]) == 0
    assert str(out.loc[0, "graham_reason"]) == ""


def test_graham_strategy_score_marks_missing_and_fail_reasons():
    df = pd.DataFrame(
        {
            "market_cap": [1_000_000_000.0],
            "profit_margin": [-0.01],
            "dividend_history_years": [2.0],
            "profit_growth_5y": [-0.02],
            "p_e_current": [30.0],
            "p_s_current": [3.0],
        }
    )

    out = _graham_strategy_score(df)

    assert int(out.loc[0, "graham_score"]) == 0
    assert int(out.loc[0, "graham_criteria_available_count"]) == 6
    assert int(out.loc[0, "graham_criteria_missing_count"]) == 1
    reasons = str(out.loc[0, "graham_reason"])
    assert "graham_size_fail" in reasons
    assert "graham_financial_strength_missing" in reasons
    assert "graham_moderate_pe_fail" in reasons


def test_build_quality_block_uses_graham_strategy_when_requested():
    df = pd.DataFrame(
        {
            "market_cap": [5_000_000_000.0],
            "nd_ebitda": [1.0],
            "profit_margin": [0.08],
            "dividend_history_years": [15.0],
            "profit_growth_5y": [0.03],
            "p_e_current": [12.0],
            "p_b_current": [1.0],
        }
    )

    out = _build_quality_block(df, {"quality_strategy": "graham_strategy"})

    assert str(out.loc[0, "quality_strategy"]) == "graham_strategy"
    assert float(out.loc[0, "quality_score"]) == float(out.loc[0, "graham_score"])
