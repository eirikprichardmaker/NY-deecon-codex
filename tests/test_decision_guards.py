from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _quality_gate, _to_decimal_rate, _value_creation_gate


def test_to_decimal_rate_normalizes_percent_points():
    s = pd.Series([12.0, 8.0, 5.0])
    out = _to_decimal_rate(s)
    assert abs(float(out.iloc[0]) - 0.12) < 1e-12
    assert abs(float(out.iloc[1]) - 0.08) < 1e-12


def test_value_creation_gate_requires_positive_3y_conservative_spread():
    df = pd.DataFrame(
        {
            "roic_current": [12.0, 8.0, None],
            "wacc_used": [0.08, 0.09, 0.09],
        }
    )

    out = _value_creation_gate(df, {"value_creation_spread_decay_per_year": 0.01})

    assert bool(out.loc[0, "value_creation_ok"]) is True
    assert bool(out.loc[1, "value_creation_ok"]) is False
    assert str(out.loc[2, "value_creation_reason"]) == "missing_roic_or_wacc"


def test_quality_gate_blocks_when_two_or_more_indicators_are_weak():
    df = pd.DataFrame(
        {
            "roic_current": [12.0, -1.0],
            "fcf_yield": [0.05, -0.01],
            "nd_ebitda": [2.0, 6.0],
            "ev_ebit": [10.0, 35.0],
        }
    )

    out = _quality_gate(df, {"quality_weak_fail_min": 2})

    assert bool(out.loc[0, "quality_gate_ok"]) is True
    assert bool(out.loc[1, "quality_gate_ok"]) is False
    assert int(out.loc[1, "quality_weak_count"]) >= 2
    assert str(out.loc[1, "quality_gate_reason"]) == "quality_weak_count_gte_2"
