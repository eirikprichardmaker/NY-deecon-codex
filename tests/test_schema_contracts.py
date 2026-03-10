"""
Tests for src/data_quality/schema_contracts.py
Kontraktstester for Pandera DataFrameModel-schemas.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data_quality.schema_contracts import (
    DecisionOutputSchema,
    MasterSchema,
    ValuationOutputSchema,
)


# ---------------------------------------------------------------------------
# MasterSchema
# ---------------------------------------------------------------------------


def _valid_master(**overrides) -> pd.DataFrame:
    base = {
        "ticker": ["EQNR", "DNB"],
        "yahoo_ticker": ["EQNR.OL", "DNB.OL"],
        "market_cap": [500_000.0, 200_000.0],
        "adj_close": [280.0, 180.0],
        "shares_outstanding": [3_000.0, 1_500.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_master_schema_accepts_valid_data():
    df = _valid_master()
    validated = MasterSchema.validate(df)
    assert len(validated) == 2


def test_master_schema_rejects_duplicate_yahoo_ticker():
    df = _valid_master(yahoo_ticker=["X.OL", "X.OL"])
    with pytest.raises(pa.errors.SchemaError):
        MasterSchema.validate(df)


def test_master_schema_rejects_negative_market_cap():
    df = _valid_master(market_cap=[-100.0, 200_000.0])
    with pytest.raises(pa.errors.SchemaError):
        MasterSchema.validate(df)


def test_master_schema_rejects_zero_adj_close():
    df = _valid_master(adj_close=[0.0, 180.0])
    with pytest.raises(pa.errors.SchemaError):
        MasterSchema.validate(df)


def test_master_schema_rejects_invalid_ticker_format():
    """Ticker med små bokstaver skal avvises."""
    df = _valid_master(ticker=["eqnr", "DNB"])
    with pytest.raises(pa.errors.SchemaError):
        MasterSchema.validate(df)


def test_master_schema_allows_null_market_cap():
    """market_cap er nullable — NaN er OK."""
    df = _valid_master(market_cap=[float("nan"), 200_000.0])
    validated = MasterSchema.validate(df)
    assert len(validated) == 2


def test_master_schema_allows_extra_columns():
    """strict=False: ekstra kolonner skal ikke feile."""
    df = _valid_master()
    df["extra_col"] = "ignored"
    validated = MasterSchema.validate(df)
    assert "extra_col" in validated.columns


# ---------------------------------------------------------------------------
# ValuationOutputSchema
# ---------------------------------------------------------------------------


def _valid_valuation(**overrides) -> pd.DataFrame:
    base = {
        "ticker": ["EQNR.OL", "DNB.OL"],
        "intrinsic_value": [320.0, 210.0],
        "wacc_used": [0.09, 0.10],
        "terminal_growth": [0.02, 0.015],
        "mos": [0.125, 0.143],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_valuation_schema_accepts_valid_data():
    df = _valid_valuation()
    validated = ValuationOutputSchema.validate(df)
    assert len(validated) == 2


def test_valuation_schema_rejects_wacc_above_30pct():
    df = _valid_valuation(wacc_used=[0.35, 0.10])
    with pytest.raises(pa.errors.SchemaError):
        ValuationOutputSchema.validate(df)


def test_valuation_schema_rejects_terminal_growth_above_2pct():
    df = _valid_valuation(terminal_growth=[0.03, 0.015])
    with pytest.raises(pa.errors.SchemaError):
        ValuationOutputSchema.validate(df)


def test_valuation_schema_rejects_mos_above_1():
    df = _valid_valuation(mos=[1.5, 0.10])
    with pytest.raises(pa.errors.SchemaError):
        ValuationOutputSchema.validate(df)


def test_valuation_schema_rejects_mos_below_minus1():
    df = _valid_valuation(mos=[-1.5, 0.10])
    with pytest.raises(pa.errors.SchemaError):
        ValuationOutputSchema.validate(df)


def test_valuation_schema_allows_null_intrinsic_value():
    df = _valid_valuation(intrinsic_value=[float("nan"), 210.0])
    validated = ValuationOutputSchema.validate(df)
    assert len(validated) == 2


# ---------------------------------------------------------------------------
# DecisionOutputSchema
# ---------------------------------------------------------------------------


def _valid_decision(**overrides) -> pd.DataFrame:
    base = {
        "ticker": ["EQNR.OL"],
        "decision": ["KANDIDAT"],
        "reason": ["MOS > 15%, technical OK"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_decision_schema_accepts_kandidat():
    df = _valid_decision(decision=["KANDIDAT"])
    validated = DecisionOutputSchema.validate(df)
    assert validated["decision"].iloc[0] == "KANDIDAT"


def test_decision_schema_accepts_cash():
    df = _valid_decision(decision=["CASH"])
    validated = DecisionOutputSchema.validate(df)
    assert validated["decision"].iloc[0] == "CASH"


def test_decision_schema_accepts_manual_review():
    df = _valid_decision(decision=["MANUAL_REVIEW"])
    validated = DecisionOutputSchema.validate(df)
    assert validated["decision"].iloc[0] == "MANUAL_REVIEW"


def test_decision_schema_rejects_unknown_decision():
    """BUY eller andre verdier skal avvises."""
    df = _valid_decision(decision=["BUY"])
    with pytest.raises(pa.errors.SchemaError):
        DecisionOutputSchema.validate(df)


def test_decision_schema_rejects_null_ticker():
    df = _valid_decision(ticker=[None])
    with pytest.raises(pa.errors.SchemaError):
        DecisionOutputSchema.validate(df)
