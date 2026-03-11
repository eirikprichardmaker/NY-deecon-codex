"""
Tester for src/agents/data_controller.py

Sjekkliste (per guiden):
  - En aksje med alle kritiske felt OK → ok=True
  - En aksje med manglende intrinsic_value → ok=False, felt i missing_fields
  - En aksje med adj_close=0 → ok=False (ikke > 0)
  - En aksje med adj_close=negativ → ok=False
  - En aksje med wacc_used=0.70 (>0.60) → ok=False (utenfor grenser)
  - valuation_row-felter brukes når ticker_row mangler dem
  - ticker_row har forrang over valuation_row
  - check_shortlist filtrerer korrekt: returnerer kun godkjente rader
  - Blokkert ticker gir maskinlesbar diagnostics-dict
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import pandas as pd
import pytest

from src.agents.data_controller import ControllerResult, check, check_shortlist


# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def _good_row(**overrides) -> dict:
    """Minimum gyldig rad som passerer alle kontroller."""
    base = {
        "ticker": "EQNR.OL",
        "adj_close": 280.0,
        "market_cap": 500_000.0,
        "intrinsic_value": 350.0,
        "wacc_used": 0.09,
        "mos": 0.20,
        "roic_current": 0.15,
        "fcf_yield": 0.06,
        "nd_ebitda": 1.5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Positive tester
# ---------------------------------------------------------------------------

def test_good_row_passes():
    """Gyldig rad med alle felt OK → ok=True, ingen blokkering."""
    result = check(_good_row())
    assert result.ok is True
    assert result.missing_fields == []
    assert result.out_of_range_fields == []
    assert result.blocked_reason == ""


def test_valuation_row_used_as_fallback():
    """Felter fra valuation_row brukes hvis de mangler i ticker_row."""
    ticker_row = {"ticker": "DNB.OL", "adj_close": 180.0, "market_cap": 200_000.0, "mos": 0.15}
    val_row = {"intrinsic_value": 220.0, "wacc_used": 0.08}
    result = check(ticker_row, val_row)
    assert result.ok is True


def test_ticker_row_overrides_valuation_row():
    """ticker_row-verdier har forrang over valuation_row."""
    ticker_row = _good_row(intrinsic_value=400.0)
    val_row = {"intrinsic_value": -999.0}  # Ugyldig, men skal overstyres
    result = check(ticker_row, val_row)
    assert result.ok is True
    assert result.diagnostics.get("intrinsic_value", {}).get("status") != "out_of_range"


# ---------------------------------------------------------------------------
# Manglende felt
# ---------------------------------------------------------------------------

def test_missing_intrinsic_value_blocks():
    """Manglende intrinsic_value → blokkert."""
    row = _good_row()
    del row["intrinsic_value"]
    result = check(row)
    assert result.ok is False
    assert "intrinsic_value" in result.missing_fields


def test_nan_adj_close_blocks():
    """adj_close=NaN → blokkert som manglende felt."""
    result = check(_good_row(adj_close=float("nan")))
    assert result.ok is False
    assert "adj_close" in result.missing_fields


def test_none_mos_blocks():
    """mos=None → blokkert."""
    result = check(_good_row(mos=None))
    assert result.ok is False
    assert "mos" in result.missing_fields


# ---------------------------------------------------------------------------
# Harde grenser
# ---------------------------------------------------------------------------

def test_adj_close_zero_blocks():
    """adj_close=0 → utenfor grenser (krever > 0)."""
    result = check(_good_row(adj_close=0.0))
    assert result.ok is False
    assert "adj_close" in result.out_of_range_fields


def test_adj_close_negative_blocks():
    """adj_close negativ → blokkert."""
    result = check(_good_row(adj_close=-10.0))
    assert result.ok is False
    assert "adj_close" in result.out_of_range_fields


def test_wacc_too_high_blocks():
    """wacc_used=0.70 (>0.60 maks) → blokkert."""
    result = check(_good_row(wacc_used=0.70))
    assert result.ok is False
    assert "wacc_used" in result.out_of_range_fields


def test_wacc_zero_blocks():
    """wacc_used=0 → blokkert (grense er > 0)."""
    result = check(_good_row(wacc_used=0.0))
    assert result.ok is False
    assert "wacc_used" in result.out_of_range_fields


def test_mos_extreme_blocks():
    """mos=55 (>50 maks) → blokkert."""
    result = check(_good_row(mos=55.0))
    assert result.ok is False
    assert "mos" in result.out_of_range_fields


# ---------------------------------------------------------------------------
# Diagnostikk
# ---------------------------------------------------------------------------

def test_blocked_result_has_readable_reason():
    """Blokkert resultat har lesbar blocked_reason-streng."""
    result = check(_good_row(adj_close=float("nan"), wacc_used=0.99))
    assert result.ok is False
    assert "adj_close" in result.blocked_reason or "wacc_used" in result.blocked_reason
    assert len(result.blocked_reason) > 0


def test_blocked_result_diagnostics_has_detail():
    """Diagnostics-dict har status og verdi for blokkerte felter."""
    result = check(_good_row(adj_close=-5.0))
    assert "adj_close" in result.diagnostics
    diag = result.diagnostics["adj_close"]
    assert "status" in diag
    assert diag["status"] in ("missing", "out_of_range", "not_numeric")


def test_to_dict_is_serializable():
    """ControllerResult.to_dict() kan serialiseres til JSON."""
    import json
    result = check(_good_row(adj_close=float("nan")))
    d = result.to_dict()
    json_str = json.dumps(d)  # skal ikke kaste
    loaded = json.loads(json_str)
    assert loaded["ticker"] == "EQNR.OL"
    assert loaded["ok"] is False


# ---------------------------------------------------------------------------
# check_shortlist
# ---------------------------------------------------------------------------

def test_check_shortlist_filters_invalid_rows():
    """check_shortlist returnerer kun rader som passerer."""
    df = pd.DataFrame([
        _good_row(ticker="EQNR.OL"),
        _good_row(ticker="DNB.OL", adj_close=float("nan")),   # blokkert
        _good_row(ticker="ORK.OL", wacc_used=0.0),             # blokkert
        _good_row(ticker="YAR.OL"),
    ])
    approved, results = check_shortlist(df, valuation_df=None)

    approved_tickers = set(approved["ticker"].tolist())
    assert "EQNR.OL" in approved_tickers
    assert "YAR.OL" in approved_tickers
    assert "DNB.OL" not in approved_tickers
    assert "ORK.OL" not in approved_tickers
    assert len(results) == 4


def test_check_shortlist_all_ok():
    """Alle rader gyldige → approved_df lik shortlist_df."""
    df = pd.DataFrame([_good_row(ticker=f"T{i}.OL") for i in range(3)])
    approved, results = check_shortlist(df, None)
    assert len(approved) == 3
    assert all(r.ok for r in results)


def test_check_shortlist_all_blocked():
    """Alle rader ugyldige → tom approved_df."""
    df = pd.DataFrame([
        _good_row(ticker="A.OL", adj_close=float("nan")),
        _good_row(ticker="B.OL", intrinsic_value=float("nan")),
    ])
    approved, results = check_shortlist(df, None)
    assert approved.empty
    assert all(not r.ok for r in results)


def test_check_shortlist_uses_valuation_df():
    """Felter fra valuation_df brukes som fallback for manglende felter i shortlist."""
    shortlist = pd.DataFrame([{"ticker": "X.OL", "adj_close": 100.0, "market_cap": 1_000.0, "mos": 0.20}])
    valuation = pd.DataFrame([{"ticker": "X.OL", "intrinsic_value": 130.0, "wacc_used": 0.09}])
    approved, results = check_shortlist(shortlist, valuation)
    assert len(approved) == 1
    assert results[0].ok is True
