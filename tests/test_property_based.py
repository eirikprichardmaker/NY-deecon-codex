"""
Fase 6.3: Property-based tester med Hypothesis.

Egenskap som testes:
  - Veto-logikk kan aldri bevege seg MOT kjøp (kun mot CASH eller REVIEW)
  - DQ-regler er monotone: legger man til FAIL-data, kan ikke FAIL-count gå ned
  - Sanitizer er idempotent: sanitize(sanitize(x)) == sanitize(x)
  - VetoAction inneholder aldri BUY
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategi-hjelpere
# ---------------------------------------------------------------------------

_finite_float = st.floats(allow_nan=False, allow_infinity=False)
_pos_float = st.floats(min_value=0.001, max_value=1e9, allow_nan=False, allow_infinity=False)
_pct = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _base_row_strategy():
    """Strategi for en gyldig ticker-rad (passerer alle FAIL-regler)."""
    return st.fixed_dictionaries({
        "ticker": st.just("TEST"),
        "adj_close": _pos_float,
        "market_cap": _pos_float,
        "enterprise_value": _pos_float,
        "ma200": _pos_float,
        "index_ma200": _pos_float,
        "intrinsic_value": _finite_float,
        "sector_group": st.sampled_from(["industrial", "bank", "aquaculture", "tech"]),
    })


# ---------------------------------------------------------------------------
# Property 1: VetoAction inneholder aldri BUY
# ---------------------------------------------------------------------------

def test_veto_action_never_contains_buy():
    """Strukturell garanti: ingen BUY-verdi i VetoAction."""
    from src.agents.schemas import VetoAction
    values = {v.value for v in VetoAction}
    assert "BUY" not in values
    assert "STRONG_BUY" not in values
    assert "RECOMMEND_BUY" not in values


# ---------------------------------------------------------------------------
# Property 2: Deterministisk pre-sjekk kan kun bevege seg mot CASH
# ---------------------------------------------------------------------------

@given(
    terminal_value_pct=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    wacc_plus_mos=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    dq_warns=st.integers(min_value=0, max_value=10),
    quality_weak=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=200, deadline=None)
def test_veto_only_moves_toward_cash(
    terminal_value_pct: float,
    wacc_plus_mos: float,
    dq_warns: int,
    quality_weak: int,
) -> None:
    """
    Egenskap: deterministisk pre-sjekk kan kun produsere PASS eller VETO_CASH.
    Aldri noe som er MER positivt enn PASS.
    """
    from src.agents.investment_skeptic import _deterministic_pre_check
    from src.agents.schemas import SkepticInput, VetoAction

    inp = SkepticInput(
        ticker="TEST",
        market="OSE",
        asof_date="2026-03-11",
        intrinsic_value=100.0,
        current_price=80.0,
        mos=0.20,
        wacc_used=0.09,
        terminal_growth=0.02,
        terminal_value_pct=terminal_value_pct,
        sensitivity_wacc_plus_1pct_mos=wacc_plus_mos,
        dq_warn_count=dq_warns,
        dq_warn_rules=";".join([f"DQ-W00{i}" for i in range(min(dq_warns, 5))]),
        quality_weak_count=quality_weak,
    )

    result = _deterministic_pre_check(inp)

    if result is not None:
        # Kan KUN returnere VETO_CASH, aldri PASS eller noe mer positivt
        assert result.veto == VetoAction.VETO_CASH, (
            f"Pre-check returnerte {result.veto}, forventet VETO_CASH"
        )
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Property 3: DQ-regler er monotone (ekstra FAIL-data øker aldri fail_count ned)
# ---------------------------------------------------------------------------

@given(
    adj_close=st.one_of(st.just(float("nan")), st.floats(min_value=-100, max_value=1000, allow_nan=False)),
    market_cap=st.one_of(st.just(float("nan")), st.floats(min_value=-1e6, max_value=1e9, allow_nan=False)),
)
@settings(max_examples=100)
def test_dq_fail_count_monotone_on_bad_data(adj_close: float, market_cap: float) -> None:
    """
    Egenskap: en rad med dårlige data skal ikke ha lavere fail_count enn
    en rad der vi eksplisitt setter de dårlige feltene tilbake til OK.
    """
    from src.data_quality.rules import run_dq_rules

    bad_row = {
        "ticker": "TEST",
        "adj_close": adj_close,
        "market_cap": market_cap,
        "enterprise_value": 1000.0,
        "ma200": 90.0,
        "index_ma200": 90.0,
        "intrinsic_value": 100.0,
        "sector_group": "industrial",
    }
    good_row = dict(bad_row)
    good_row["adj_close"] = 100.0
    good_row["market_cap"] = 500_000.0

    bad_df = pd.DataFrame([bad_row])
    good_df = pd.DataFrame([good_row])

    bad_flags, _ = run_dq_rules(bad_df)
    good_flags, _ = run_dq_rules(good_df)

    bad_fail = int(bad_flags["dq_fail_count_v2"].iloc[0])
    good_fail = int(good_flags["dq_fail_count_v2"].iloc[0])

    # En rad med potensielt dårlige data kan ha >= like mange feil som god rad
    assert bad_fail >= good_fail or good_fail == 0, (
        f"God rad har {good_fail} feil, dårlig rad har {bad_fail} — monotonitet brutt"
    )


# ---------------------------------------------------------------------------
# Property 4: Sanitizer er idempotent
# ---------------------------------------------------------------------------

@given(text=st.text(max_size=500))
@settings(max_examples=200)
def test_sanitizer_idempotent(text: str) -> None:
    """sanitize(sanitize(x)) == sanitize(x) — ingen dobbel-behandling endrer output."""
    from src.agents.sanitizer import sanitize_text

    once, _ = sanitize_text(text, max_chars=200)
    twice, _ = sanitize_text(once, max_chars=200)
    assert once == twice, f"Sanitizer er ikke idempotent:\n  1: {repr(once)}\n  2: {repr(twice)}"


# ---------------------------------------------------------------------------
# Property 5: run_dq_rules returnerer alltid de forventede kolonnene
# ---------------------------------------------------------------------------

@given(
    n_rows=st.integers(min_value=1, max_value=20),
    adj_close=st.lists(
        st.one_of(st.just(float("nan")), _pos_float),
        min_size=1, max_size=20,
    ),
)
@settings(max_examples=50)
def test_run_dq_rules_always_returns_expected_columns(n_rows: int, adj_close: list) -> None:
    """run_dq_rules returnerer alltid de samme kolonnenavn uavhengig av input."""
    from src.data_quality.rules import run_dq_rules

    rows = [{"ticker": f"T{i}", "adj_close": adj_close[i % len(adj_close)]} for i in range(n_rows)]
    df = pd.DataFrame(rows)
    flags, audit = run_dq_rules(df)

    expected_cols = {"dq_fail_v2", "dq_fail_count_v2", "dq_warn_count_v2", "dq_fail_rules", "dq_warn_rules"}
    assert expected_cols.issubset(set(flags.columns)), (
        f"Manglende kolonner: {expected_cols - set(flags.columns)}"
    )
    assert len(flags) == len(df)
