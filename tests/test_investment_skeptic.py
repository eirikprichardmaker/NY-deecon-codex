"""
Tests for src/agents/investment_skeptic.py
Fokus på deterministiske pre-check og conservatism enforcement (ingen LLM-kall).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.agents.schemas import RiskFinding, SkepticInput, SkepticOutput, VetoAction
from src.agents.investment_skeptic import (
    _deterministic_pre_check,
    _enforce_conservatism,
    run_skeptic,
)


# ---------------------------------------------------------------------------
# Hjelpefunksjoner
# ---------------------------------------------------------------------------

def _make_input(**overrides) -> SkepticInput:
    base = dict(
        ticker="EQNR.OL",
        market="OSE",
        asof_date="2026-03-10",
        intrinsic_value=300.0,
        current_price=250.0,
        mos=0.167,
        wacc_used=0.09,
        terminal_growth=0.02,
    )
    base.update(overrides)
    return SkepticInput(**base)


def _make_pass_output(ticker: str = "EQNR.OL") -> SkepticOutput:
    return SkepticOutput(
        ticker=ticker,
        veto=VetoAction.PASS,
        confidence=0.8,
        reasoning="Ingen kritiske risikoer funnet",
    )


class _FailClient:
    """Mock-klient som alltid kaster exception — simulerer API-nedetid."""
    class beta:
        class chat:
            class completions:
                @staticmethod
                def parse(**kwargs):
                    raise RuntimeError("API utilgjengelig")


# ---------------------------------------------------------------------------
# _deterministic_pre_check: terminal value dominance
# ---------------------------------------------------------------------------

def test_vetoes_on_terminal_value_dominance():
    """Fra planen: TV > 60% skal gi deterministisk VETO_CASH."""
    inp = _make_input(terminal_value_pct=0.72)
    result = _deterministic_pre_check(inp)
    assert result is not None
    assert result.veto == VetoAction.VETO_CASH


def test_no_veto_when_tv_below_threshold():
    inp = _make_input(terminal_value_pct=0.55)
    result = _deterministic_pre_check(inp)
    assert result is None


def test_no_veto_when_tv_exactly_at_threshold():
    """60% er ikke over grensen — ingen veto."""
    inp = _make_input(terminal_value_pct=0.60)
    result = _deterministic_pre_check(inp)
    assert result is None


def test_tv_veto_finding_has_correct_category():
    inp = _make_input(terminal_value_pct=0.80)
    result = _deterministic_pre_check(inp)
    assert any(f.category == "terminal_value_dominance" for f in result.risk_findings)


# ---------------------------------------------------------------------------
# _deterministic_pre_check: sensitivity flip
# ---------------------------------------------------------------------------

def test_vetoes_on_sensitivity_flip():
    """MOS blir negativ med +1% WACC → VETO_CASH."""
    inp = _make_input(sensitivity_wacc_plus_1pct_mos=-0.05)
    result = _deterministic_pre_check(inp)
    assert result is not None
    assert result.veto == VetoAction.VETO_CASH


def test_no_veto_when_sensitivity_still_positive():
    inp = _make_input(sensitivity_wacc_plus_1pct_mos=0.02)
    result = _deterministic_pre_check(inp)
    assert result is None


def test_no_veto_when_sensitivity_not_provided():
    """Ingen sensitivity-data → ingen veto fra denne regelen."""
    inp = _make_input(sensitivity_wacc_plus_1pct_mos=None)
    result = _deterministic_pre_check(inp)
    assert result is None


def test_sensitivity_veto_finding_has_correct_category():
    inp = _make_input(sensitivity_wacc_plus_1pct_mos=-0.10)
    result = _deterministic_pre_check(inp)
    assert any(f.category == "dcf_sensitivity" for f in result.risk_findings)


# ---------------------------------------------------------------------------
# _deterministic_pre_check: DQ warnings
# ---------------------------------------------------------------------------

def test_vetoes_on_three_dq_warnings():
    inp = _make_input(dq_warn_count=3, dq_warn_rules="DQ-W001;DQ-W002;DQ-W003")
    result = _deterministic_pre_check(inp)
    assert result is not None
    assert result.veto == VetoAction.VETO_CASH


def test_no_veto_on_two_dq_warnings():
    inp = _make_input(dq_warn_count=2, dq_warn_rules="DQ-W001;DQ-W002")
    result = _deterministic_pre_check(inp)
    assert result is None


def test_dq_veto_finding_has_correct_category():
    inp = _make_input(dq_warn_count=4)
    result = _deterministic_pre_check(inp)
    assert any(f.category == "data_quality_warning" for f in result.risk_findings)


def test_dq_veto_includes_affected_fields():
    inp = _make_input(dq_warn_count=3, dq_warn_rules="DQ-W001;DQ-W002;DQ-W003")
    result = _deterministic_pre_check(inp)
    dq_finding = next(f for f in result.risk_findings if f.category == "data_quality_warning")
    assert len(dq_finding.affected_fields) == 3


# ---------------------------------------------------------------------------
# _deterministic_pre_check: quality weak count
# ---------------------------------------------------------------------------

def test_vetoes_on_two_quality_weak():
    inp = _make_input(quality_weak_count=2)
    result = _deterministic_pre_check(inp)
    assert result is not None
    assert result.veto == VetoAction.VETO_CASH


def test_no_veto_on_one_quality_weak():
    inp = _make_input(quality_weak_count=1)
    result = _deterministic_pre_check(inp)
    assert result is None


# ---------------------------------------------------------------------------
# _deterministic_pre_check: clean input gir None
# ---------------------------------------------------------------------------

def test_clean_input_returns_none():
    """Ingen veto-triggere → None (slipp gjennom til LLM)."""
    inp = _make_input(
        terminal_value_pct=0.45,
        sensitivity_wacc_plus_1pct_mos=0.05,
        dq_warn_count=1,
        quality_weak_count=0,
    )
    result = _deterministic_pre_check(inp)
    assert result is None


def test_minimal_input_returns_none():
    """Kun obligatoriske felt, ingen optionals → ingen veto."""
    inp = _make_input()
    result = _deterministic_pre_check(inp)
    assert result is None


# ---------------------------------------------------------------------------
# _deterministic_pre_check: multiple triggers
# ---------------------------------------------------------------------------

def test_multiple_triggers_all_appear_in_findings():
    """Alle fire triggere aktive → fire findings i resultatet."""
    inp = _make_input(
        terminal_value_pct=0.72,
        sensitivity_wacc_plus_1pct_mos=-0.05,
        dq_warn_count=3,
        quality_weak_count=2,
    )
    result = _deterministic_pre_check(inp)
    assert result is not None
    assert result.veto == VetoAction.VETO_CASH
    assert len(result.risk_findings) == 4


def test_pre_check_confidence_is_high():
    """Deterministisk veto skal ha høy confidence (0.95)."""
    inp = _make_input(terminal_value_pct=0.80)
    result = _deterministic_pre_check(inp)
    assert result.confidence == 0.95


# ---------------------------------------------------------------------------
# _enforce_conservatism
# ---------------------------------------------------------------------------

def test_enforce_overrides_llm_pass_when_pre_check_would_veto():
    """LLM sier PASS men pre-check ville sagt VETO → override til VETO_CASH."""
    inp = _make_input(terminal_value_pct=0.72)
    llm_output = _make_pass_output(ticker=inp.ticker)
    result = _enforce_conservatism(llm_output, inp)
    assert result.veto == VetoAction.VETO_CASH
    assert "OVERRIDE" in result.reasoning


def test_enforce_keeps_llm_veto_when_pre_check_also_vetoes():
    """LLM sier VETO_CASH og pre-check sier det samme → behold LLM-output."""
    inp = _make_input(terminal_value_pct=0.72)
    llm_output = SkepticOutput(
        ticker=inp.ticker, veto=VetoAction.VETO_CASH,
        confidence=0.9, reasoning="TV dominerer"
    )
    result = _enforce_conservatism(llm_output, inp)
    assert result.veto == VetoAction.VETO_CASH
    assert "OVERRIDE" not in result.reasoning


def test_enforce_keeps_llm_pass_when_pre_check_passes():
    """Pre-check gir None → LLM-PASS beholdes."""
    inp = _make_input()  # ingen triggere
    llm_output = _make_pass_output(ticker=inp.ticker)
    result = _enforce_conservatism(llm_output, inp)
    assert result.veto == VetoAction.PASS


def test_enforce_keeps_request_review_unchanged():
    inp = _make_input()
    llm_output = SkepticOutput(
        ticker=inp.ticker, veto=VetoAction.REQUEST_REVIEW,
        confidence=0.6, reasoning="Trenger mer data"
    )
    result = _enforce_conservatism(llm_output, inp)
    assert result.veto == VetoAction.REQUEST_REVIEW


# ---------------------------------------------------------------------------
# run_skeptic: fail-safe og deterministisk bypass
# ---------------------------------------------------------------------------

def test_skeptic_failsafe_on_error():
    """Fra planen: alle forsøk feiler → VETO_CASH som fail-safe."""
    inp = _make_input()
    result = run_skeptic(inp, client=_FailClient(), max_retries=1)
    assert result.veto == VetoAction.VETO_CASH
    assert result.confidence == 0.0
    assert "RF-FAILSAFE" in result.risk_findings[0].finding_id


def test_skeptic_bypasses_llm_on_deterministic_veto():
    """Pre-check trigger → returnerer uten å kalle client i det hele tatt."""
    inp = _make_input(terminal_value_pct=0.80)
    # _FailClient ville feile hvis den ble kalt — men pre-check skal stoppe før
    result = run_skeptic(inp, client=_FailClient(), max_retries=3)
    assert result.veto == VetoAction.VETO_CASH
    assert result.confidence == 0.95  # deterministisk confidence, ikke 0.0


def test_skeptic_failsafe_finding_category_is_other():
    inp = _make_input()
    result = run_skeptic(inp, client=_FailClient(), max_retries=1)
    assert result.risk_findings[0].category == "other"


def test_skeptic_output_agent_literal_is_correct():
    inp = _make_input(terminal_value_pct=0.80)
    result = run_skeptic(inp, client=_FailClient(), max_retries=1)
    assert result.agent == "investment_skeptic"


def test_skeptic_output_ticker_matches_input():
    inp = _make_input(ticker="DNB.OL", terminal_value_pct=0.80)
    result = run_skeptic(inp, client=_FailClient(), max_retries=1)
    assert result.ticker == "DNB.OL"
