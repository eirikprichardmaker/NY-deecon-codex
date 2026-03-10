"""
Tests for src/agents/business_quality_evaluator.py og run_quality_on_shortlist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.agents.schemas import QualityInput, QualityOutput, VetoAction
from src.agents.business_quality_evaluator import (
    _deterministic_pre_check,
    _fallback_output,
    build_quality_input,
    run_quality_evaluator,
)
from src.agents.runner import run_quality_on_shortlist


# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def _quality_input(**overrides) -> QualityInput:
    base = dict(
        ticker="EQNR.OL",
        market="OSE",
        asof_date="2026-03-10",
        text_evidence=[{"source_id": "AR2025", "source_type": "annual_report", "content": "Strong FCF"}],
        numeric_snapshot={"roic_current": 0.12, "nd_ebitda": 1.5, "mos": 0.18},
    )
    base.update(overrides)
    return QualityInput(**base)


class _FailClient:
    class beta:
        class chat:
            class completions:
                @staticmethod
                def parse(**kwargs):
                    raise RuntimeError("API utilgjengelig")


def _shortlist_df(*ticker_overrides) -> pd.DataFrame:
    defaults = {
        "ticker": "EQNR.OL",
        "adj_close": 280.0,
        "intrinsic_value": 320.0,
        "mos": 0.125,
        "wacc_used": 0.09,
        "terminal_growth": 0.02,
        "quality_weak_count": 0,
        "roic_current": 0.12,
        "nd_ebitda": 1.5,
        "relevant_index_key": "OSE",
    }
    rows = [dict(defaults, **o) for o in (ticker_overrides or [{}])]
    return pd.DataFrame(rows)


def _agent_cfg(enabled: bool = True, quality_enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "model": "gpt-4o",
        "business_quality_evaluator": {"enabled": quality_enabled},
        "cost_control": {"max_tickers_to_analyze": 5},
    }


# ---------------------------------------------------------------------------
# _deterministic_pre_check
# ---------------------------------------------------------------------------

def test_pre_check_no_text_gives_unknown():
    inp = _quality_input(text_evidence=[])
    out = _deterministic_pre_check(inp)
    assert out is not None
    assert out.quality_verdict == "unknown"
    assert out.veto == VetoAction.REQUEST_REVIEW


def test_pre_check_good_metrics_returns_none():
    inp = _quality_input()
    out = _deterministic_pre_check(inp)
    assert out is None


def test_pre_check_very_negative_roic_and_high_debt_gives_veto():
    inp = _quality_input(numeric_snapshot={"roic_current": -0.20, "nd_ebitda": 9.0})
    out = _deterministic_pre_check(inp)
    assert out is not None
    assert out.veto == VetoAction.VETO_CASH
    assert out.quality_verdict == "weak"


def test_pre_check_only_one_flag_no_veto():
    """Én negativ faktor alene er ikke nok til automatisk veto."""
    inp = _quality_input(numeric_snapshot={"roic_current": -0.15, "nd_ebitda": 1.0})
    out = _deterministic_pre_check(inp)
    assert out is None


# ---------------------------------------------------------------------------
# _fallback_output
# ---------------------------------------------------------------------------

def test_fallback_ticker_preserved():
    inp = _quality_input(ticker="DNB.OL")
    out = _fallback_output(inp, reason="test")
    assert out.ticker == "DNB.OL"


def test_fallback_verdict_is_unknown():
    inp = _quality_input()
    out = _fallback_output(inp, reason="test")
    assert out.quality_verdict == "unknown"


def test_fallback_veto_is_request_review():
    inp = _quality_input()
    out = _fallback_output(inp, reason="test")
    assert out.veto == VetoAction.REQUEST_REVIEW


def test_fallback_confidence_is_zero():
    inp = _quality_input()
    out = _fallback_output(inp, reason="LLM feil")
    assert out.confidence == 0.0


def test_fallback_reason_in_flags():
    inp = _quality_input()
    out = _fallback_output(inp, reason="Timeout etter 30s")
    assert any("Timeout" in f for f in out.flags)


def test_fallback_agent_literal():
    inp = _quality_input()
    out = _fallback_output(inp, reason="test")
    assert out.agent == "business_quality_evaluator"


# ---------------------------------------------------------------------------
# run_quality_evaluator: feilhåndtering
# ---------------------------------------------------------------------------

def test_run_quality_client_none_returns_fallback_when_text_present():
    inp = _quality_input()
    out = run_quality_evaluator(inp, client=None)
    assert isinstance(out, QualityOutput)
    assert out.ticker == inp.ticker


def test_run_quality_client_none_no_text_gives_request_review():
    inp = _quality_input(text_evidence=[])
    out = run_quality_evaluator(inp, client=None)
    assert out.veto == VetoAction.REQUEST_REVIEW


def test_run_quality_api_error_returns_fallback():
    inp = _quality_input()
    out = run_quality_evaluator(inp, client=_FailClient())
    assert isinstance(out, QualityOutput)


def test_run_quality_no_decision_field():
    """QualityOutput har ikke 'decision'-felt — kan aldri overstyre."""
    assert "decision" not in QualityOutput.model_fields


def test_run_quality_deterministic_veto_bypasses_client():
    """Pre-check veto skal skje uten å kalle client."""
    inp = _quality_input(
        text_evidence=[],
        numeric_snapshot={"roic_current": -0.20, "nd_ebitda": 9.5},
    )
    # client=None — men pre-check skal gi svar basert på numeric_snapshot (ingen tekst → REQUEST_REVIEW)
    out = run_quality_evaluator(inp, client=None)
    assert out.veto in (VetoAction.REQUEST_REVIEW, VetoAction.VETO_CASH)


# ---------------------------------------------------------------------------
# build_quality_input
# ---------------------------------------------------------------------------

def test_build_quality_input_fields():
    inp = build_quality_input(
        ticker="EQNR.OL",
        asof="2026-03-10",
        numeric_snapshot={"roic_current": 0.15},
    )
    assert inp.ticker == "EQNR.OL"
    assert inp.asof_date == "2026-03-10"
    assert inp.market == "OSE"
    assert inp.text_evidence == []
    assert inp.numeric_snapshot["roic_current"] == 0.15


def test_build_quality_input_with_text_evidence():
    evidence = [{"source_id": "AR2025", "source_type": "annual_report", "content": "ok"}]
    inp = build_quality_input(
        ticker="DNB.OL", asof="2026-03-10", numeric_snapshot={},
        text_evidence=evidence,
    )
    assert len(inp.text_evidence) == 1


# ---------------------------------------------------------------------------
# run_quality_on_shortlist: bypass
# ---------------------------------------------------------------------------

def test_quality_bypass_when_agents_disabled(tmp_path):
    df = _shortlist_df({"ticker": "EQNR.OL"}, {"ticker": "DNB.OL"})
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(enabled=False),
    )
    assert results == {}


def test_quality_bypass_when_quality_disabled(tmp_path):
    df = _shortlist_df()
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(quality_enabled=False),
    )
    assert results == {}


def test_quality_bypass_writes_empty_json(tmp_path):
    df = _shortlist_df()
    run_quality_on_shortlist(df, run_dir=tmp_path, agent_cfg=_agent_cfg(enabled=False))
    data = json.loads((tmp_path / "quality_results.json").read_text())
    assert data == {}


def test_quality_empty_shortlist(tmp_path):
    df = pd.DataFrame()
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path, agent_cfg=_agent_cfg(),
        _quality_fn=lambda inp, **kw: _fallback_output(inp, "test"),
    )
    assert results == {}


# ---------------------------------------------------------------------------
# run_quality_on_shortlist: aktiv kjøring
# ---------------------------------------------------------------------------

def _pass_quality(inp, **kwargs) -> QualityOutput:
    return QualityOutput(
        ticker=inp.ticker, quality_verdict="strong",
        veto=VetoAction.PASS, confidence=0.9, flags=[],
    )


def _weak_quality(inp, **kwargs) -> QualityOutput:
    return QualityOutput(
        ticker=inp.ticker, quality_verdict="weak",
        veto=VetoAction.VETO_CASH, confidence=0.85,
        flags=["Svake kvalitetsindikatorer"],
    )


def test_quality_pass_result_returned(tmp_path):
    df = _shortlist_df()
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        asof="2026-03-10",
        _quality_fn=_pass_quality,
    )
    assert results["EQNR.OL"].quality_verdict == "strong"
    assert results["EQNR.OL"].veto == VetoAction.PASS


def test_quality_veto_result_returned(tmp_path):
    df = _shortlist_df()
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _quality_fn=_weak_quality,
    )
    assert results["EQNR.OL"].veto == VetoAction.VETO_CASH


def test_quality_results_written_to_json(tmp_path):
    df = _shortlist_df()
    run_quality_on_shortlist(
        df, run_dir=tmp_path, agent_cfg=_agent_cfg(),
        _quality_fn=_pass_quality,
    )
    path = tmp_path / "quality_results.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "EQNR.OL" in data
    assert data["EQNR.OL"]["quality_verdict"] == "strong"


def test_quality_exception_gives_request_review(tmp_path):
    def _exploding(inp, **kwargs):
        raise RuntimeError("Uventet feil")

    df = _shortlist_df()
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path, agent_cfg=_agent_cfg(),
        _quality_fn=_exploding,
    )
    assert results["EQNR.OL"].veto == VetoAction.REQUEST_REVIEW
    assert results["EQNR.OL"].confidence == 0.0


def test_quality_partial_failure(tmp_path):
    def _selective(inp, **kwargs):
        if inp.ticker == "BAD.OL":
            raise RuntimeError("Feil")
        return _pass_quality(inp)

    df = _shortlist_df({"ticker": "OK.OL"}, {"ticker": "BAD.OL"})
    results = run_quality_on_shortlist(
        df, run_dir=tmp_path, agent_cfg=_agent_cfg(),
        _quality_fn=_selective,
    )
    assert results["OK.OL"].quality_verdict == "strong"
    assert results["BAD.OL"].confidence == 0.0


def test_quality_max_tickers_respected(tmp_path):
    df = _shortlist_df(
        {"ticker": "A.OL"}, {"ticker": "B.OL"},
        {"ticker": "C.OL"}, {"ticker": "D.OL"},
    )
    calls = []

    def _counting(inp, **kwargs):
        calls.append(inp.ticker)
        return _pass_quality(inp)

    cfg = {**_agent_cfg(), "cost_control": {"max_tickers_to_analyze": 2}}
    run_quality_on_shortlist(df, run_dir=tmp_path, agent_cfg=cfg, _quality_fn=_counting)
    assert len(calls) == 2
