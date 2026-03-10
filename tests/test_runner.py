"""
Tests for src/agents/runner.py
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

from src.agents.schemas import RiskFinding, SkepticInput, SkepticOutput, VetoAction
from src.agents.runner import (
    _build_skeptic_input,
    _make_bypass_output,
    run_skeptic_on_shortlist,
)


# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def _shortlist(*overrides_list) -> pd.DataFrame:
    """Bygger en minimal shortlist-df med 1+ rader."""
    defaults = {
        "ticker": "EQNR.OL",
        "adj_close": 280.0,
        "intrinsic_value": 320.0,
        "mos": 0.125,
        "wacc_used": 0.09,
        "terminal_growth": 0.02,
        "quality_weak_count": 0,
        "value_creation_ok": True,
        "dq_warn_count_v2": 0,
        "dq_warn_rules": "",
        "relevant_index_key": "OSE",
    }
    rows = []
    for overrides in (overrides_list or [{}]):
        row = {**defaults, **overrides}
        rows.append(row)
    return pd.DataFrame(rows)


def _pass_skeptic(inp: SkepticInput, **kwargs) -> SkepticOutput:
    return SkepticOutput(ticker=inp.ticker, veto=VetoAction.PASS,
                         confidence=0.9, reasoning="ok")


def _veto_skeptic(inp: SkepticInput, **kwargs) -> SkepticOutput:
    return SkepticOutput(
        ticker=inp.ticker, veto=VetoAction.VETO_CASH,
        confidence=0.95, reasoning="TV dominance",
        risk_findings=[RiskFinding(
            finding_id="RF-TV001", category="terminal_value_dominance",
            severity="critical", description="TV > 60%"
        )],
    )


def _agent_cfg(enabled: bool = True, skeptic_enabled: bool = True,
               max_tickers: int = 5) -> dict:
    return {
        "enabled": enabled,
        "model": "gpt-4o",
        "max_retries": 1,
        "investment_skeptic": {"enabled": skeptic_enabled},
        "cost_control": {"max_tickers_to_analyze": max_tickers},
    }


# ---------------------------------------------------------------------------
# Bypass: agents.enabled=false
# ---------------------------------------------------------------------------

def test_bypass_when_agents_disabled(tmp_path):
    df = _shortlist({"ticker": "EQNR.OL"}, {"ticker": "DNB.OL"})
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(enabled=False),
        _skeptic_fn=_veto_skeptic,   # skal IKKE kalles
    )
    for ticker in ["EQNR.OL", "DNB.OL"]:
        assert results[ticker].veto == VetoAction.PASS
    assert "agents.enabled=false" in results["EQNR.OL"].reasoning


def test_bypass_when_skeptic_disabled(tmp_path):
    df = _shortlist()
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(skeptic_enabled=False),
        _skeptic_fn=_veto_skeptic,
    )
    assert results["EQNR.OL"].veto == VetoAction.PASS


def test_bypass_writes_skeptic_results_json(tmp_path):
    df = _shortlist()
    run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(enabled=False),
    )
    assert (tmp_path / "skeptic_results.json").exists()


# ---------------------------------------------------------------------------
# Aktiv kjøring: PASS
# ---------------------------------------------------------------------------

def test_pass_result_returned(tmp_path):
    df = _shortlist()
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        asof="2026-03-10",
        _skeptic_fn=_pass_skeptic,
    )
    assert results["EQNR.OL"].veto == VetoAction.PASS


def test_results_written_to_json(tmp_path):
    df = _shortlist()
    run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _skeptic_fn=_pass_skeptic,
    )
    path = tmp_path / "skeptic_results.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert "EQNR.OL" in data
    assert data["EQNR.OL"]["veto"] == "PASS"


# ---------------------------------------------------------------------------
# Aktiv kjøring: VETO_CASH
# ---------------------------------------------------------------------------

def test_veto_result_returned(tmp_path):
    df = _shortlist()
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _skeptic_fn=_veto_skeptic,
    )
    assert results["EQNR.OL"].veto == VetoAction.VETO_CASH


def test_veto_written_to_json(tmp_path):
    df = _shortlist()
    run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _skeptic_fn=_veto_skeptic,
    )
    data = json.loads((tmp_path / "skeptic_results.json").read_text())
    assert data["EQNR.OL"]["veto"] == "VETO_CASH"


# ---------------------------------------------------------------------------
# cost_control: max_tickers_to_analyze
# ---------------------------------------------------------------------------

def test_max_tickers_limits_analysis(tmp_path):
    df = _shortlist(
        {"ticker": "A.OL"}, {"ticker": "B.OL"},
        {"ticker": "C.OL"}, {"ticker": "D.OL"},
        {"ticker": "E.OL"}, {"ticker": "F.OL"},
    )
    calls = []

    def _counting_skeptic(inp, **kwargs):
        calls.append(inp.ticker)
        return _pass_skeptic(inp)

    run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(max_tickers=3),
        _skeptic_fn=_counting_skeptic,
    )
    assert len(calls) == 3


def test_empty_shortlist_returns_empty_dict(tmp_path):
    df = pd.DataFrame()
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _skeptic_fn=_pass_skeptic,
    )
    assert results == {}


# ---------------------------------------------------------------------------
# Feilhåndtering i runner
# ---------------------------------------------------------------------------

def test_exception_in_skeptic_gives_veto_cash(tmp_path):
    def _exploding_skeptic(inp, **kwargs):
        raise RuntimeError("Uventet feil")

    df = _shortlist()
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _skeptic_fn=_exploding_skeptic,
    )
    assert results["EQNR.OL"].veto == VetoAction.VETO_CASH
    assert results["EQNR.OL"].confidence == 0.0


def test_multiple_tickers_partial_failure(tmp_path):
    """Én ticker feiler → resten behandles normalt."""
    def _selective_skeptic(inp, **kwargs):
        if inp.ticker == "BAD.OL":
            raise RuntimeError("Feil for BAD")
        return _pass_skeptic(inp)

    df = _shortlist({"ticker": "OK.OL"}, {"ticker": "BAD.OL"})
    results = run_skeptic_on_shortlist(
        df, run_dir=tmp_path,
        agent_cfg=_agent_cfg(),
        _skeptic_fn=_selective_skeptic,
    )
    assert results["OK.OL"].veto == VetoAction.PASS
    assert results["BAD.OL"].veto == VetoAction.VETO_CASH


# ---------------------------------------------------------------------------
# _build_skeptic_input
# ---------------------------------------------------------------------------

def test_build_skeptic_input_basic():
    row = pd.Series({
        "ticker": "EQNR.OL",
        "adj_close": 280.0,
        "intrinsic_value": 320.0,
        "mos": 0.125,
        "wacc_used": 0.09,
        "terminal_growth": 0.02,
        "quality_weak_count": 1,
        "value_creation_ok": True,
        "dq_warn_count_v2": 2,
        "dq_warn_rules": "DQ-W001;DQ-W002",
        "relevant_index_key": "OSE",
    })
    inp = _build_skeptic_input(row, "2026-03-10")
    assert inp.ticker == "EQNR.OL"
    assert inp.asof_date == "2026-03-10"
    assert inp.mos == pytest.approx(0.125)
    assert inp.dq_warn_count == 2
    assert inp.quality_weak_count == 1


def test_build_skeptic_input_handles_nan():
    row = pd.Series({
        "ticker": "TEST.OL",
        "adj_close": float("nan"),
        "intrinsic_value": float("nan"),
        "mos": 0.10,
        "wacc_used": 0.09,
        "terminal_growth": 0.02,
    })
    inp = _build_skeptic_input(row, "2026-03-10")
    assert inp.current_price == 0.0    # fallback for NaN
    assert inp.terminal_value_pct is None


def test_build_skeptic_input_optional_sensitivity():
    row = pd.Series({
        "ticker": "T.OL",
        "mos": 0.15,
        "wacc_used": 0.09,
        "terminal_growth": 0.02,
        "sensitivity_wacc_plus_1pct_mos": -0.03,
    })
    inp = _build_skeptic_input(row, "2026-03-10")
    assert inp.sensitivity_wacc_plus_1pct_mos == pytest.approx(-0.03)


# ---------------------------------------------------------------------------
# _make_bypass_output
# ---------------------------------------------------------------------------

def test_bypass_output_is_pass():
    out = _make_bypass_output("EQNR.OL")
    assert out.veto == VetoAction.PASS
    assert out.ticker == "EQNR.OL"
    assert out.confidence == 1.0
