"""
Tests for src/agents/dossier_writer.py og src/output_contract.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.agents.schemas import DossierInput, DossierOutput, SkepticOutput, VetoAction
from src.agents.dossier_writer import (
    _fallback_output,
    build_dossier_input,
    run_dossier_writer,
)
from src.output_contract import (
    OPTIONAL_OUTPUTS,
    REQUIRED_OUTPUTS,
    list_optional_outputs,
    validate_run_outputs,
)


# ---------------------------------------------------------------------------
# Hjelpere
# ---------------------------------------------------------------------------

def _dossier_input(**overrides) -> DossierInput:
    base = dict(
        ticker="EQNR.OL",
        market="OSE",
        asof_date="2026-03-10",
        final_decision="KANDIDAT",
        gate_log=[{"gate": "mos", "passed": True}, {"gate": "technical_ok", "passed": True}],
        valuation_summary={"mos": 0.17, "intrinsic_value": 320.0, "wacc_used": 0.09},
    )
    base.update(overrides)
    return DossierInput(**base)


class _FailClient:
    class beta:
        class chat:
            class completions:
                @staticmethod
                def parse(**kwargs):
                    raise RuntimeError("API utilgjengelig")


# ---------------------------------------------------------------------------
# _fallback_output
# ---------------------------------------------------------------------------

def test_fallback_output_contains_ticker():
    inp = _dossier_input()
    out = _fallback_output(inp, reason="test")
    assert inp.ticker in out.narrative


def test_fallback_output_contains_decision():
    inp = _dossier_input(final_decision="CASH")
    out = _fallback_output(inp, reason="test")
    assert "CASH" in out.narrative


def test_fallback_output_contains_reason():
    inp = _dossier_input()
    out = _fallback_output(inp, reason="OpenAI nede")
    assert "OpenAI nede" in out.narrative


def test_fallback_output_has_no_decision_field():
    inp = _dossier_input()
    out = _fallback_output(inp, reason="test")
    assert not hasattr(out, "decision") or "decision" not in DossierOutput.model_fields


def test_fallback_output_agent_literal():
    inp = _dossier_input()
    out = _fallback_output(inp, reason="test")
    assert out.agent == "decision_dossier_writer"


def test_fallback_output_ticker_matches():
    inp = _dossier_input(ticker="DNB.OL")
    out = _fallback_output(inp, reason="test")
    assert out.ticker == "DNB.OL"


# ---------------------------------------------------------------------------
# run_dossier_writer: feilhåndtering
# ---------------------------------------------------------------------------

def test_run_dossier_returns_fallback_on_client_none():
    inp = _dossier_input()
    out = run_dossier_writer(inp, client=None)
    assert isinstance(out, DossierOutput)
    assert out.ticker == inp.ticker
    assert "ikke tilgjengelig" in out.narrative


def test_run_dossier_returns_fallback_on_api_error():
    inp = _dossier_input()
    out = run_dossier_writer(inp, client=_FailClient())
    assert isinstance(out, DossierOutput)
    assert out.ticker == inp.ticker


def test_run_dossier_fallback_has_key_risks():
    inp = _dossier_input()
    out = run_dossier_writer(inp, client=None)
    assert isinstance(out.key_risks, list)
    assert len(out.key_risks) >= 1


def test_run_dossier_cannot_change_decision():
    """DossierOutput har ikke 'decision'-felt — kan aldri overstyre."""
    inp = _dossier_input(final_decision="CASH")
    out = run_dossier_writer(inp, client=None)
    assert "decision" not in DossierOutput.model_fields


# ---------------------------------------------------------------------------
# build_dossier_input
# ---------------------------------------------------------------------------

def test_build_dossier_input_fields():
    inp = build_dossier_input(
        ticker="EQNR.OL",
        asof="2026-03-10",
        final_decision="KANDIDAT",
        gate_log=[{"gate": "mos", "passed": True}],
        valuation_summary={"mos": 0.17},
    )
    assert inp.ticker == "EQNR.OL"
    assert inp.asof_date == "2026-03-10"
    assert inp.final_decision == "KANDIDAT"
    assert inp.market == "OSE"


def test_build_dossier_input_with_skeptic():
    skeptic = SkepticOutput(
        ticker="EQNR.OL", veto=VetoAction.PASS, confidence=0.9, reasoning="ok"
    )
    inp = build_dossier_input(
        ticker="EQNR.OL",
        asof="2026-03-10",
        final_decision="KANDIDAT",
        gate_log=[],
        valuation_summary={},
        skeptic_output=skeptic,
    )
    assert inp.skeptic_output.veto == VetoAction.PASS


def test_build_dossier_input_no_agents():
    inp = build_dossier_input(
        ticker="T.OL",
        asof="2026-03-10",
        final_decision="CASH",
        gate_log=[],
        valuation_summary={},
    )
    assert inp.skeptic_output is None
    assert inp.quality_output is None


# ---------------------------------------------------------------------------
# validate_run_outputs
# ---------------------------------------------------------------------------

def test_validate_all_present(tmp_path):
    for fname in REQUIRED_OUTPUTS:
        (tmp_path / fname).write_text("ok")
    ok, missing = validate_run_outputs(tmp_path)
    assert ok is True
    assert missing == []


def test_validate_missing_one(tmp_path):
    for fname in REQUIRED_OUTPUTS:
        (tmp_path / fname).write_text("ok")
    (tmp_path / "decision.csv").unlink()
    ok, missing = validate_run_outputs(tmp_path)
    assert ok is False
    assert "decision.csv" in missing


def test_validate_empty_dir(tmp_path):
    ok, missing = validate_run_outputs(tmp_path)
    assert ok is False
    assert len(missing) == len(REQUIRED_OUTPUTS)


def test_validate_returns_all_missing_files(tmp_path):
    ok, missing = validate_run_outputs(tmp_path)
    assert set(missing) == set(REQUIRED_OUTPUTS)


def test_validate_optional_not_required(tmp_path):
    """Manglende optional-filer skal ikke feile validering."""
    for fname in REQUIRED_OUTPUTS:
        (tmp_path / fname).write_text("ok")
    # Ingen optional-filer laget
    ok, missing = validate_run_outputs(tmp_path)
    assert ok is True


# ---------------------------------------------------------------------------
# list_optional_outputs
# ---------------------------------------------------------------------------

def test_list_optional_empty_dir(tmp_path):
    present = list_optional_outputs(tmp_path)
    assert present == []


def test_list_optional_some_present(tmp_path):
    (tmp_path / "decision_agent.md").write_text("rapport")
    (tmp_path / "skeptic_results.json").write_text("{}")
    present = list_optional_outputs(tmp_path)
    assert "decision_agent.md" in present
    assert "skeptic_results.json" in present


def test_list_optional_all_present(tmp_path):
    for fname in OPTIONAL_OUTPUTS:
        (tmp_path / fname).write_text("ok")
    present = list_optional_outputs(tmp_path)
    assert set(present) == set(OPTIONAL_OUTPUTS)


# ---------------------------------------------------------------------------
# Output-kontrakt: struktur
# ---------------------------------------------------------------------------

def test_required_outputs_count():
    assert len(REQUIRED_OUTPUTS) == 10


def test_required_outputs_includes_key_files():
    required = set(REQUIRED_OUTPUTS)
    assert "manifest.json" in required
    assert "decision.csv" in required
    assert "decision.md" in required
    assert "valuation.csv" in required
    assert "data_quality_audit_v2.csv" in required


def test_optional_outputs_includes_agent_files():
    optional = set(OPTIONAL_OUTPUTS)
    assert "decision_agent.md" in optional
    assert "decision_agent.json" in optional
    assert "skeptic_results.json" in optional
