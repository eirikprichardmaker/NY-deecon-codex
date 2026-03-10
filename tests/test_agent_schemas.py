"""
Tests for src/agents/schemas.py
Kontraktstester for Pydantic agent I/O-schemas.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from pydantic import ValidationError

from src.agents.schemas import (
    DossierInput,
    DossierOutput,
    QualityInput,
    QualityOutput,
    RiskFinding,
    SkepticInput,
    SkepticOutput,
    VetoAction,
)


# ---------------------------------------------------------------------------
# VetoAction: strukturell garanti — ingen BUY
# ---------------------------------------------------------------------------

def test_skeptic_output_cannot_contain_buy():
    """Strukturell garanti: ingen BUY i VetoAction enum."""
    veto_values = [v.value for v in VetoAction]
    assert "BUY" not in veto_values
    assert "STRONG_BUY" not in veto_values
    assert "RECOMMEND_BUY" not in veto_values


def test_veto_action_has_exactly_three_values():
    values = {v.value for v in VetoAction}
    assert values == {"PASS", "VETO_CASH", "REQUEST_REVIEW"}


# ---------------------------------------------------------------------------
# RiskFinding
# ---------------------------------------------------------------------------

def test_risk_finding_valid():
    f = RiskFinding(
        finding_id="RF-001",
        category="terminal_value_dominance",
        severity="critical",
        description="TV > 60% of DCF",
    )
    assert f.finding_id == "RF-001"
    assert f.affected_fields == []


def test_risk_finding_with_affected_fields():
    f = RiskFinding(
        finding_id="RF-002",
        category="dcf_sensitivity",
        severity="high",
        description="MOS flips with +1% WACC",
        affected_fields=["wacc_used", "intrinsic_value"],
    )
    assert "wacc_used" in f.affected_fields


def test_risk_finding_rejects_unknown_category():
    with pytest.raises(ValidationError):
        RiskFinding(
            finding_id="RF-X",
            category="buy_signal",   # ugyldig kategori
            severity="low",
            description="test",
        )


def test_risk_finding_rejects_unknown_severity():
    with pytest.raises(ValidationError):
        RiskFinding(
            finding_id="RF-X",
            category="other",
            severity="extreme",   # ugyldig severity
            description="test",
        )


def test_risk_finding_description_max_length():
    with pytest.raises(ValidationError):
        RiskFinding(
            finding_id="RF-X",
            category="other",
            severity="low",
            description="A" * 501,   # Over 500 tegn
        )


# ---------------------------------------------------------------------------
# SkepticInput
# ---------------------------------------------------------------------------

def _valid_skeptic_input(**overrides) -> dict:
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
    return base


def test_skeptic_input_valid_minimal():
    inp = SkepticInput(**_valid_skeptic_input())
    assert inp.ticker == "EQNR.OL"
    assert inp.dq_warn_count == 0
    assert inp.value_creation_ok is True


def test_skeptic_input_with_optional_fields():
    inp = SkepticInput(**_valid_skeptic_input(
        terminal_value_pct=0.72,
        sensitivity_wacc_plus_1pct_mos=-0.05,
        roic_current=0.15,
        fcf_yield=0.08,
        nd_ebitda=2.5,
        dq_warn_count=2,
        dq_warn_rules="DQ-W001;DQ-W002",
    ))
    assert inp.terminal_value_pct == 0.72
    assert inp.dq_warn_count == 2


def test_skeptic_input_rejects_missing_required():
    with pytest.raises(ValidationError):
        SkepticInput(ticker="EQNR.OL")  # mangler mange felt


# ---------------------------------------------------------------------------
# SkepticOutput
# ---------------------------------------------------------------------------

def test_skeptic_output_validates():
    """Fra planen: test at SkepticOutput validerer korrekt."""
    output = SkepticOutput(
        ticker="EQNR.OL",
        veto=VetoAction.VETO_CASH,
        confidence=0.85,
        reasoning="Terminal value dominates",
        risk_findings=[
            RiskFinding(
                finding_id="RF-001",
                category="terminal_value_dominance",
                severity="critical",
                description="TV > 60% of DCF",
            )
        ],
    )
    assert output.veto == VetoAction.VETO_CASH
    assert output.agent == "investment_skeptic"
    assert len(output.risk_findings) == 1


def test_skeptic_output_rejects_confidence_above_1():
    with pytest.raises(ValidationError):
        SkepticOutput(
            ticker="EQNR.OL",
            veto=VetoAction.PASS,
            confidence=1.1,
            reasoning="test",
        )


def test_skeptic_output_rejects_confidence_below_0():
    with pytest.raises(ValidationError):
        SkepticOutput(
            ticker="EQNR.OL",
            veto=VetoAction.PASS,
            confidence=-0.1,
            reasoning="test",
        )


def test_skeptic_output_reasoning_max_length():
    with pytest.raises(ValidationError):
        SkepticOutput(
            ticker="EQNR.OL",
            veto=VetoAction.PASS,
            confidence=0.5,
            reasoning="X" * 2001,
        )


def test_skeptic_output_default_agent_literal():
    out = SkepticOutput(
        ticker="T", veto=VetoAction.PASS, confidence=0.5, reasoning="ok"
    )
    assert out.agent == "investment_skeptic"
    assert out.version == "1.0"


def test_skeptic_output_all_veto_actions_valid():
    for action in VetoAction:
        out = SkepticOutput(
            ticker="T", veto=action, confidence=0.5, reasoning="ok"
        )
        assert out.veto == action


# ---------------------------------------------------------------------------
# QualityOutput
# ---------------------------------------------------------------------------

def test_quality_output_valid():
    out = QualityOutput(
        ticker="DNB.OL",
        quality_verdict="strong",
        veto=VetoAction.PASS,
        confidence=0.80,
    )
    assert out.agent == "business_quality_evaluator"
    assert out.quality_verdict == "strong"


def test_quality_output_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        QualityOutput(
            ticker="DNB.OL",
            quality_verdict="excellent",   # ugyldig
            veto=VetoAction.PASS,
            confidence=0.80,
        )


def test_quality_output_accepts_all_verdicts():
    for verdict in ["strong", "mixed", "weak", "unknown"]:
        out = QualityOutput(
            ticker="T",
            quality_verdict=verdict,
            veto=VetoAction.PASS,
            confidence=0.5,
        )
        assert out.quality_verdict == verdict


# ---------------------------------------------------------------------------
# DossierInput / DossierOutput
# ---------------------------------------------------------------------------

def test_dossier_input_valid():
    inp = DossierInput(
        ticker="EQNR.OL",
        market="OSE",
        asof_date="2026-03-10",
        final_decision="KANDIDAT",
        gate_log=[{"gate": "mos", "passed": True}],
        valuation_summary={"mos": 0.17, "intrinsic_value": 300.0},
    )
    assert inp.skeptic_output is None
    assert inp.quality_output is None


def test_dossier_input_with_agent_outputs():
    skeptic = SkepticOutput(
        ticker="EQNR.OL", veto=VetoAction.PASS, confidence=0.9, reasoning="ok"
    )
    quality = QualityOutput(
        ticker="EQNR.OL", quality_verdict="strong",
        veto=VetoAction.PASS, confidence=0.85,
    )
    inp = DossierInput(
        ticker="EQNR.OL",
        market="OSE",
        asof_date="2026-03-10",
        final_decision="KANDIDAT",
        gate_log=[],
        valuation_summary={},
        skeptic_output=skeptic,
        quality_output=quality,
    )
    assert inp.skeptic_output.veto == VetoAction.PASS
    assert inp.quality_output.quality_verdict == "strong"


def test_dossier_output_has_no_decision_field():
    """DossierOutput kan IKKE overstyre beslutningen — ingen 'decision'-felt."""
    out = DossierOutput(ticker="EQNR.OL", narrative="Solid selskap.")
    field_names = set(out.model_fields_set) | set(DossierOutput.model_fields.keys())
    assert "decision" not in field_names


def test_dossier_output_narrative_max_length():
    with pytest.raises(ValidationError):
        DossierOutput(ticker="T", narrative="X" * 5001)


def test_dossier_output_defaults():
    out = DossierOutput(ticker="T", narrative="ok")
    assert out.key_risks == []
    assert out.key_strengths == []
    assert out.data_quality_summary == ""
    assert out.agent == "decision_dossier_writer"


# ---------------------------------------------------------------------------
# JSON-serialisering (round-trip)
# ---------------------------------------------------------------------------

def test_skeptic_output_json_roundtrip():
    original = SkepticOutput(
        ticker="EQNR.OL",
        veto=VetoAction.VETO_CASH,
        confidence=0.95,
        reasoning="TV dominance",
        risk_findings=[
            RiskFinding(
                finding_id="RF-001",
                category="terminal_value_dominance",
                severity="critical",
                description="TV > 60%",
            )
        ],
    )
    json_str = original.model_dump_json()
    restored = SkepticOutput.model_validate_json(json_str)
    assert restored.veto == original.veto
    assert restored.confidence == original.confidence
    assert len(restored.risk_findings) == 1


def test_quality_output_json_roundtrip():
    original = QualityOutput(
        ticker="DNB.OL",
        quality_verdict="weak",
        veto=VetoAction.VETO_CASH,
        confidence=0.70,
        flags=["low_roic", "high_leverage"],
    )
    restored = QualityOutput.model_validate_json(original.model_dump_json())
    assert restored.quality_verdict == "weak"
    assert restored.flags == ["low_roic", "high_leverage"]
