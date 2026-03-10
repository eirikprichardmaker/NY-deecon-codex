"""
Strict Pydantic schemas for all agent I/O.
These are the "output contracts" — agents MUST produce valid JSON matching these.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class VetoAction(str, Enum):
    PASS = "PASS"
    VETO_CASH = "VETO_CASH"
    REQUEST_REVIEW = "REQUEST_REVIEW"


class RiskFinding(BaseModel):
    finding_id: str
    category: Literal[
        "terminal_value_dominance",
        "dcf_sensitivity",
        "data_quality_warning",
        "margin_assumption_aggressive",
        "growth_assumption_aggressive",
        "accounting_risk",
        "regulatory_risk",
        "governance_risk",
        "cyclical_peak_earnings",
        "liquidity_risk",
        "other",
    ]
    severity: Literal["critical", "high", "medium", "low"]
    description: str = Field(max_length=500)
    affected_fields: list[str] = Field(default_factory=list)


class SkepticInput(BaseModel):
    """Input to Investment Skeptic agent."""

    ticker: str
    market: str
    asof_date: str
    intrinsic_value: float
    current_price: float
    mos: float
    wacc_used: float
    terminal_growth: float
    terminal_value_pct: Optional[float] = None
    sensitivity_wacc_plus_1pct_mos: Optional[float] = None
    sensitivity_wacc_minus_1pct_mos: Optional[float] = None
    roic_current: Optional[float] = None
    fcf_yield: Optional[float] = None
    nd_ebitda: Optional[float] = None
    dq_warn_count: int = 0
    dq_warn_rules: str = ""
    quality_weak_count: int = 0
    value_creation_ok: bool = True


class SkepticOutput(BaseModel):
    """Output from Investment Skeptic agent. Note: no BUY option exists."""

    agent: Literal["investment_skeptic"] = "investment_skeptic"
    version: str = "1.0"
    ticker: str
    veto: VetoAction  # Can ONLY be PASS, VETO_CASH, or REQUEST_REVIEW
    confidence: float = Field(ge=0.0, le=1.0)
    risk_findings: list[RiskFinding] = Field(default_factory=list)
    reasoning: str = Field(max_length=2000)


class QualityInput(BaseModel):
    """Input to Business Quality Evaluator agent."""

    ticker: str
    market: str
    asof_date: str
    text_evidence: list[dict]   # {source_id, source_type, content}
    numeric_snapshot: dict      # Pre-calculated metrics


class QualityOutput(BaseModel):
    """Output from Business Quality Evaluator. Note: no BUY option."""

    agent: Literal["business_quality_evaluator"] = "business_quality_evaluator"
    version: str = "1.0"
    ticker: str
    quality_verdict: Literal["strong", "mixed", "weak", "unknown"]
    veto: VetoAction
    confidence: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    evidence_citations: list[dict] = Field(default_factory=list)


class DossierInput(BaseModel):
    """Input to Decision Dossier Writer."""

    ticker: str
    market: str
    asof_date: str
    final_decision: str  # "KANDIDAT" or "CASH"
    gate_log: list[dict]
    valuation_summary: dict
    skeptic_output: Optional[SkepticOutput] = None
    quality_output: Optional[QualityOutput] = None


class DossierOutput(BaseModel):
    """Output from Dossier Writer. NO decision field — cannot override."""

    agent: Literal["decision_dossier_writer"] = "decision_dossier_writer"
    version: str = "1.0"
    ticker: str
    narrative: str = Field(max_length=5000)
    key_risks: list[str] = Field(default_factory=list)
    key_strengths: list[str] = Field(default_factory=list)
    data_quality_summary: str = ""
