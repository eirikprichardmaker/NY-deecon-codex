"""
Agent A: Business Quality Evaluator (no-veto on numbers, qualitative only).
Evaluates business quality from text evidence + numeric snapshot.
Cannot introduce new numbers or override the pipeline's valuation.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from src.agents.schemas import QualityInput, QualityOutput, VetoAction

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Business Quality Evaluator for the Deecon investment system.

ROLE: Evaluate the qualitative business quality of an investment candidate.
You assess competitive moat, management quality, and structural business risks.

CONSTRAINTS (NON-NEGOTIABLE):
1. You CANNOT change or override the valuation or MOS.
2. You CANNOT introduce new numbers. All numeric references must come from numeric_snapshot.
3. You CANNOT recommend buying. Only PASS, VETO_CASH, or REQUEST_REVIEW.
4. Every qualitative claim MUST cite a specific source_id from text_evidence.
5. Write flags in Norwegian (bokmål), max 10 words each.

VETO_CASH triggers (any one is sufficient):
- Evidence of systematic accounting manipulation
- Evidence of fraudulent management behavior
- Regulatory existential threat (license revocation risk)
- Structural business model collapse (technology disruption, commodity substitution)

REQUEST_REVIEW triggers:
- Governance concerns without definitive fraud evidence
- Significant environmental/regulatory risk requiring expert verification
- Customer concentration risk (single customer > 40% revenue)

PASS when:
- No material qualitative red flags found
- quality_verdict is "strong" or "mixed" without VETO triggers

OUTPUT: Respond ONLY with valid JSON matching the required schema."""


def _deterministic_pre_check(inp: QualityInput) -> QualityOutput | None:
    """
    Fast-path: returns a result without LLM when input is clearly insufficient
    or when numeric_snapshot has hard disqualifiers.
    """
    flags: list[str] = []

    # Ingen tekstlig evidens → kan ikke vurdere kvalitet
    if not inp.text_evidence:
        return QualityOutput(
            ticker=inp.ticker,
            quality_verdict="unknown",
            veto=VetoAction.REQUEST_REVIEW,
            confidence=0.5,
            flags=["Ingen tekstlig evidens tilgjengelig"],
            evidence_citations=[],
        )

    # Sjekk numeric_snapshot for harde disqualifiers
    snap = inp.numeric_snapshot or {}
    roic = snap.get("roic_current")
    nd_ebitda = snap.get("nd_ebitda")

    if roic is not None and isinstance(roic, (int, float)) and roic < -0.10:
        flags.append(f"ROIC svært negativ ({roic:.1%}) — strukturell lønnsomhetskrise")

    if nd_ebitda is not None and isinstance(nd_ebitda, (int, float)) and nd_ebitda > 8.0:
        flags.append(f"ND/EBITDA ekstremt høy ({nd_ebitda:.1f}x) — finansiell stressrisiko")

    if len(flags) >= 2:
        return QualityOutput(
            ticker=inp.ticker,
            quality_verdict="weak",
            veto=VetoAction.VETO_CASH,
            confidence=0.85,
            flags=flags,
            evidence_citations=[],
        )

    return None


def run_quality_evaluator(
    input_data: QualityInput,
    client: Any,
    model: str = "claude-sonnet-4-6",
) -> QualityOutput:
    """
    Run Business Quality Evaluator. On failure, returns minimal valid output.
    Note: client may be None if anthropic is not installed — returns fallback.
    """
    # Deterministisk pre-sjekk
    pre = _deterministic_pre_check(input_data)
    if pre is not None:
        logger.info(f"quality: pre-check resultat for {input_data.ticker}: {pre.veto}")
        return pre

    if client is None:
        return _fallback_output(input_data, reason="Anthropic-klient ikke tilgjengelig")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(
                input_data.model_dump(), indent=2, ensure_ascii=False
            )}],
        )
        text = response.content[0].text
        output = QualityOutput.model_validate_json(text)
        return output
    except Exception as e:
        logger.error(f"Quality evaluator feilet for {input_data.ticker}: {e}")
        return _fallback_output(input_data, reason=str(e))


def _fallback_output(input_data: QualityInput, reason: str) -> QualityOutput:
    """Minimal gyldig output når LLM-kallet feiler."""
    return QualityOutput(
        ticker=input_data.ticker,
        quality_verdict="unknown",
        veto=VetoAction.REQUEST_REVIEW,
        confidence=0.0,
        flags=[f"Rapport ikke generert: {reason[:200]}"],
        evidence_citations=[],
    )


def build_quality_input(
    ticker: str,
    asof: str,
    numeric_snapshot: dict,
    text_evidence: list[dict] | None = None,
    market: str = "OSE",
) -> QualityInput:
    """Bygger QualityInput fra pipeline-data."""
    return QualityInput(
        ticker=ticker,
        market=market,
        asof_date=asof,
        text_evidence=text_evidence or [],
        numeric_snapshot=numeric_snapshot,
    )
