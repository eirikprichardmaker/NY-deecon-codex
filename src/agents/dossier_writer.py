"""
Agent C: Decision Dossier Writer (no-veto, no-math).
Produces auditable decision narrative. Cannot introduce new numbers or change decisions.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from src.agents.schemas import DossierInput, DossierOutput

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Decision Dossier Writer for the Deecon system.

ROLE: Write an auditable, human-readable decision report in Norwegian.

CONSTRAINTS (NON-NEGOTIABLE):
1. You CANNOT change or override the decision (KANDIDAT or CASH).
2. You CANNOT introduce new numbers. All numbers must come from the input data.
3. You CANNOT compute ratios, multiples, or valuations.
4. Every claim must reference a specific data point from the input.
5. Write in clear, professional Norwegian (bokmål).

STRUCTURE:
- Beslutningssammendrag (1 paragraph)
- Verdsettelse og margin of safety (reference exact numbers from input)
- Risikofaktorer (from skeptic findings if available)
- Kvalitetsvurdering (from quality evaluation if available)
- Datakvalitet (summary of DQ status)
- Teknisk status (MA200/MAD/index)

OUTPUT: Respond ONLY with valid JSON matching the required schema."""


def run_dossier_writer(
    input_data: DossierInput,
    client: Any,
    model: str = "claude-sonnet-4-6",
) -> DossierOutput:
    """
    Run Decision Dossier Writer. On failure, returns minimal valid output.
    Note: client may be None if anthropic is not installed — returns fallback.
    """
    if client is None:
        return _fallback_output(input_data, reason="Anthropic-klient ikke tilgjengelig")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(
                input_data.model_dump(), indent=2, ensure_ascii=False
            )}],
        )
        text = response.content[0].text
        output = DossierOutput.model_validate_json(text)
        return output
    except Exception as e:
        logger.error(f"Dossier writer feilet for {input_data.ticker}: {e}")
        return _fallback_output(input_data, reason=str(e))


def _fallback_output(input_data: DossierInput, reason: str) -> DossierOutput:
    """Minimal gyldig output når LLM-kallet feiler."""
    decision_text = (
        "KANDIDAT" if input_data.final_decision == "KANDIDAT" else "CASH"
    )
    return DossierOutput(
        ticker=input_data.ticker,
        narrative=(
            f"Automatisk rapport for {input_data.ticker} ({input_data.asof_date}): "
            f"Beslutning er {decision_text}. "
            f"Detaljert begrunnelse kunne ikke genereres: {reason}"
        ),
        key_risks=["Rapport ikke generert — se decision.csv for detaljer"],
        data_quality_summary="Ukjent — rapport ikke generert",
    )


def build_dossier_input(
    ticker: str,
    asof: str,
    final_decision: str,
    gate_log: list[dict],
    valuation_summary: dict,
    skeptic_output=None,
    quality_output=None,
    market: str = "OSE",
) -> DossierInput:
    """Bygger DossierInput fra pipeline-data."""
    return DossierInput(
        ticker=ticker,
        market=market,
        asof_date=asof,
        final_decision=final_decision,
        gate_log=gate_log,
        valuation_summary=valuation_summary,
        skeptic_output=skeptic_output,
        quality_output=quality_output,
    )
