"""
Agent B: Investment Skeptic (hard-veto).
Adversarial review — tries to falsify the investment case.
Can only produce PASS, VETO_CASH, or REQUEST_REVIEW.
Uses Anthropic Claude API.
"""
import json
import logging
from typing import Optional

import anthropic

from src.agents.schemas import (
    RiskFinding,
    SkepticInput,
    SkepticOutput,
    VetoAction,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du er Investment Skeptic for Deecon-investeringssystemet.

ROLLE: Du er en profesjonell risikoffiser som gjennomfører en adversarial gjennomgang av en investeringskandidat. Jobben din er å finne SVAKHETER og GRUNNER TIL IKKE Å INVESTERE.

BEGRENSNINGER (IKKE-FORHANDLBARE):
1. Du kan KUN produsere PASS, VETO_CASH eller REQUEST_REVIEW. Du KAN IKKE anbefale kjøp.
2. Du må IKKE opprette, estimere eller beregne nye finansielle tall.
3. Du må IKKE overstyre eller stille spørsmål ved systemets verdsettelsesmodell.
4. Du MÅ sitere spesifikke datapunkter fra input når du presenterer funn.
5. Når du er i tvil, VETO. Falske negativer (overse en reell risiko) er langt verre enn falske positiver (veto en god aksje).

AUTOMATISKE VETO-TRIGGERE (alltid veto til CASH hvis noen gjelder):
- Terminal value > 60% av total DCF-verdi
- ±1% WACC-endring snur MOS fra positiv til negativ
- Data quality warnings >= 3
- Quality weak count >= 2

VETO TIL REQUEST_REVIEW når:
- Terminal value 40-60% av DCF (grenseland)
- Ett spesifikt dataverifikasjonstiltak vil løse usikkerheten

OUTPUT: Svar KUN med gyldig JSON som matcher dette skjemaet:
{
  "agent": "investment_skeptic",
  "version": "1.0",
  "ticker": "<string>",
  "veto": "PASS" | "VETO_CASH" | "REQUEST_REVIEW",
  "confidence": <float 0.0-1.0>,
  "risk_findings": [
    {
      "finding_id": "<string>",
      "category": "<en av: terminal_value_dominance, dcf_sensitivity, data_quality_warning, margin_assumption_aggressive, growth_assumption_aggressive, accounting_risk, regulatory_risk, governance_risk, cyclical_peak_earnings, liquidity_risk, other>",
      "severity": "<critical | high | medium | low>",
      "description": "<string, maks 500 tegn>",
      "affected_fields": ["<string>"]
    }
  ],
  "reasoning": "<string, maks 2000 tegn>"
}"""


def run_skeptic(
    input_data: SkepticInput,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 3,
) -> SkepticOutput:
    """
    Run the Investment Skeptic agent.
    Returns SkepticOutput with veto decision.
    On persistent failure, returns VETO_CASH (fail-safe).
    """
    # Deterministic pre-checks (bypass LLM entirely)
    pre_veto = _deterministic_pre_check(input_data)
    if pre_veto is not None:
        logger.info(f"Skeptic pre-check veto for {input_data.ticker}: {pre_veto.veto}")
        return pre_veto

    user_content = json.dumps(input_data.model_dump(), indent=2, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw_text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            output = SkepticOutput.model_validate_json(raw_text)

            # Post-LLM deterministic enforcement: can only be MORE conservative
            return _enforce_conservatism(output, input_data)

        except Exception as e:
            logger.warning(f"Skeptic attempt {attempt + 1} failed: {e}")

    # All retries failed → fail-safe to CASH
    logger.error(
        f"Skeptic failed all retries for {input_data.ticker}, defaulting to VETO_CASH"
    )
    return SkepticOutput(
        ticker=input_data.ticker,
        veto=VetoAction.VETO_CASH,
        confidence=0.0,
        reasoning="Agent failed after max retries — fail-safe to CASH",
        risk_findings=[
            RiskFinding(
                finding_id="RF-FAILSAFE",
                category="other",
                severity="critical",
                description="Agent execution failure",
            )
        ],
    )


def _deterministic_pre_check(inp: SkepticInput) -> Optional[SkepticOutput]:
    """
    Hard-coded veto rules that bypass LLM entirely.
    These are the same rules the LLM is instructed to follow,
    but enforced deterministically as a safety net.
    """
    findings = []

    # Terminal value dominance
    if inp.terminal_value_pct is not None and inp.terminal_value_pct > 0.60:
        findings.append(
            RiskFinding(
                finding_id="RF-TV001",
                category="terminal_value_dominance",
                severity="critical",
                description=f"Terminal value er {inp.terminal_value_pct:.0%} av DCF",
            )
        )

    # Sensitivity flip
    if (
        inp.sensitivity_wacc_plus_1pct_mos is not None
        and inp.sensitivity_wacc_plus_1pct_mos < 0
    ):
        findings.append(
            RiskFinding(
                finding_id="RF-SENS001",
                category="dcf_sensitivity",
                severity="critical",
                description="MOS blir negativ med +1% WACC",
            )
        )

    # DQ warnings
    if inp.dq_warn_count >= 3:
        findings.append(
            RiskFinding(
                finding_id="RF-DQ001",
                category="data_quality_warning",
                severity="critical",
                description=f"Datakvalitet har {inp.dq_warn_count} advarsler",
                affected_fields=inp.dq_warn_rules.split(";") if inp.dq_warn_rules else [],
            )
        )

    # Quality weak
    if inp.quality_weak_count >= 2:
        findings.append(
            RiskFinding(
                finding_id="RF-QUAL001",
                category="other",
                severity="critical",
                description=f"Kvalitetssvakheter: {inp.quality_weak_count}",
            )
        )

    if findings:
        return SkepticOutput(
            ticker=inp.ticker,
            veto=VetoAction.VETO_CASH,
            confidence=0.95,
            reasoning="Deterministisk pre-sjekk utløste automatisk veto",
            risk_findings=findings,
        )
    return None


def _enforce_conservatism(
    output: SkepticOutput,
    input_data: SkepticInput,
) -> SkepticOutput:
    """
    Post-LLM enforcement: output kan kun være like konservativ eller MER konservativ
    enn hva deterministiske regler tilsier. Aldri mindre.
    """
    pre = _deterministic_pre_check(input_data)
    if pre is not None and output.veto == VetoAction.PASS:
        logger.warning(
            f"LLM sa PASS men deterministiske regler sier VETO for {input_data.ticker}. "
            f"Overstyrer til VETO_CASH."
        )
        output.veto = VetoAction.VETO_CASH
        output.reasoning = output.reasoning + " [OVERRIDE: deterministisk pre-sjekk tvinger VETO]"
    return output
