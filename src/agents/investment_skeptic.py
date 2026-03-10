"""
Agent B: Investment Skeptic (hard-veto).
Adversarial review — tries to falsify the investment case.
Can only produce PASS, VETO_CASH, or REQUEST_REVIEW.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from src.agents.schemas import RiskFinding, SkepticInput, SkepticOutput, VetoAction

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Investment Skeptic for the Deecon investment system.

ROLE: You are a professional risk officer conducting an adversarial review of an
investment candidate. Your job is to find WEAKNESSES and REASONS NOT TO INVEST.

CONSTRAINTS (NON-NEGOTIABLE):
1. You can ONLY output PASS, VETO_CASH, or REQUEST_REVIEW. You CANNOT recommend buying.
2. You must NOT create, estimate, or compute any financial numbers.
3. You must NOT override or question the system's valuation model.
4. You MUST cite specific data points from the input when making findings.
5. When in doubt, VETO. False negatives (missing a real risk) are far worse than
   false positives (vetoing a good stock).

AUTOMATIC VETO TRIGGERS (always veto to CASH if any apply):
- Terminal value > 60% of total DCF value
- ±1% WACC change flips MOS from positive to negative
- Data quality warnings count >= 3
- Quality weak count >= 2

VETO TO REQUEST_REVIEW when:
- Terminal value 40-60% of DCF (borderline)
- One specific data verification would resolve uncertainty

OUTPUT: Respond ONLY with valid JSON matching the required schema."""


def _deterministic_pre_check(inp: SkepticInput) -> Optional[SkepticOutput]:
    """
    Hard-coded veto rules that bypass LLM entirely.
    These are the same rules the LLM is instructed to follow,
    but enforced deterministically as a safety net.
    """
    findings: list[RiskFinding] = []

    # Terminal value dominance
    if inp.terminal_value_pct is not None and inp.terminal_value_pct > 0.60:
        findings.append(RiskFinding(
            finding_id="RF-TV001",
            category="terminal_value_dominance",
            severity="critical",
            description=f"Terminal value er {inp.terminal_value_pct:.0%} av DCF — over 60%-grensen",
        ))

    # Sensitivity flip: +1% WACC gir negativ MOS
    if (
        inp.sensitivity_wacc_plus_1pct_mos is not None
        and inp.sensitivity_wacc_plus_1pct_mos < 0
    ):
        findings.append(RiskFinding(
            finding_id="RF-SENS001",
            category="dcf_sensitivity",
            severity="critical",
            description=(
                f"MOS blir negativ ({inp.sensitivity_wacc_plus_1pct_mos:.1%}) "
                f"ved +1% WACC — verdsettelsen er for sensitiv"
            ),
        ))

    # DQ warnings
    if inp.dq_warn_count >= 3:
        findings.append(RiskFinding(
            finding_id="RF-DQ001",
            category="data_quality_warning",
            severity="critical",
            description=f"Datakvalitet har {inp.dq_warn_count} advarsler: {inp.dq_warn_rules}",
            affected_fields=inp.dq_warn_rules.split(";") if inp.dq_warn_rules else [],
        ))

    # Quality weak
    if inp.quality_weak_count >= 2:
        findings.append(RiskFinding(
            finding_id="RF-QUAL001",
            category="governance_risk",
            severity="critical",
            description=f"Forretningskvalitet har {inp.quality_weak_count} svake indikatorer",
        ))

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
    Post-LLM enforcement: output can only be as conservative or MORE conservative
    than what deterministic rules suggest. Never less.
    """
    pre = _deterministic_pre_check(input_data)
    if pre is not None and output.veto == VetoAction.PASS:
        logger.warning(
            f"LLM sa PASS men deterministiske regler sier VETO for {input_data.ticker}. "
            f"Overstyrer til VETO_CASH."
        )
        output = output.model_copy(update={
            "veto": VetoAction.VETO_CASH,
            "reasoning": output.reasoning + " [OVERRIDE: deterministisk pre-sjekk tvinger VETO]",
        })
    return output


def run_skeptic(
    input_data: SkepticInput,
    client: Any,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 3,
) -> SkepticOutput:
    """
    Run the Investment Skeptic agent.
    Returns SkepticOutput with veto decision.
    On persistent failure, returns VETO_CASH (fail-safe).
    """
    # Deterministisk pre-sjekk (bypass LLM helt)
    pre_veto = _deterministic_pre_check(input_data)
    if pre_veto is not None:
        logger.info(f"Skeptic pre-check veto for {input_data.ticker}: {pre_veto.veto}")
        return pre_veto

    user_content = json.dumps(input_data.model_dump(), indent=2, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            text = response.content[0].text
            output = SkepticOutput.model_validate_json(text)
            return _enforce_conservatism(output, input_data)
        except Exception as e:
            logger.warning(f"Skeptic forsøk {attempt + 1} feilet: {e}")

    # Alle forsøk feilet → fail-safe til CASH
    logger.error(
        f"Skeptic feilet alle {max_retries} forsøk for {input_data.ticker}, "
        f"bruker VETO_CASH som fail-safe"
    )
    return SkepticOutput(
        ticker=input_data.ticker,
        veto=VetoAction.VETO_CASH,
        confidence=0.0,
        reasoning=f"Agent feilet etter {max_retries} forsøk — fail-safe til CASH",
        risk_findings=[
            RiskFinding(
                finding_id="RF-FAILSAFE",
                category="other",
                severity="critical",
                description="Agent execution failure",
            )
        ],
    )
