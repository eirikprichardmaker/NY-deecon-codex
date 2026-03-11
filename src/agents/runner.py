"""
Agent orchestrator: kaller agentene sekvensielt på shortlisted kandidater.
Brukes av src/run_weekly.py.
"""
import json
import logging
import os
from pathlib import Path

import anthropic
import pandas as pd

from src.agents.investment_skeptic import run_skeptic
from src.agents.schemas import (
    DossierInput,
    SkepticInput,
    SkepticOutput,
    VetoAction,
)

logger = logging.getLogger(__name__)


def _make_anthropic_client() -> anthropic.Anthropic:
    """Opprett Anthropic-klient fra miljøvariabel ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY ikke satt. "
            "Sett miljøvariabelen eller deaktiver agenter i agent_config.yaml."
        )
    return anthropic.Anthropic(api_key=api_key)


def _build_skeptic_input(row: pd.Series, valuation_row: pd.Series | None, asof: str) -> SkepticInput:
    """Bygg SkepticInput fra pipeline-rader."""
    val = valuation_row if valuation_row is not None else pd.Series(dtype=float)

    def _get(series: pd.Series, key: str, default=None):
        try:
            v = series.get(key, default)
            return None if pd.isna(v) else v
        except Exception:
            return default

    return SkepticInput(
        ticker=str(row.get("ticker", "")),
        market=str(row.get("market", "OSE")),
        asof_date=asof,
        intrinsic_value=float(_get(val, "intrinsic_value", 0) or 0),
        current_price=float(_get(row, "adj_close", 0) or 0),
        mos=float(_get(val, "mos", 0) or 0),
        wacc_used=float(_get(val, "wacc_used", 0.09) or 0.09),
        terminal_growth=float(_get(val, "terminal_growth", 0.02) or 0.02),
        terminal_value_pct=_get(val, "terminal_value_pct"),
        sensitivity_wacc_plus_1pct_mos=_get(val, "sensitivity_wacc_plus_1pct_mos"),
        sensitivity_wacc_minus_1pct_mos=_get(val, "sensitivity_wacc_minus_1pct_mos"),
        roic_current=_get(row, "roic_current"),
        fcf_yield=_get(row, "fcf_yield"),
        nd_ebitda=_get(row, "nd_ebitda"),
        dq_warn_count=int(_get(row, "dq_warn_count_v2", 0) or 0),
        dq_warn_rules=str(_get(row, "dq_warn_rules", "") or ""),
        quality_weak_count=int(_get(row, "quality_weak_count", 0) or 0),
        value_creation_ok=bool(_get(row, "value_creation_ok", True)),
    )


def run_skeptic_on_shortlist(
    shortlist_df: pd.DataFrame,
    valuation_df: pd.DataFrame | None,
    run_dir: Path,
    agent_cfg: dict,
) -> dict[str, SkepticOutput]:
    """
    Kjør Investment Skeptic på shortlisted kandidater.
    Returnerer dict ticker -> SkepticOutput.
    Skriv resultater til run_dir/skeptic_results.json.
    """
    skeptic_cfg = agent_cfg.get("investment_skeptic", {})
    model = agent_cfg.get("model", "claude-sonnet-4-6")
    fallback_model = agent_cfg.get("fallback_model", "claude-haiku-4-5-20251001")
    max_retries = int(agent_cfg.get("max_retries", 3))
    asof = agent_cfg.get("asof_date", "unknown")
    max_tickers = int(
        agent_cfg.get("dossier_writer", {}).get("cost_control", {}).get("max_tickers_to_analyze", 5)
    )

    if shortlist_df.empty:
        logger.info("Ingen kandidater på shortlist — hopper over skeptiker-kjøring.")
        return {}

    # Begrens til topp N kandidater (sorter på MOS fallback til ticker)
    if "mos" in shortlist_df.columns:
        candidates = shortlist_df.nlargest(max_tickers, "mos")
    else:
        candidates = shortlist_df.head(max_tickers)

    try:
        client = _make_anthropic_client()
    except EnvironmentError as e:
        logger.error(f"Kan ikke starte Anthropic-klient: {e}")
        # Fail-safe: veto alle kandidater
        results = {}
        for _, row in candidates.iterrows():
            ticker = str(row.get("ticker", "UNKNOWN"))
            results[ticker] = SkepticOutput(
                ticker=ticker,
                veto=VetoAction.VETO_CASH,
                confidence=0.0,
                reasoning=f"Anthropic-klient utilgjengelig: {e}",
            )
        return results

    results: dict[str, SkepticOutput] = {}

    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker", "UNKNOWN"))
        val_row = None
        if valuation_df is not None and not valuation_df.empty:
            matches = valuation_df[valuation_df["ticker"] == ticker]
            if not matches.empty:
                val_row = matches.iloc[0]

        skeptic_input = _build_skeptic_input(row, val_row, asof)

        try:
            output = run_skeptic(
                input_data=skeptic_input,
                client=client,
                model=model,
                max_retries=max_retries,
            )
        except Exception as e:
            logger.error(f"Skeptic krasjet for {ticker}: {e}")
            output = SkepticOutput(
                ticker=ticker,
                veto=VetoAction.VETO_CASH,
                confidence=0.0,
                reasoning=f"Uventet feil: {e}",
            )

        results[ticker] = output
        logger.info(f"Skeptic [{ticker}]: {output.veto} (confidence={output.confidence:.2f})")

    # Skriv til fil
    serializable = {t: r.model_dump() for t, r in results.items()}
    (run_dir / "skeptic_results.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False)
    )

    return results
