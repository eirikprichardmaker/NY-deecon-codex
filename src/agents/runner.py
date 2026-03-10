"""
Agent runner: orchestrator som kaller agentene sekvensielt på shortlist.
Lesbar av run_weekly via decision.py — ingen direkte pipeline-avhengighet.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.agents.schemas import (
    QualityInput, QualityOutput,
    RiskFinding, SkepticInput, SkepticOutput, VetoAction,
)

logger = logging.getLogger(__name__)

# Fallback-output når agenten er deaktivert eller feiler
_BYPASS_OUTPUT_CACHE: dict[str, SkepticOutput] = {}


def _make_bypass_output(ticker: str) -> SkepticOutput:
    """Returnerer PASS uten LLM-kall — brukes når agents.enabled=false."""
    if ticker not in _BYPASS_OUTPUT_CACHE:
        _BYPASS_OUTPUT_CACHE[ticker] = SkepticOutput(
            ticker=ticker,
            veto=VetoAction.PASS,
            confidence=1.0,
            reasoning="Agent bypass: agents.enabled=false i konfigurasjon",
        )
    return _BYPASS_OUTPUT_CACHE[ticker]


def _to_float(val: Any) -> Optional[float]:
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _build_skeptic_input(row: pd.Series, asof: str) -> SkepticInput:
    """Bygger SkepticInput fra en rad i eligible-df."""
    return SkepticInput(
        ticker=str(row.get("ticker", "UNKNOWN")),
        market=str(row.get("relevant_index_key", "OSE")),
        asof_date=asof,
        intrinsic_value=_to_float(row.get("intrinsic_value")) or 0.0,
        current_price=_to_float(row.get("adj_close")) or 0.0,
        mos=_to_float(row.get("mos")) or 0.0,
        wacc_used=_to_float(row.get("wacc_used")) or 0.09,
        terminal_growth=_to_float(row.get("terminal_growth")) or 0.02,
        terminal_value_pct=_to_float(row.get("terminal_value_pct")),
        sensitivity_wacc_plus_1pct_mos=_to_float(row.get("sensitivity_wacc_plus_1pct_mos")),
        sensitivity_wacc_minus_1pct_mos=_to_float(row.get("sensitivity_wacc_minus_1pct_mos")),
        roic_current=_to_float(row.get("roic_current")),
        fcf_yield=_to_float(row.get("fcf_yield")),
        nd_ebitda=_to_float(row.get("nd_ebitda")),
        dq_warn_count=int(row.get("dq_warn_count_v2", row.get("data_quality_warn_count", 0)) or 0),
        dq_warn_rules=str(row.get("dq_warn_rules", "") or ""),
        quality_weak_count=int(row.get("quality_weak_count", 0) or 0),
        value_creation_ok=bool(row.get("value_creation_ok", True)),
    )


def _get_anthropic_client(agent_cfg: dict) -> Any:
    """Oppretter Anthropic-klient. Kaster ImportError hvis anthropic ikke er installert."""
    from anthropic import Anthropic  # lazy import — ikke tilgjengelig i testmiljø
    api_key = agent_cfg.get("api_key")
    return Anthropic(api_key=api_key) if api_key else Anthropic()


def run_skeptic_on_shortlist(
    shortlist_df: pd.DataFrame,
    run_dir: Path,
    agent_cfg: dict,
    asof: str = "",
    *,
    _skeptic_fn: Optional[Callable] = None,   # injection-punkt for tester
) -> dict[str, SkepticOutput]:
    """
    Kjører Investment Skeptic på topp-N kandidater fra shortlist.

    Args:
        shortlist_df: DataFrame med eligible tickers (fundamental_ok & technical_ok).
        run_dir:      Run-katalog for å skrive skeptic_results.json.
        agent_cfg:    Agent-konfigurasjon fra config (agents-seksjonen).
        asof:         As-of-dato (YYYY-MM-DD).
        _skeptic_fn:  Injisert skeptic-funksjon for testing (default: run_skeptic).

    Returns:
        dict[ticker -> SkepticOutput]
    """
    results: dict[str, SkepticOutput] = {}

    # Sjekk om agenter er aktivert globalt
    if not agent_cfg.get("enabled", False):
        logger.info("runner: agents.enabled=false — bypass alle agenter")
        for _, row in shortlist_df.iterrows():
            ticker = str(row.get("ticker", "UNKNOWN"))
            results[ticker] = _make_bypass_output(ticker)
        _write_results(run_dir, results)
        return results

    # Sjekk om Investment Skeptic spesifikt er aktivert
    skeptic_cfg = agent_cfg.get("investment_skeptic", {}) or {}
    if not skeptic_cfg.get("enabled", False):
        logger.info("runner: investment_skeptic.enabled=false — bypass skeptic")
        for _, row in shortlist_df.iterrows():
            ticker = str(row.get("ticker", "UNKNOWN"))
            results[ticker] = _make_bypass_output(ticker)
        _write_results(run_dir, results)
        return results

    # Begrens antall tickers som analyseres
    cost_cfg = agent_cfg.get("cost_control", {}) or {}
    max_tickers = int(cost_cfg.get("max_tickers_to_analyze", 5))
    candidates = shortlist_df.head(max_tickers)

    if candidates.empty:
        logger.info("runner: ingen kandidater å analysere")
        _write_results(run_dir, results)
        return results

    # Hent skeptic-funksjon (injisert i tester, ellers ekte)
    if _skeptic_fn is None:
        from src.agents.investment_skeptic import run_skeptic as _real_skeptic
        _skeptic_fn = _real_skeptic
        # Hent LLM-klient kun når ekte skeptic-funksjon brukes
        try:
            client = _get_anthropic_client(agent_cfg)
        except (ImportError, Exception) as exc:
            logger.warning(f"runner: kunne ikke opprette Anthropic-klient: {exc} — bruker client=None")
            client = None
    else:
        # Injisert _skeptic_fn i tester — klient ikke nødvendig
        client = None

    model = agent_cfg.get("model", "claude-sonnet-4-6")
    max_retries = int(agent_cfg.get("max_retries", 3))

    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker", "UNKNOWN"))
        try:
            inp = _build_skeptic_input(row, asof)
            result = _skeptic_fn(inp, client=client, model=model, max_retries=max_retries)
            results[ticker] = result
            log_level = logging.WARNING if result.veto == VetoAction.VETO_CASH else logging.INFO
            logger.log(log_level, f"runner: {ticker} → {result.veto.value} (confidence={result.confidence:.2f})")
        except Exception as exc:
            logger.error(f"runner: feil ved analyse av {ticker}: {exc}")
            results[ticker] = SkepticOutput(
                ticker=ticker,
                veto=VetoAction.VETO_CASH,
                confidence=0.0,
                reasoning=f"Runner-feil: {exc}",
                risk_findings=[RiskFinding(
                    finding_id="RF-RUNNER-ERR",
                    category="other",
                    severity="critical",
                    description=f"Uventet feil i runner: {exc}",
                )],
            )

    _write_results(run_dir, results)
    return results


def _write_results(run_dir: Path, results: dict[str, SkepticOutput]) -> None:
    """Skriver skeptic_results.json til run_dir."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {ticker: r.model_dump() for ticker, r in results.items()}
        (run_dir / "skeptic_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
    except Exception as exc:
        logger.warning(f"runner: kunne ikke skrive skeptic_results.json: {exc}")


def _build_quality_input(row: pd.Series, asof: str) -> QualityInput:
    """Bygger QualityInput fra en rad i shortlist-df."""
    snap = {
        k: _to_float(row.get(k))
        for k in ("roic_current", "fcf_yield", "nd_ebitda", "mos", "wacc_used")
        if _to_float(row.get(k)) is not None
    }
    return QualityInput(
        ticker=str(row.get("ticker", "UNKNOWN")),
        market=str(row.get("relevant_index_key", "OSE")),
        asof_date=asof,
        text_evidence=[],   # pipeline kan injisere IR-rapporttekst her
        numeric_snapshot=snap,
    )


def _write_quality_results(run_dir: Path, results: dict[str, QualityOutput]) -> None:
    """Skriver quality_results.json til run_dir."""
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {ticker: r.model_dump() for ticker, r in results.items()}
        (run_dir / "quality_results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
    except Exception as exc:
        logger.warning(f"runner: kunne ikke skrive quality_results.json: {exc}")


def run_quality_on_shortlist(
    shortlist_df: pd.DataFrame,
    run_dir: Path,
    agent_cfg: dict,
    asof: str = "",
    *,
    _quality_fn: Optional[Callable] = None,   # injection-punkt for tester
) -> dict[str, QualityOutput]:
    """
    Kjører Business Quality Evaluator på topp-N kandidater fra shortlist.

    Args:
        shortlist_df: DataFrame med eligible tickers.
        run_dir:      Run-katalog for å skrive quality_results.json.
        agent_cfg:    Agent-konfigurasjon fra config (agents-seksjonen).
        asof:         As-of-dato (YYYY-MM-DD).
        _quality_fn:  Injisert quality-funksjon for testing.

    Returns:
        dict[ticker -> QualityOutput]
    """
    results: dict[str, QualityOutput] = {}

    if not agent_cfg.get("enabled", False):
        logger.info("runner: agents.enabled=false — bypass quality evaluator")
        _write_quality_results(run_dir, results)
        return results

    quality_cfg = agent_cfg.get("business_quality_evaluator", {}) or {}
    if not quality_cfg.get("enabled", False):
        logger.info("runner: business_quality_evaluator.enabled=false — bypass")
        _write_quality_results(run_dir, results)
        return results

    cost_cfg = agent_cfg.get("cost_control", {}) or {}
    max_tickers = int(cost_cfg.get("max_tickers_to_analyze", 5))
    candidates = shortlist_df.head(max_tickers)

    if candidates.empty:
        _write_quality_results(run_dir, results)
        return results

    if _quality_fn is None:
        from src.agents.business_quality_evaluator import run_quality_evaluator as _real_quality
        _quality_fn = _real_quality
        try:
            client = _get_anthropic_client(agent_cfg)
        except (ImportError, Exception) as exc:
            logger.warning(f"runner: quality: kunne ikke opprette klient: {exc}")
            client = None
    else:
        client = None

    model = agent_cfg.get("model", "claude-sonnet-4-6")

    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker", "UNKNOWN"))
        try:
            inp = _build_quality_input(row, asof)
            result = _quality_fn(inp, client=client, model=model)
            results[ticker] = result
            logger.info(
                f"runner: quality {ticker} → {result.quality_verdict} / {result.veto.value}"
            )
        except Exception as exc:
            logger.error(f"runner: quality feil for {ticker}: {exc}")
            results[ticker] = QualityOutput(
                ticker=ticker,
                quality_verdict="unknown",
                veto=VetoAction.REQUEST_REVIEW,
                confidence=0.0,
                flags=[f"Runner-feil: {str(exc)[:200]}"],
            )

    _write_quality_results(run_dir, results)
    return results
