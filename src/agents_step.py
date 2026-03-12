"""
Optional pipeline step: run AI agents (Investment Skeptic + Dossier Writer + Business Quality)
on the shortlisted candidates after the deterministic decision step.

Rekkefølge (alle agenter er veto-only):
  1. Business Quality Evaluator (Agent A) — tekst-basert, deaktivert til Fase 4 ferdig
  2. Investment Skeptic (Agent B)          — tall-basert, veto-only
  3. Decision Dossier Writer (Agent C)     — narrativ, ingen veto-rett

Contract:
  run(ctx, log) -> int (0 = success)

Aktiveres via --steps agents i run_weekly.py, eller som selvstendig steg.
Krever: ANTHROPIC_API_KEY miljøvariabel + agents.enabled: true i agent_config.yaml
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.agents.runner import _make_anthropic_client, run_skeptic_on_shortlist
from src.agents.schemas import DossierInput, QualityInput, VetoAction


def _load_agent_cfg(config_path: Path) -> dict:
    agent_cfg_path = config_path.parent / "agent_config.yaml"
    if not agent_cfg_path.exists():
        return {"agents": {"enabled": False}}
    with open(agent_cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run(ctx, log) -> int:
    """
    Pipeline-steg: Kjør AI-agenter på shortlisted kandidater.

    Lese-input:
      - {run_dir}/decision.csv         (shortlist fra decision-steget)
      - {run_dir}/valuation.csv        (verdsettelsesdata)
      - {run_dir}/decision_reasons.json (gate-logg per ticker)

    Skriv-output:
      - {run_dir}/skeptic_results.json
      - {run_dir}/decision_agent.md
      - {run_dir}/decision_agent.json
      - {run_dir}/decision.csv         (oppdatert med skeptiker-veto)
    """
    agent_cfg_full = _load_agent_cfg(ctx.config_path)
    agents_cfg = agent_cfg_full.get("agents", {})

    if not agents_cfg.get("enabled", False):
        log.info("agents_step: agents.enabled=false i agent_config.yaml — hopper over.")
        return 0

    model = agents_cfg.get("model", "claude-sonnet-4-6")
    asof = ctx.asof

    # --- Les decision.csv ---
    decision_csv = ctx.run_dir / "decision.csv"
    if not decision_csv.exists():
        log.warning("agents_step: decision.csv mangler — hopper over.")
        return 0

    decision_df = pd.read_csv(decision_csv)

    # Shortlist = kandidater som passerte fundamental_ok & technical_ok
    shortlist_mask = (
        decision_df.get("fundamental_ok", pd.Series(False, index=decision_df.index)).fillna(False).astype(bool)
        & decision_df.get("technical_ok", pd.Series(False, index=decision_df.index)).fillna(False).astype(bool)
    )
    shortlist_df = decision_df[shortlist_mask].copy()

    if shortlist_df.empty:
        log.info("agents_step: ingen shortlisted kandidater — hopper over agent-kjøring.")
        _write_empty_outputs(ctx.run_dir)
        return 0

    log.info(f"agents_step: {len(shortlist_df)} kandidat(er) fra deterministisk screening.")

    # --- Les valuation.csv ---
    valuation_df = None
    valuation_csv = ctx.run_dir / "valuation.csv"
    if valuation_csv.exists():
        try:
            valuation_df = pd.read_csv(valuation_csv)
        except Exception as e:
            log.warning(f"agents_step: kunne ikke lese valuation.csv: {e}")

    # --- DATA CONTROLLER GATE: blokkerer garbage-in FØR agentene kalles ---
    from src.agents.data_controller import check_shortlist
    shortlist_df, controller_results = check_shortlist(shortlist_df, valuation_df)
    controller_log = [r.to_dict() for r in controller_results]
    (ctx.run_dir / "data_controller_log.json").write_text(
        json.dumps({"asof": asof, "results": controller_log}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    blocked = [r for r in controller_results if not r.ok]
    approved = [r for r in controller_results if r.ok]
    for r in blocked:
        log.warning(f"agents_step [DataController] BLOKKERT {r.ticker}: {r.blocked_reason}")
    log.info(
        f"agents_step [DataController]: {len(approved)} godkjent, {len(blocked)} blokkert"
    )
    if shortlist_df.empty:
        log.warning("agents_step: alle kandidater blokkert av Data Controller — ingen LLM-kall.")
        _write_empty_outputs(ctx.run_dir)
        return 0

    # Sett asof i agent_cfg slik at runner kan bruke det
    agents_cfg["asof_date"] = asof

    # --- Kjør Business Quality Evaluator (Agent A) — kun hvis aktivert ---
    quality_results: dict = {}
    quality_cfg = agents_cfg.get("business_quality", {})
    if quality_cfg.get("enabled", False):
        log.info("agents_step: starter Business Quality Evaluator (Agent A)...")
        quality_results = _run_quality_on_shortlist(
            shortlist_df=shortlist_df,
            run_dir=ctx.run_dir,
            asof=asof,
            agents_cfg=agents_cfg,
            model=model,
        )
        # Appli quality-veto
        for ticker, qr in quality_results.items():
            if qr.get("veto") == "VETO_CASH":
                mask = decision_df["ticker"].astype(str) == ticker
                if mask.any():
                    decision_df.loc[mask, "decision_reasons"] = (
                        decision_df.loc[mask, "decision_reasons"].fillna("").astype(str)
                        + ";quality_veto"
                    )
                    decision_df.loc[mask, "fundamental_ok"] = False
                    log.info(f"agents_step: quality VETO_CASH for {ticker}")
        (ctx.run_dir / "quality_results.json").write_text(
            json.dumps(quality_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        log.info("agents_step: business_quality.enabled=false — hopper over Agent A.")

    # --- Kjør Investment Skeptic ---
    log.info("agents_step: starter Investment Skeptic...")
    try:
        skeptic_results = run_skeptic_on_shortlist(
            shortlist_df=shortlist_df,
            valuation_df=valuation_df,
            run_dir=ctx.run_dir,
            agent_cfg=agents_cfg,
        )
    except Exception as e:
        log.error(f"agents_step: Skeptic krasjet: {e}")
        skeptic_results = {}

    # --- Appli veto: VETO_CASH overstyrer KANDIDAT i decision.csv ---
    veto_count = 0
    for ticker, result in skeptic_results.items():
        if result.veto == VetoAction.VETO_CASH:
            mask = decision_df["ticker"].astype(str) == ticker
            if mask.any():
                decision_df.loc[mask, "decision_reasons"] = (
                    decision_df.loc[mask, "decision_reasons"].fillna("").astype(str)
                    + ";skeptic_veto"
                )
                decision_df.loc[mask, "fundamental_ok"] = False
                veto_count += 1
                log.info(f"agents_step: VETO_CASH for {ticker} — overstyrer til CASH")

    if veto_count > 0:
        decision_df.to_csv(decision_csv, index=False)
        log.info(f"agents_step: {veto_count} ticker(e) endret til CASH via skeptiker-veto")

    # --- Kjør Dossier Writer på kandidater (etter veto) ---
    dossier_cfg = agents_cfg.get("dossier_writer", {})
    if not dossier_cfg.get("enabled", True):
        log.info("agents_step: dossier_writer deaktivert.")
        return 0

    # Les gate-logg
    gate_log = _load_gate_log(ctx.run_dir, ticker=str(final_candidates.iloc[0].get("ticker", "")) if not final_candidates.empty else None)

    # Finn kandidater etter veto
    final_candidates = decision_df[
        decision_df.get("fundamental_ok", pd.Series(False)).fillna(False).astype(bool)
        & decision_df.get("technical_ok", pd.Series(False)).fillna(False).astype(bool)
    ]

    if final_candidates.empty:
        log.info("agents_step: ingen kandidater etter veto — skriver CASH-rapport.")
        _write_cash_dossier(ctx.run_dir, asof)
        return 0

    # Kjør dossier for topp kandidat
    try:
        from src.agents.dossier_writer import run_dossier_writer

        client = _make_anthropic_client()
        pick = final_candidates.iloc[0]
        ticker = str(pick.get("ticker", ""))

        val_summary = _build_valuation_summary(pick, valuation_df, ticker)
        skeptic_out = skeptic_results.get(ticker)
        quality_raw = quality_results.get(ticker)
        quality_out = None
        if quality_raw:
            try:
                from src.agents.schemas import QualityOutput
                quality_out = QualityOutput.model_validate(quality_raw)
            except Exception:
                pass

        dossier_input = DossierInput(
            ticker=ticker,
            market=str(pick.get("relevant_index_symbol", "OSE")),
            asof_date=asof,
            final_decision="KANDIDAT",
            gate_log=gate_log,
            valuation_summary=val_summary,
            skeptic_output=skeptic_out,
            quality_output=quality_out,
        )

        log.info(f"agents_step: skriver dossier for {ticker}...")
        dossier = run_dossier_writer(dossier_input, client, model)

        (ctx.run_dir / "decision_agent.md").write_text(dossier.narrative, encoding="utf-8")
        (ctx.run_dir / "decision_agent.json").write_text(
            dossier.model_dump_json(indent=2), encoding="utf-8"
        )
        log.info(f"agents_step: dossier skrevet for {ticker}")

    except Exception as e:
        log.error(f"agents_step: Dossier Writer feilet: {e}")
        _write_cash_dossier(ctx.run_dir, asof)

    return 0


# ---------------------------------------------------------------------------
# Hjelpefunksjoner
# ---------------------------------------------------------------------------

def _load_gate_log(run_dir: Path, ticker: str | None = None) -> list[dict]:
    """Les decision_reasons.json som gate-logg. Filtrer til én ticker hvis oppgitt."""
    path = run_dir / "decision_reasons.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        decisions = data.get("decisions", [])
        if ticker:
            decisions = [d for d in decisions if d.get("ticker") == ticker]
        return decisions
    except Exception:
        return []


def _build_valuation_summary(pick: pd.Series, valuation_df: pd.DataFrame | None, ticker: str) -> dict:
    """Bygg verdsettelsessummary fra pick-rad og valuation_df."""
    summary = {
        "ticker": ticker,
        "market_cap": _safe_float(pick.get("market_cap")),
        "intrinsic_value": _safe_float(pick.get("intrinsic_value")),
        "mos": _safe_float(pick.get("mos")),
        "mos_req": _safe_float(pick.get("mos_req")),
        "wacc_used": _safe_float(pick.get("wacc_used")),
        "roic_wacc_spread": _safe_float(pick.get("roic_wacc_spread")),
        "quality_score": _safe_float(pick.get("quality_score")),
        "adj_close": _safe_float(pick.get("adj_close")),
        "ma200": _safe_float(pick.get("ma200")),
    }
    if valuation_df is not None and not valuation_df.empty:
        val_row = valuation_df[valuation_df["ticker"].astype(str) == ticker]
        if not val_row.empty:
            vr = val_row.iloc[0]
            summary["terminal_growth"] = _safe_float(vr.get("terminal_growth"))
            summary["terminal_value_pct"] = _safe_float(vr.get("terminal_value_pct"))
    return summary


def _safe_float(val) -> float | None:
    try:
        import math
        v = float(val)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None


def _write_empty_outputs(run_dir: Path) -> None:
    """Skriv tomme placeholder-filer for valgfrie agent-output."""
    for fname in ["decision_agent.md", "decision_agent.json", "skeptic_results.json"]:
        p = run_dir / fname
        if not p.exists():
            p.write_text("{}", encoding="utf-8")


def _run_quality_on_shortlist(
    shortlist_df: pd.DataFrame,
    run_dir: Path,
    asof: str,
    agents_cfg: dict,
    model: str,
) -> dict:
    """Kjør Agent A på shortlisted kandidater. Returnerer dict ticker -> output-dict."""
    from src.agents.business_quality import run_quality_evaluator
    from src.agents.evidence_builder import build_evidence_pack

    downloads_dir = Path(agents_cfg.get("downloads_dir", "data/raw/ir_auto"))
    max_tickers = int(
        agents_cfg.get("dossier_writer", {}).get("cost_control", {}).get("max_tickers_to_analyze", 5)
    )
    min_evidence = int(agents_cfg.get("business_quality", {}).get("min_evidence_tokens", 200))

    try:
        client = _make_anthropic_client()
    except EnvironmentError as e:
        logger.error(f"_run_quality_on_shortlist: {e}")
        return {}

    results = {}
    candidates = shortlist_df.head(max_tickers)

    for _, row in candidates.iterrows():
        ticker = str(row.get("ticker", "UNKNOWN"))
        evidence = build_evidence_pack(ticker, asof, downloads_dir)

        numeric_snapshot = {
            "roic_current": _safe_float(row.get("roic_current")),
            "fcf_yield": _safe_float(row.get("fcf_yield")),
            "nd_ebitda": _safe_float(row.get("nd_ebitda")),
            "quality_score": _safe_float(row.get("quality_score")),
            "quality_weak_count": int(row.get("quality_weak_count", 0) or 0),
        }

        q_input = QualityInput(
            ticker=ticker,
            market=str(row.get("relevant_index_symbol", "OSE")),
            asof_date=asof,
            text_evidence=evidence,
            numeric_snapshot=numeric_snapshot,
        )

        try:
            output = run_quality_evaluator(
                q_input, client, model=model, min_evidence_tokens=min_evidence
            )
            results[ticker] = output.model_dump()
            logger.info(f"Quality [{ticker}]: {output.quality_verdict} / {output.veto}")
        except Exception as e:
            logger.error(f"Quality [{ticker}] feilet: {e}")

    return results


def _write_cash_dossier(run_dir: Path, asof: str) -> None:
    """Minimal CASH-rapport når ingen kandidater finnes etter veto."""
    md = f"# Beslutningsrapport {asof}\n\n**Anbefaling: CASH**\n\nIngen kandidater passerte alle filtre inkludert agent-gjennomgang.\n"
    (run_dir / "decision_agent.md").write_text(md, encoding="utf-8")
    payload = {"agent": "decision_dossier_writer", "version": "1.0", "ticker": "N/A",
               "narrative": md, "key_risks": [], "key_strengths": [], "data_quality_summary": ""}
    (run_dir / "decision_agent.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
