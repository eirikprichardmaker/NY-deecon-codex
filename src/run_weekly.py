# src/run_weekly.py
from __future__ import annotations

import argparse
import sys
import importlib
from typing import Callable, List, Tuple

from src.common.config import build_run_context
from src.common.log import setup_logger
from src.common.errors import AppError


Step = Tuple[str, str]  # (step_name, module_path)


DEFAULT_STEPS: List[Step] = [
    ("ingest", "src.ingest_validate"),
    ("transform_fundamentals", "src.transform_fundamentals"),
    ("freeze_golden", "src.freeze_golden_fundamentals"),
    ("transform_prices", "src.transform_prices"),
    ("master", "src.build_master"),
    ("factors", "src.compute_factors"),
    ("cost_of_capital", "src.cost_of_capital"),
    ("valuation", "src.valuation"),
    ("decision", "src.decision"),
]

OPTIONAL_STEPS: List[Step] = [
    ("ir_reports", "src.ir_reports"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--asof", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", default=r"config\config.yaml")
    p.add_argument("--run-dir", default=None, help="Override run directory (optional)")
    p.add_argument("--dry-run", action="store_true", help="Init run-dir + manifest + log only")
    p.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated list of steps to run, in order. "
            "Use 'all' (default) or e.g. 'ingest,transform_fundamentals,transform_prices,master'"
        ),
    )
    p.add_argument(
        "--list-steps",
        action="store_true",
        help="Print available steps and exit",
    )
    return p.parse_args()


def resolve_steps_arg(steps_arg: str) -> List[str]:
    s = (steps_arg or "").strip().lower()
    if s in ("all", "*"):
        return [name for name, _ in DEFAULT_STEPS]
    return [x.strip() for x in s.split(",") if x.strip()]


def import_step_runner(module_path: str) -> Callable:
    """
    Contract:
      Each module must expose: run(ctx, log) -> None/int
    """
    mod = importlib.import_module(module_path)
    runner = getattr(mod, "run", None)
    if runner is None or not callable(runner):
        raise RuntimeError(
            f"Module '{module_path}' is missing callable 'run(ctx, log)'. "
            f"Add e.g.\n\n"
            f"def run(ctx, log):\n"
            f"    ...\n"
        )
    return runner


def main() -> int:
    args = parse_args()

    if args.list_steps:
        for name, module in DEFAULT_STEPS:
            print(f"{name}: {module}")
        for name, module in OPTIONAL_STEPS:
            print(f"{name}: {module} (optional)")
        return 0

    try:
        ctx = build_run_context(args.asof, args.config, args.run_dir)
        log = setup_logger(ctx.run_dir)

        log.info("Run init OK")
        log.info(f"asof={ctx.asof} run_id={ctx.run_id}")
        log.info(f"config={ctx.config_path}")
        log.info(f"run_dir={ctx.run_dir}")

        try:
            from src.manifest import write_manifest
            write_manifest(
                run_dir=ctx.run_dir,
                asof=ctx.asof,
                config_path=ctx.config_path,
                seed=int(ctx.cfg.get("seed", 0)),
                ticker_count=0,  # ukjent på dette tidspunktet — oppdateres ikke i v1
                data_sources=list(ctx.cfg.get("paths", {}).keys()),
                agent_config=ctx.cfg.get("agents"),
            )
            log.info("manifest: wrote manifest.json")
        except Exception as _manifest_exc:
            log.warning(f"manifest: kunne ikke skrive manifest (ikke-kritisk): {_manifest_exc}")

        if args.dry_run:
            log.info("dry-run: stopping here (no pipeline executed).")
            return 0

        requested = resolve_steps_arg(args.steps)
        available = {name: module for name, module in (DEFAULT_STEPS + OPTIONAL_STEPS)}

        # validate requested steps
        unknown = [s for s in requested if s not in available]
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}. Use --list-steps to see valid names.")

        # run in requested order
        for step_name in requested:
            module_path = available[step_name]
            log.info(f"STEP START: {step_name} ({module_path})")

            runner = import_step_runner(module_path)
            out = runner(ctx, log)

            # allow run() to return 0/None for success
            if isinstance(out, int) and out != 0:
                log.error(f"STEP FAIL: {step_name} returned {out}")
                return out

            log.info(f"STEP OK: {step_name}")

        log.info("PIPELINE OK")

        # --- Business Quality Evaluator (Agent A) ---
        # Kjøres etter alle steg. business_quality_evaluator.enabled=false → skip.
        try:
            _agent_cfg_q = ctx.cfg.get("agents") or {}
            _quality_cfg = _agent_cfg_q.get("business_quality_evaluator", {}) or {}
            if _agent_cfg_q.get("enabled", False) and _quality_cfg.get("enabled", False):
                import pandas as _pd
                from src.agents.runner import run_quality_on_shortlist

                _shortlist_csv = ctx.run_dir / "valuation.csv"
                if _shortlist_csv.exists():
                    _sq_df = _pd.read_csv(_shortlist_csv)
                    _quality_results = run_quality_on_shortlist(
                        _sq_df, run_dir=ctx.run_dir,
                        agent_cfg=_agent_cfg_q, asof=ctx.asof,
                    )
                    log.info(
                        f"quality: evaluerte {len(_quality_results)} tickers → quality_results.json"
                    )
            else:
                log.info("quality: business_quality_evaluator deaktivert — skip")
        except Exception as _q_exc:
            log.warning(f"quality: feilet (ikke-kritisk): {_q_exc}")

        # --- Dossier Writer (Agent C) ---
        # Kjøres etter alle steg, leser fra run_dir-output.
        # dossier_writer.enabled=false → skip.
        try:
            _agent_cfg = ctx.cfg.get("agents") or {}
            _dossier_cfg = _agent_cfg.get("dossier_writer", {}) or {}
            if _agent_cfg.get("enabled", False) and _dossier_cfg.get("enabled", False):
                import csv
                from src.agents.dossier_writer import build_dossier_input, run_dossier_writer
                from src.agents.runner import _get_openai_client

                decision_csv = ctx.run_dir / "decision.csv"
                if decision_csv.exists():
                    with open(decision_csv, newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    pick = rows[0] if rows else {}
                    ticker = pick.get("ticker", "UNKNOWN")
                    final_decision = pick.get("model", pick.get("decision", "CASH"))
                    valuation_summary = {
                        k: pick.get(k) for k in
                        ["mos", "intrinsic_value", "wacc_used", "market_cap",
                         "adj_close", "quality_score", "roic_wacc_spread"]
                        if pick.get(k)
                    }
                    gate_log = [
                        {"gate": "fundamental_ok", "passed": pick.get("fundamental_ok")},
                        {"gate": "technical_ok",   "passed": pick.get("technical_ok")},
                    ]
                    try:
                        _client = _get_openai_client(_agent_cfg)
                    except (ImportError, Exception):
                        _client = None

                    dossier_input = build_dossier_input(
                        ticker=ticker,
                        asof=ctx.asof,
                        final_decision=final_decision,
                        gate_log=gate_log,
                        valuation_summary=valuation_summary,
                    )
                    dossier = run_dossier_writer(
                        dossier_input,
                        client=_client,
                        model=_agent_cfg.get("model", "gpt-4o"),
                    )
                    (ctx.run_dir / "decision_agent.md").write_text(
                        dossier.narrative, encoding="utf-8"
                    )
                    (ctx.run_dir / "decision_agent.json").write_text(
                        dossier.model_dump_json(indent=2), encoding="utf-8"
                    )
                    log.info(f"dossier: wrote decision_agent.md for {ticker}")
        except Exception as _dossier_exc:
            log.warning(f"dossier: feilet (ikke-kritisk): {_dossier_exc}")

        # --- Output-kontrakt ---
        try:
            from src.output_contract import validate_run_outputs
            _ok, _missing = validate_run_outputs(ctx.run_dir)
            if not _ok:
                log.warning(f"output_contract: manglende filer: {_missing}")
            else:
                log.info("output_contract: alle påkrevde filer er til stede")
        except Exception as _oc_exc:
            log.warning(f"output_contract: validering feilet (ikke-kritisk): {_oc_exc}")

        return 0

    except AppError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.exit_code
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
