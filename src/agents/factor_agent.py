"""Faktor & Kvalitetsagent — kjører compute_factors + cost_of_capital.

Krav:  data/processed/master.parquet må eksistere (kjør ingest/transform/master-steg først).
Produserer: master.parquet oppdatert med faktorer + cost-of-capital i data/processed/.

CLI:
    python -m src.agents.factor_agent --asof 2025-12-31 --config config\\config.yaml
    python -m src.agents.factor_agent --asof 2025-12-31 --run-dir runs\\20251231_120000
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.common.config import build_run_context
from src.common.log import setup_logger
import src.compute_factors as compute_factors
import src.cost_of_capital as cost_of_capital


AGENT_NAME = "factor_agent"
DESCRIPTION = "Faktor & Kvalitetsagent: beregner faktorer (quality/value/lowrisk/balance) og kapitalkostnad (WACC/COE)."
REQUIRES = ["data/processed/master.parquet"]
PRODUCES = ["data/processed/master.parquet (med faktorer)"]


def run(asof: str, config: str, run_dir: str | None = None) -> int:
    ctx = build_run_context(asof, config, run_dir)
    log = setup_logger(ctx.run_dir, AGENT_NAME)
    log.info("=" * 60)
    log.info(f"  {DESCRIPTION}")
    log.info("=" * 60)
    log.info(f"asof={ctx.asof}  run_dir={ctx.run_dir}")

    log.info("--- Steg 1/2: compute_factors ---")
    ret = compute_factors.run(ctx, log)
    if ret:
        log.error(f"compute_factors feiler med kode {ret}")
        return ret

    log.info("--- Steg 2/2: cost_of_capital ---")
    ret = cost_of_capital.run(ctx, log)
    if ret:
        log.error(f"cost_of_capital feiler med kode {ret}")
        return ret

    log.info("=== Faktor & Kvalitetsagent FERDIG ===")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument("--asof", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", default=r"config\config.yaml")
    p.add_argument("--run-dir", default=None, help="Eksisterende run-mappe (valgfritt)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    sys.exit(run(args.asof, args.config, args.run_dir))


if __name__ == "__main__":
    main()
