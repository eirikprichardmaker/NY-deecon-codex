"""Verdsettelses-agent — kjører DCF/DDM/RIM verdsettelse.

Krav:  data/processed/master.parquet med faktorer (kjør factor_agent først).
Produserer:
  - runs/<run_id>/valuation.csv
  - runs/<run_id>/valuation_sensitivity.csv
  - data/processed/master_valued.parquet

CLI:
    python -m src.agents.valuation_agent --asof 2025-12-31 --config config\\config.yaml
    python -m src.agents.valuation_agent --asof 2025-12-31 --run-dir runs\\20251231_120000
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.common.config import build_run_context
from src.common.log import setup_logger
import src.valuation as valuation


AGENT_NAME = "valuation_agent"
DESCRIPTION = "Verdsettelses-agent: beregner intrinsic value via DCF/DDM/RIM og margin of safety (MOS)."
REQUIRES = ["data/processed/master.parquet (med faktorer)"]
PRODUCES = [
    "runs/<run_id>/valuation.csv",
    "runs/<run_id>/valuation_sensitivity.csv",
    "data/processed/master_valued.parquet",
]


def run(asof: str, config: str, run_dir: str | None = None) -> int:
    ctx = build_run_context(asof, config, run_dir)
    log = setup_logger(ctx.run_dir, AGENT_NAME)
    log.info("=" * 60)
    log.info(f"  {DESCRIPTION}")
    log.info("=" * 60)
    log.info(f"asof={ctx.asof}  run_dir={ctx.run_dir}")

    log.info("--- Kjorer valuation (DCF/DDM/RIM) ---")
    ret = valuation.run(ctx, log)
    if ret:
        log.error(f"valuation feiler med kode {ret}")
        return ret

    log.info("=== Verdsettelses-agent FERDIG ===")
    log.info(f"Output: {ctx.run_dir / 'valuation.csv'}")
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
