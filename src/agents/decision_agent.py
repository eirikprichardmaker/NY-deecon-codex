"""Beslutnings-agent — screening, rangering og endelig beslutning (aksje eller CASH).

Krav:  data/processed/master_valued.parquet + runs/<run_id>/valuation.csv
       (kjor valuation_agent forst).
Produserer:
  - runs/<run_id>/screen_basic.csv   (alle selskaper med gate-resultat)
  - runs/<run_id>/shortlist.csv      (kandidater som passerer alle filtre)
  - runs/<run_id>/decision.csv       (endelig beslutning)
  - runs/<run_id>/decision.md        (lesbar rapport)
  - runs/<run_id>/top_candidates.md

CLI:
    python -m src.agents.decision_agent --asof 2025-12-31 --config config\\config.yaml
    python -m src.agents.decision_agent --asof 2025-12-31 --run-dir runs\\20251231_120000
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.common.config import build_run_context
from src.common.log import setup_logger
import src.decision as decision


AGENT_NAME = "decision_agent"
DESCRIPTION = (
    "Beslutnings-agent: screening (MOS/kvalitet/teknisk), rangering og "
    "endelig valg — maks 1 aksje ellers CASH."
)
REQUIRES = [
    "data/processed/master_valued.parquet",
    "runs/<run_id>/valuation.csv",
]
PRODUCES = [
    "runs/<run_id>/screen_basic.csv",
    "runs/<run_id>/shortlist.csv",
    "runs/<run_id>/decision.csv",
    "runs/<run_id>/decision.md",
]


def run(asof: str, config: str, run_dir: str | None = None) -> int:
    ctx = build_run_context(asof, config, run_dir)
    log = setup_logger(ctx.run_dir, AGENT_NAME)
    log.info("=" * 60)
    log.info(f"  {DESCRIPTION}")
    log.info("=" * 60)
    log.info(f"asof={ctx.asof}  run_dir={ctx.run_dir}")

    log.info("--- Kjorer decision (screening + ranking + pick) ---")
    ret = decision.run(ctx, log)
    if ret:
        log.error(f"decision feiler med kode {ret}")
        return ret

    log.info("=== Beslutnings-agent FERDIG ===")
    log.info(f"Beslutning: {ctx.run_dir / 'decision.md'}")
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
