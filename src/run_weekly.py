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
        return 0

    try:
        ctx = build_run_context(args.asof, args.config, args.run_dir)
        log = setup_logger(ctx.run_dir)

        log.info("Run init OK")
        log.info(f"asof={ctx.asof} run_id={ctx.run_id}")
        log.info(f"config={ctx.config_path}")
        log.info(f"run_dir={ctx.run_dir}")

        if args.dry_run:
            log.info("dry-run: stopping here (no pipeline executed).")
            return 0

        requested = resolve_steps_arg(args.steps)
        available = {name: module for name, module in DEFAULT_STEPS}

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
        return 0

    except AppError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.exit_code
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
