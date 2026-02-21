from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RUN_FILES = (
    "optimize_summary.md",
    "best_config.yaml",
    "trials.csv",
    "holdout_results.csv",
)
BASELINE_MARKERS = ("baseline_strategyA.marker", "baseline.marker", ".baseline")


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _to_float(v: Any) -> float:
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _first_valid_number(*vals: Any) -> float:
    for v in vals:
        n = _to_float(v)
        if not math.isnan(n):
            return n
    return math.nan


def _parse_seeds(seeds_raw: str | None) -> list[int]:
    if not seeds_raw:
        return []
    out: list[int] = []
    for tok in str(seeds_raw).split(","):
        s = tok.strip()
        if not s:
            continue
        out.append(int(s))
    return sorted(set(out))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not parse YAML: {path}: {exc}") from exc
    if not isinstance(obj, dict):
        return {}
    return obj


def _format_num(v: Any, decimals: int = 6) -> str:
    n = _to_float(v)
    if math.isnan(n):
        return ""
    return f"{n:.{decimals}f}"


def _md_table(df: pd.DataFrame, columns: list[str]) -> str:
    d = df.copy()
    d = d[columns]
    if d.empty:
        return "_(empty)_"
    lines: list[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in d.iterrows():
        vals: list[str] = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, float):
                vals.append(_format_num(v))
            elif v is None:
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _holdout_metrics_from_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}

    out: dict[str, float] = {}
    if "return" in df.columns:
        r = pd.to_numeric(df["return"], errors="coerce").fillna(0.0)
        out["cumulative_return_net"] = float((1.0 + r).prod() - 1.0)
    if "max_dd" in df.columns:
        dd = pd.to_numeric(df["max_dd"], errors="coerce").dropna()
        if not dd.empty:
            out["max_dd"] = float(dd.min())
    if "turnover" in df.columns:
        trn = pd.to_numeric(df["turnover"], errors="coerce").dropna()
        if not trn.empty:
            out["turnover"] = float(trn.mean())
    if "pct_cash" in df.columns:
        cash = pd.to_numeric(df["pct_cash"], errors="coerce").dropna()
        if not cash.empty:
            out["pct_cash"] = float(cash.mean())
    if "switches" in df.columns:
        sw = pd.to_numeric(df["switches"], errors="coerce").fillna(0.0)
        out["switches"] = float(sw.sum())
    elif "trade_count" in df.columns:
        sw = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0.0)
        out["switches"] = float(sw.sum())
    return out


def _selection_metrics_from_trials(trials_path: Path, trial_id: str | None) -> dict[str, float]:
    if not trials_path.exists():
        return {}
    try:
        df = pd.read_csv(trials_path)
    except Exception:
        return {}
    if df.empty:
        return {}

    d = df.copy()
    if trial_id and "trial_id" in d.columns:
        d = d[d["trial_id"].astype(str) == str(trial_id)].copy()
    if d.empty and "is_selected_train" in df.columns and "trial_id" in df.columns:
        chosen = df[df["is_selected_train"].astype(bool)].copy()
        if not chosen.empty:
            selected_id = str(chosen.iloc[0]["trial_id"])
            d = df[df["trial_id"].astype(str) == selected_id].copy()
    if d.empty:
        d = df.copy()

    out: dict[str, float] = {}
    if "excess_return_net" in d.columns:
        s = pd.to_numeric(d["excess_return_net"], errors="coerce").dropna()
        if not s.empty:
            out["median_excess_return_net"] = float(s.median())
            out["worst_fold_excess_return_net"] = float(s.min())
    if "test_max_dd" in d.columns:
        s = pd.to_numeric(d["test_max_dd"], errors="coerce").dropna()
        if not s.empty:
            out["median_test_max_dd"] = float(s.median())
    if "test_turnover" in d.columns:
        s = pd.to_numeric(d["test_turnover"], errors="coerce").dropna()
        if not s.empty:
            out["median_test_turnover"] = float(s.median())
    if "test_pct_cash" in d.columns:
        s = pd.to_numeric(d["test_pct_cash"], errors="coerce").dropna()
        if not s.empty:
            out["median_test_pct_cash"] = float(s.median())
    return out


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_dir: Path
    summary_path: Path
    best_config_path: Path
    trials_path: Path
    holdout_path: Path
    mtime: float
    cfg: dict[str, Any]

    @property
    def optimizer(self) -> dict[str, Any]:
        opt = self.cfg.get("optimizer", {})
        return opt if isinstance(opt, dict) else {}

    @property
    def seed(self) -> int | None:
        return _to_int(self.optimizer.get("seed"))


def _discover_runs(runs_dir: Path) -> list[RunInfo]:
    out: list[RunInfo] = []
    if not runs_dir.exists():
        return out

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        files = {name: d / name for name in REQUIRED_RUN_FILES}
        if not all(p.exists() for p in files.values()):
            continue
        best_cfg = _load_yaml(files["best_config.yaml"])
        out.append(
            RunInfo(
                run_id=d.name,
                run_dir=d,
                summary_path=files["optimize_summary.md"],
                best_config_path=files["best_config.yaml"],
                trials_path=files["trials.csv"],
                holdout_path=files["holdout_results.csv"],
                mtime=d.stat().st_mtime,
                cfg=best_cfg,
            )
        )

    out.sort(key=lambda r: (r.mtime, r.run_id), reverse=True)
    return out


def _match_optimizer_metadata(run: RunInfo, args: argparse.Namespace) -> bool:
    o = run.optimizer
    checks = [
        (args.start, _to_int(o.get("start"))),
        (args.end, _to_int(o.get("end"))),
        (args.train_window_years, _to_int(o.get("train_window_years"))),
        (args.test_window_years, _to_int(o.get("test_window_years"))),
        (args.holdout_start, _to_int(o.get("holdout_start"))),
        (args.holdout_end, _to_int(o.get("holdout_end"))),
    ]
    for expected, actual in checks:
        if expected is not None and actual != expected:
            return False
    if args.rebalance and str(o.get("rebalance", "")).strip().lower() != str(args.rebalance).strip().lower():
        return False
    return True


def _pick_latest_per_seed(
    runs: list[RunInfo],
    seeds: list[int],
    latest: int,
    exclude_run_ids: set[str] | None = None,
) -> tuple[list[RunInfo], list[int]]:
    excluded = exclude_run_ids or set()
    d = [r for r in runs if r.run_id not in excluded]
    if latest > 0:
        d = d[:latest]

    if not seeds:
        return d, []

    by_seed: dict[int, RunInfo] = {}
    for r in d:
        seed = r.seed
        if seed is None:
            continue
        if seed not in seeds:
            continue
        if seed not in by_seed:
            by_seed[seed] = r

    selected = [by_seed[s] for s in sorted(by_seed.keys())]
    missing = [s for s in seeds if s not in by_seed]
    return selected, missing


def _has_baseline_marker(run_dir: Path) -> bool:
    return any((run_dir / marker).exists() for marker in BASELINE_MARKERS)


def _extract_row(run: RunInfo) -> dict[str, Any]:
    cfg = run.cfg
    opt = run.optimizer
    sel_params = cfg.get("selected_params", {})
    if not isinstance(sel_params, dict):
        sel_params = {}
    sel_metrics = cfg.get("selection_metrics", {})
    if not isinstance(sel_metrics, dict):
        sel_metrics = {}
    holdout_cfg = cfg.get("holdout_metrics", {})
    if not isinstance(holdout_cfg, dict):
        holdout_cfg = {}
    strat_cfg = cfg.get("strategy_a", {})
    if not isinstance(strat_cfg, dict):
        strat_cfg = {}
    weights = sel_params.get("weights", {})
    if not isinstance(weights, dict):
        weights = {}

    selected_trial_id = str(sel_metrics.get("trial_id", "")).strip() or None
    trial_metrics = _selection_metrics_from_trials(run.trials_path, selected_trial_id)
    holdout_csv_metrics = _holdout_metrics_from_csv(run.holdout_path)

    cumulative_return_net = _first_valid_number(
        holdout_cfg.get("cumulative_return_net"),
        holdout_csv_metrics.get("cumulative_return_net"),
    )
    holdout_excess_return_net = _first_valid_number(
        holdout_cfg.get("excess_return_net"),
    )
    holdout_max_dd = _first_valid_number(
        holdout_cfg.get("max_dd"),
        holdout_csv_metrics.get("max_dd"),
    )
    holdout_turnover = _first_valid_number(
        holdout_cfg.get("turnover"),
        holdout_csv_metrics.get("turnover"),
    )
    holdout_pct_cash = _first_valid_number(
        holdout_cfg.get("pct_cash"),
        holdout_csv_metrics.get("pct_cash"),
    )
    holdout_switches = _first_valid_number(
        holdout_cfg.get("switches"),
        holdout_cfg.get("trade_count"),
        holdout_csv_metrics.get("switches"),
    )

    excess_for_utility = _first_valid_number(holdout_excess_return_net, cumulative_return_net, 0.0)
    dd_for_utility = _first_valid_number(holdout_max_dd, 0.0)
    turn_for_utility = _first_valid_number(holdout_turnover, 0.0)
    cash_for_utility = _first_valid_number(holdout_pct_cash, 0.0)
    holdout_utility = float(excess_for_utility - 0.50 * abs(dd_for_utility) - 0.10 * turn_for_utility - 0.05 * cash_for_utility)

    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "summary_path": str(run.summary_path),
        "best_config_path": str(run.best_config_path),
        "trials_path": str(run.trials_path),
        "holdout_results_path": str(run.holdout_path),
        "seed": _to_int(opt.get("seed")),
        "optimizer_start": _to_int(opt.get("start")),
        "optimizer_end": _to_int(opt.get("end")),
        "train_window_years": _to_int(opt.get("train_window_years")),
        "test_window_years": _to_int(opt.get("test_window_years")),
        "rebalance": str(opt.get("rebalance", "")),
        "holdout_start": _to_int(opt.get("holdout_start")),
        "holdout_end": _to_int(opt.get("holdout_end")),
        "selection_folds": _to_int(opt.get("selection_folds")),
        "holdout_folds": _to_int(opt.get("holdout_folds")),
        "n_trials": _to_int(opt.get("n_trials")),
        "selected_trial_id": selected_trial_id or "",
        "mos_threshold": _to_float(sel_params.get("mos_threshold")),
        "mad_min": _to_float(sel_params.get("mad_min")),
        "mad_penalty_k": _to_float(sel_params.get("mad_penalty_k")),
        "min_hold_months": _to_int(sel_params.get("min_hold_months")),
        "score_gap": _to_float(sel_params.get("score_gap")),
        "weight_quality": _to_float(weights.get("quality")),
        "weight_value": _to_float(weights.get("value")),
        "weight_lowrisk": _to_float(weights.get("lowrisk")),
        "weight_balance": _to_float(weights.get("balance")),
        "weights_reg_lambda": _to_float(strat_cfg.get("weights_reg_lambda")),
        "median_excess_return_net": _first_valid_number(
            trial_metrics.get("median_excess_return_net"),
            sel_metrics.get("median_excess_return_net"),
        ),
        "worst_fold_excess_return_net": _first_valid_number(
            trial_metrics.get("worst_fold_excess_return_net"),
            sel_metrics.get("worst_fold_excess_return_net"),
        ),
        "median_test_max_dd": _first_valid_number(
            trial_metrics.get("median_test_max_dd"),
            sel_metrics.get("median_test_max_dd"),
        ),
        "median_test_turnover": _first_valid_number(
            trial_metrics.get("median_test_turnover"),
            sel_metrics.get("median_test_turnover"),
            sel_metrics.get("mean_test_turnover"),
        ),
        "median_test_pct_cash": _first_valid_number(
            trial_metrics.get("median_test_pct_cash"),
            sel_metrics.get("median_test_pct_cash"),
            sel_metrics.get("mean_test_pct_cash"),
        ),
        "cumulative_return_net": cumulative_return_net,
        "holdout_excess_return_net": holdout_excess_return_net,
        "holdout_max_dd": holdout_max_dd,
        "holdout_turnover": holdout_turnover,
        "holdout_pct_cash": holdout_pct_cash,
        "holdout_switches": holdout_switches,
        "holdout_utility": holdout_utility,
    }


def _resolve_baseline_run(
    args: argparse.Namespace,
    matching_runs: list[RunInfo],
    runs_dir: Path,
    strategy_rows: list[dict[str, Any]],
) -> RunInfo | None:
    if args.baseline_run_id:
        p = Path(args.baseline_run_id)
        run_dir = p if p.is_absolute() else (runs_dir / args.baseline_run_id)
        if not run_dir.exists():
            raise RuntimeError(f"--baseline-run-id not found: {run_dir}")
        required = [run_dir / name for name in REQUIRED_RUN_FILES]
        if not all(x.exists() for x in required):
            raise RuntimeError(f"--baseline-run-id missing required files: {run_dir}")
        return RunInfo(
            run_id=run_dir.name,
            run_dir=run_dir,
            summary_path=run_dir / "optimize_summary.md",
            best_config_path=run_dir / "best_config.yaml",
            trials_path=run_dir / "trials.csv",
            holdout_path=run_dir / "holdout_results.csv",
            mtime=run_dir.stat().st_mtime,
            cfg=_load_yaml(run_dir / "best_config.yaml"),
        )

    if args.run_baseline:
        return _run_baseline(args=args, runs_dir=runs_dir, strategy_rows=strategy_rows)

    marked = [r for r in matching_runs if _has_baseline_marker(r.run_dir)]
    if not marked:
        return None
    marked.sort(key=lambda r: (r.mtime, r.run_id), reverse=True)
    return marked[0]


def _run_baseline(args: argparse.Namespace, runs_dir: Path, strategy_rows: list[dict[str, Any]]) -> RunInfo:
    if not strategy_rows:
        raise RuntimeError("Cannot infer baseline metadata because no Strategy A runs matched filters.")

    template = strategy_rows[0]
    start = args.start if args.start is not None else _to_int(template.get("optimizer_start"))
    end = args.end if args.end is not None else _to_int(template.get("optimizer_end"))
    train_years = args.train_window_years if args.train_window_years is not None else _to_int(template.get("train_window_years"))
    test_years = args.test_window_years if args.test_window_years is not None else _to_int(template.get("test_window_years"))
    rebalance = args.rebalance if args.rebalance else str(template.get("rebalance", "monthly"))
    if None in (start, end, train_years, test_years):
        raise RuntimeError("Missing start/end/train/test metadata required for --run-baseline.")

    run_id = f"wft_opt_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = runs_dir / run_id
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_wft_optimize.py"),
        "--config",
        str(_resolve_path(args.config)),
        "--start",
        str(int(start)),
        "--end",
        str(int(end)),
        "--train-window-years",
        str(int(train_years)),
        "--test-window-years",
        str(int(test_years)),
        "--rebalance",
        str(rebalance),
        "--n-trials",
        "1",
        "--seed",
        str(int(args.baseline_seed)),
        "--holdout-start",
        str(int(args.holdout_start)),
        "--holdout-end",
        str(int(args.holdout_end)),
        "--search-mode",
        "grid",
        "--mos-grid",
        "0.30",
        "--mad-grid",
        "-0.02",
        "--mad-penalty-k-grid",
        "0.0",
        "--min-hold-grid",
        "3",
        "--score-gap-grid",
        "0.25",
        "--weights-grid-step",
        "1.0",
        "--weakness-rule-variant",
        "baseline",
        "--output-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        msg = "\n".join(
            [
                "Baseline run failed.",
                f"Command: {' '.join(cmd)}",
                f"stdout:\n{proc.stdout}",
                f"stderr:\n{proc.stderr}",
            ]
        )
        raise RuntimeError(msg)

    marker_path = out_dir / BASELINE_MARKERS[0]
    marker_path.write_text("strategyA baseline marker\n", encoding="utf-8")

    return RunInfo(
        run_id=out_dir.name,
        run_dir=out_dir,
        summary_path=out_dir / "optimize_summary.md",
        best_config_path=out_dir / "best_config.yaml",
        trials_path=out_dir / "trials.csv",
        holdout_path=out_dir / "holdout_results.csv",
        mtime=out_dir.stat().st_mtime,
        cfg=_load_yaml(out_dir / "best_config.yaml"),
    )


def _attach_baseline_deltas(rows: list[dict[str, Any]], baseline: dict[str, Any] | None) -> None:
    for r in rows:
        if not baseline:
            r["baseline_run_id"] = ""
            r["delta_cumulative_return_net"] = math.nan
            r["delta_holdout_excess_return_net"] = math.nan
            r["delta_holdout_max_dd"] = math.nan
            r["delta_holdout_turnover"] = math.nan
            r["delta_holdout_pct_cash"] = math.nan
            r["delta_holdout_switches"] = math.nan
            r["delta_utility"] = math.nan
            continue

        r["baseline_run_id"] = baseline["run_id"]
        r["delta_cumulative_return_net"] = _to_float(r["cumulative_return_net"]) - _to_float(baseline["cumulative_return_net"])
        r["delta_holdout_excess_return_net"] = _to_float(r["holdout_excess_return_net"]) - _to_float(baseline["holdout_excess_return_net"])
        r["delta_holdout_max_dd"] = _to_float(r["holdout_max_dd"]) - _to_float(baseline["holdout_max_dd"])
        r["delta_holdout_turnover"] = _to_float(r["holdout_turnover"]) - _to_float(baseline["holdout_turnover"])
        r["delta_holdout_pct_cash"] = _to_float(r["holdout_pct_cash"]) - _to_float(baseline["holdout_pct_cash"])
        r["delta_holdout_switches"] = _to_float(r["holdout_switches"]) - _to_float(baseline["holdout_switches"])

        delta_excess = _first_valid_number(r["delta_holdout_excess_return_net"], r["delta_cumulative_return_net"], 0.0)
        delta_dd_penalty = abs(_to_float(r["holdout_max_dd"])) - abs(_to_float(baseline["holdout_max_dd"]))
        delta_turn_penalty = _to_float(r["delta_holdout_turnover"])
        delta_cash_penalty = _to_float(r["delta_holdout_pct_cash"])
        r["delta_utility"] = float(
            delta_excess
            - 0.50 * max(delta_dd_penalty, 0.0)
            - 0.10 * max(delta_turn_penalty, 0.0)
            - 0.05 * max(delta_cash_penalty, 0.0)
        )


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda r: (
            -_first_valid_number(r.get("holdout_utility"), -1e9),
            -_first_valid_number(r.get("holdout_excess_return_net"), r.get("cumulative_return_net"), -1e9),
            abs(_first_valid_number(r.get("holdout_max_dd"), 0.0)),
            _first_valid_number(r.get("holdout_turnover"), 1e9),
            _first_valid_number(r.get("holdout_pct_cash"), 1e9),
            _to_int(r.get("seed")) if r.get("seed") is not None else 10**9,
            str(r.get("run_id", "")),
        )
    )
    for idx, r in enumerate(rows, start=1):
        r["rank"] = int(idx)


def _write_summary_markdown(
    rows_df: pd.DataFrame,
    out_path: Path,
    args: argparse.Namespace,
    baseline_row: dict[str, Any] | None,
    missing_seeds: list[int],
) -> None:
    cols = [
        "rank",
        "seed",
        "run_id",
        "holdout_excess_return_net",
        "holdout_max_dd",
        "holdout_turnover",
        "holdout_pct_cash",
        "delta_holdout_excess_return_net",
        "delta_holdout_max_dd",
        "delta_holdout_turnover",
        "delta_holdout_pct_cash",
        "delta_utility",
    ]
    show = rows_df.copy()
    for c in cols:
        if c not in show.columns:
            show[c] = math.nan
    show = show.sort_values(by=["rank", "seed", "run_id"], kind="mergesort").reset_index(drop=True)

    lines: list[str] = []
    lines.append("# Strategy A Seed Summary")
    lines.append("")
    lines.append("## Filters")
    lines.append(f"- mode: `{args.mode}`")
    lines.append(f"- holdout_start/holdout_end: `{args.holdout_start}` / `{args.holdout_end}`")
    lines.append(f"- seeds: `{args.seeds}`")
    lines.append(f"- latest: `{args.latest}`")
    if args.start is not None:
        lines.append(f"- start/end: `{args.start}` / `{args.end}`")
    if args.train_window_years is not None:
        lines.append(f"- train/test years: `{args.train_window_years}` / `{args.test_window_years}`")
    if args.rebalance:
        lines.append(f"- rebalance: `{args.rebalance}`")
    lines.append("")

    lines.append("## Baseline")
    if baseline_row:
        lines.append(f"- baseline_run_id: `{baseline_row['run_id']}`")
        lines.append(f"- cumulative_return_net: {_format_num(baseline_row.get('cumulative_return_net'))}")
        lines.append(f"- holdout_excess_return_net: {_format_num(baseline_row.get('holdout_excess_return_net'))}")
        lines.append(f"- holdout_max_dd: {_format_num(baseline_row.get('holdout_max_dd'))}")
        lines.append(f"- holdout_turnover: {_format_num(baseline_row.get('holdout_turnover'))}")
        lines.append(f"- holdout_pct_cash: {_format_num(baseline_row.get('holdout_pct_cash'))}")
    else:
        lines.append("- baseline_run_id: _not found_")
    lines.append("")

    if missing_seeds:
        lines.append("## Missing Seeds")
        lines.append(f"- {', '.join(str(s) for s in missing_seeds)}")
        lines.append("")

    lines.append("## Ranked Table")
    lines.append(_md_table(show, cols))
    lines.append("")
    lines.append("## Notes")
    lines.append("- `delta_*` values are `StrategyA - Baseline`.")
    lines.append("- Ranking uses holdout utility = excess - 0.50*abs(max_dd) - 0.10*turnover - 0.05*pct_cash.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommendation(rows_df: pd.DataFrame, baseline_row: dict[str, Any] | None) -> tuple[str, list[str]]:
    lines: list[str] = []
    lines.append("# Strategy A Recommendation")
    lines.append("")
    lines.append("## Decision Rule")
    lines.append("1. Prefer higher holdout_excess_return_net (fallback: cumulative_return_net).")
    lines.append("2. Penalize worse drawdown, turnover, and pct_cash vs baseline.")
    lines.append("3. Require stability: at least 3 seeds with the same direction vs baseline.")
    lines.append("")

    decision = "tighten regularization/penalties"
    rationale: list[str] = []
    if baseline_row is None:
        rationale.append("No baseline run available, so deltas are unavailable.")
        rationale.append("Defaulting to tighten regularization/penalties until baseline comparison is completed.")
        return decision, lines + ["## Result", f"- decision: **{decision}**", "", "## Why"] + [f"- {x}" for x in rationale]

    d = rows_df.copy()
    if d.empty:
        rationale.append("No strategy runs matched filters.")
        rationale.append("Defaulting to tighten regularization/penalties.")
        return decision, lines + ["## Result", f"- decision: **{decision}**", "", "## Why"] + [f"- {x}" for x in rationale]

    delta_util = pd.to_numeric(d.get("delta_utility"), errors="coerce")
    delta_excess = pd.to_numeric(d.get("delta_holdout_excess_return_net"), errors="coerce")
    delta_cum = pd.to_numeric(d.get("delta_cumulative_return_net"), errors="coerce")
    delta_excess_fallback = delta_excess.where(delta_excess.notna(), delta_cum)

    improve = int((delta_util > 0).sum())
    degrade = int((delta_util < 0).sum())
    flat = int((delta_util == 0).sum())
    n = int(len(d))
    stability = max(improve, degrade)
    median_delta_excess = float(delta_excess_fallback.median(skipna=True)) if n > 0 else math.nan
    mean_delta_excess = float(delta_excess_fallback.mean(skipna=True)) if n > 0 else math.nan

    if improve >= 3 and stability >= 3 and _first_valid_number(median_delta_excess, 0.0) >= 0.0:
        decision = "freeze config"
    else:
        decision = "tighten regularization/penalties"

    rationale.append(f"seeds_evaluated={n}, improve={improve}, degrade={degrade}, flat={flat}")
    rationale.append(f"stability_count={stability} (must be >=3)")
    rationale.append(f"median_delta_excess={_format_num(median_delta_excess)}")
    rationale.append(f"mean_delta_excess={_format_num(mean_delta_excess)}")

    result_lines = lines + ["## Result", f"- decision: **{decision}**", "", "## Why"] + [f"- {x}" for x in rationale]
    return decision, result_lines


def _write_recommendation_markdown(
    rows_df: pd.DataFrame,
    out_path: Path,
    baseline_row: dict[str, Any] | None,
) -> None:
    _, lines = _recommendation(rows_df, baseline_row)

    display_cols = [
        "seed",
        "run_id",
        "holdout_excess_return_net",
        "holdout_max_dd",
        "holdout_turnover",
        "holdout_pct_cash",
        "delta_holdout_excess_return_net",
        "delta_holdout_max_dd",
        "delta_holdout_turnover",
        "delta_holdout_pct_cash",
        "delta_utility",
    ]
    d = rows_df.copy()
    for c in display_cols:
        if c not in d.columns:
            d[c] = math.nan
    d = d.sort_values(by=["seed", "run_id"], kind="mergesort").reset_index(drop=True)

    lines.append("")
    lines.append("## Seed-Level Table")
    lines.append(_md_table(d, display_cols))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate Strategy A optimize runs across seeds and compare vs baseline.")
    p.add_argument("--mode", choices=["strategyA"], required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output-dir", default="experiments")
    p.add_argument("--config", default="config/config.yaml")

    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--train-window-years", type=int, default=None)
    p.add_argument("--test-window-years", type=int, default=None)
    p.add_argument("--rebalance", default=None)
    p.add_argument("--holdout-start", type=int, required=True)
    p.add_argument("--holdout-end", type=int, required=True)
    p.add_argument("--seeds", default="")
    p.add_argument("--latest", type=int, default=10)

    p.add_argument("--baseline-run-id", default=None)
    p.add_argument("--run-baseline", action="store_true")
    p.add_argument("--baseline-seed", type=int, default=1)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.holdout_start > args.holdout_end:
        raise ValueError("--holdout-start must be <= --holdout-end")
    if args.latest <= 0:
        raise ValueError("--latest must be > 0")

    seeds = _parse_seeds(args.seeds)
    runs_dir = _resolve_path(args.runs_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_runs(runs_dir)
    matching = [r for r in discovered if _match_optimizer_metadata(r, args)]
    selected_runs, missing_seeds = _pick_latest_per_seed(matching, seeds=seeds, latest=int(args.latest))
    strategy_rows = [_extract_row(r) for r in selected_runs]

    baseline_run = _resolve_baseline_run(args=args, matching_runs=matching, runs_dir=runs_dir, strategy_rows=strategy_rows)
    if baseline_run:
        selected_runs, missing_seeds = _pick_latest_per_seed(
            matching,
            seeds=seeds,
            latest=int(args.latest),
            exclude_run_ids={baseline_run.run_id},
        )
        strategy_rows = [_extract_row(r) for r in selected_runs]
    baseline_row = _extract_row(baseline_run) if baseline_run else None

    _attach_baseline_deltas(strategy_rows, baseline_row)
    _rank_rows(strategy_rows)

    rows_df = pd.DataFrame(strategy_rows)
    if not rows_df.empty:
        rows_df = rows_df.sort_values(by=["rank", "seed", "run_id"], kind="mergesort").reset_index(drop=True)

    csv_path = output_dir / "strategyA_seed_summary.csv"
    md_path = output_dir / "strategyA_seed_summary.md"
    rec_path = output_dir / "strategyA_recommendation.md"

    # Stable column ordering keeps git diffs easy to audit.
    column_order = [
        "rank",
        "seed",
        "run_id",
        "run_dir",
        "summary_path",
        "best_config_path",
        "trials_path",
        "holdout_results_path",
        "optimizer_start",
        "optimizer_end",
        "train_window_years",
        "test_window_years",
        "rebalance",
        "holdout_start",
        "holdout_end",
        "n_trials",
        "selection_folds",
        "holdout_folds",
        "selected_trial_id",
        "mos_threshold",
        "mad_min",
        "mad_penalty_k",
        "min_hold_months",
        "score_gap",
        "weight_quality",
        "weight_value",
        "weight_lowrisk",
        "weight_balance",
        "weights_reg_lambda",
        "median_excess_return_net",
        "worst_fold_excess_return_net",
        "median_test_max_dd",
        "median_test_turnover",
        "median_test_pct_cash",
        "cumulative_return_net",
        "holdout_excess_return_net",
        "holdout_max_dd",
        "holdout_turnover",
        "holdout_pct_cash",
        "holdout_switches",
        "holdout_utility",
        "baseline_run_id",
        "delta_cumulative_return_net",
        "delta_holdout_excess_return_net",
        "delta_holdout_max_dd",
        "delta_holdout_turnover",
        "delta_holdout_pct_cash",
        "delta_holdout_switches",
        "delta_utility",
    ]
    for c in column_order:
        if c not in rows_df.columns:
            rows_df[c] = math.nan
    rows_df = rows_df[column_order]
    rows_df.to_csv(csv_path, index=False)

    _write_summary_markdown(rows_df, md_path, args, baseline_row, missing_seeds)
    _write_recommendation_markdown(rows_df, rec_path, baseline_row)

    print(f"OK: {csv_path}")
    print(f"OK: {md_path}")
    print(f"OK: {rec_path}")
    if baseline_row:
        print(f"Baseline: {baseline_row['run_id']}")
    else:
        print("Baseline: not found")
    if missing_seeds:
        print(f"Missing seeds: {','.join(str(s) for s in missing_seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
