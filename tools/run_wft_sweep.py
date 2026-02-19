from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config, resolve_paths


@dataclass(frozen=True)
class SweepConfig:
    mos_threshold: float
    mad_min: float
    weakness_rule_variant: str

    def as_params_json(self) -> str:
        return json.dumps(
            {
                "mos_threshold": float(self.mos_threshold),
                "mad_min": float(self.mad_min),
                "weakness_rule_variant": str(self.weakness_rule_variant),
            },
            sort_keys=True,
        )


def _parse_float_grid(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Grid cannot be empty.")
    return out


def _parse_str_grid(text: str) -> list[str]:
    out: list[str] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(token)
    if not out:
        raise ValueError("Grid cannot be empty.")
    return out


def _detect_supported_weakness_variants(project_root: Path) -> set[str]:
    src_path = project_root / "src" / "wft.py"
    txt = src_path.read_text(encoding="utf-8")
    out: set[str] = set()
    for variant in ("baseline", "strict", "stricter"):
        if f'"{variant}"' in txt or f"'{variant}'" in txt:
            out.add(variant)
    return out


def _resolve_weakness_grid(args: argparse.Namespace, project_root: Path) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    supported = _detect_supported_weakness_variants(project_root)

    if args.weakness_variants:
        requested = _parse_str_grid(args.weakness_variants)
        resolved = [v for v in requested if v in supported]
        dropped = [v for v in requested if v not in supported]
        if dropped:
            notes.append(f"Dropped unsupported weakness variants: {', '.join(dropped)}")
        if not resolved:
            resolved = ["baseline"] if "baseline" in supported else ["baseline"]
            notes.append("No requested weakness variant was supported; fallback to baseline.")
        return resolved, notes

    # Default policy requested by user: strict only if it exists, otherwise baseline only.
    if "strict" in supported:
        return ["baseline", "strict"], notes

    if "stricter" in supported:
        notes.append("Variant 'strict' not found in code; using baseline only by default (found: stricter).")
    else:
        notes.append("Variant 'strict' not found in code; using baseline only by default.")
    return ["baseline"], notes


def _annual_cagr(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n == 0:
        return 0.0
    nav = float((1.0 + r).prod())
    if nav <= 0:
        return -1.0
    return float(nav ** (1.0 / n) - 1.0)


def _candidate_stability(states: pd.Series) -> float:
    s = states.fillna("CASH").astype(str).tolist()
    if len(s) <= 1:
        return 1.0
    stable = 0
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            stable += 1
    return float(stable / (len(s) - 1))


def _aggregate_wft_results(results_path: Path, cost_rate: float) -> dict:
    df = pd.read_csv(results_path)
    if "mode" in df.columns:
        tuned = df[df["mode"].astype(str).str.lower().eq("tuned")].copy()
        if tuned.empty:
            tuned = df.copy()
    else:
        tuned = df.copy()

    tuned["test_year"] = pd.to_numeric(tuned["test_year"], errors="coerce")
    for col in ("return", "max_dd", "turnover", "pct_cash"):
        tuned[col] = pd.to_numeric(tuned[col], errors="coerce")
    tuned = tuned.dropna(subset=["test_year"]).copy()
    tuned["test_year"] = tuned["test_year"].astype(int)

    by_year = (
        tuned.groupby("test_year", as_index=True)
        .agg(
            return_gross=("return", "mean"),
            max_dd=("max_dd", "min"),
            turnover=("turnover", "mean"),
            pct_cash=("pct_cash", "mean"),
            candidate_or_cash=("candidate_or_cash", "last"),
        )
        .sort_index()
    )

    if by_year.empty:
        return {
            "folds": 0,
            "oos_return_mean": 0.0,
            "oos_return_median": 0.0,
            "oos_worst_year": 0.0,
            "oos_return_mean_gross": 0.0,
            "oos_return_median_gross": 0.0,
            "oos_worst_year_gross": 0.0,
            "oos_maxdd": 0.0,
            "oos_turnover": 0.0,
            "oos_pct_cash": 1.0,
            "oos_stability": 1.0,
            "oos_cagr": 0.0,
            "oos_cagr_gross": 0.0,
            "oos_return_mean_net": 0.0,
            "oos_return_median_net": 0.0,
            "oos_worst_year_net": 0.0,
            "oos_cagr_net": 0.0,
        }

    by_year["return_net"] = by_year["return_gross"] - (by_year["turnover"] * cost_rate)

    return {
        "folds": int(len(by_year)),
        "oos_return_mean": float(by_year["return_gross"].mean()),
        "oos_return_median": float(by_year["return_gross"].median()),
        "oos_worst_year": float(by_year["return_gross"].min()),
        "oos_return_mean_gross": float(by_year["return_gross"].mean()),
        "oos_return_median_gross": float(by_year["return_gross"].median()),
        "oos_worst_year_gross": float(by_year["return_gross"].min()),
        "oos_maxdd": float(by_year["max_dd"].min()),
        "oos_turnover": float(by_year["turnover"].mean()),
        "oos_pct_cash": float(by_year["pct_cash"].mean()),
        "oos_stability": _candidate_stability(by_year["candidate_or_cash"]),
        "oos_cagr": _annual_cagr(by_year["return_gross"]),
        "oos_cagr_gross": _annual_cagr(by_year["return_gross"]),
        "oos_return_mean_net": float(by_year["return_net"].mean()),
        "oos_return_median_net": float(by_year["return_net"].median()),
        "oos_worst_year_net": float(by_year["return_net"].min()),
        "oos_cagr_net": _annual_cagr(by_year["return_net"]),
    }


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    d = df.copy()
    for c in d.columns:
        if c == "rank":
            d[c] = d[c].map(lambda x: str(int(x)) if pd.notna(x) else "")
        elif pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
        else:
            d[c] = d[c].astype(str)
    cols = list(d.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for i in range(len(d)):
        lines.append("| " + " | ".join(d.iloc[i].tolist()) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run conservative WFT sweep and aggregate OOS performance.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--rebalance", default="monthly")
    p.add_argument("--test-window-years", type=int, default=1)
    p.add_argument("--train-window-years", type=int, default=12)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--cost-bps", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mos-grid", default="0.30,0.35,0.40")
    p.add_argument("--mad-grid", default="-0.05,0.00,0.02,0.05")
    p.add_argument("--weakness-variants", default=None)
    p.add_argument("--max-combos", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--master-path", default=None)
    p.add_argument("--prices-path", default=None)
    return p.parse_args()


def _resolve_path(base_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (base_root / p).resolve()


def main() -> int:
    args = parse_args()
    if int(args.start) > int(args.end):
        raise ValueError("--start must be <= --end")
    if int(args.test_window_years) != 1:
        raise ValueError("--test-window-years must be 1 for annual folds")
    if str(args.rebalance).lower() != "monthly":
        raise ValueError("--rebalance must be monthly")

    config_path = _resolve_path(PROJECT_ROOT, args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(config_path)
    paths = resolve_paths(cfg, PROJECT_ROOT)
    runs_dir = paths["runs_dir"]

    output_dir = _resolve_path(PROJECT_ROOT, args.output_dir)
    sweep_root = output_dir if output_dir else (runs_dir / f"wft_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    sweep_root.mkdir(parents=True, exist_ok=True)

    mos_values = _parse_float_grid(args.mos_grid)
    mad_values = _parse_float_grid(args.mad_grid)
    weakness_values, notes = _resolve_weakness_grid(args, PROJECT_ROOT)

    combos = [
        SweepConfig(mos_threshold=mos, mad_min=mad, weakness_rule_variant=wv)
        for mos, mad, wv in itertools.product(mos_values, mad_values, weakness_values)
    ]
    if args.max_combos is not None and int(args.max_combos) > 0:
        combos = combos[: int(args.max_combos)]
        notes.append(f"Limited combos with --max-combos={int(args.max_combos)}")

    if not combos:
        raise RuntimeError("No sweep combinations to run.")

    cost_rate = float(args.cost_bps) / 10000.0
    rows: list[dict] = []

    for idx, combo in enumerate(combos, start=1):
        config_id = f"cfg_{idx:03d}"
        run_dir = sweep_root / config_id

        cmd = [
            sys.executable,
            "-m",
            "src.wft",
            "--config",
            str(config_path),
            "--start",
            str(int(args.start)),
            "--end",
            str(int(args.end)),
            "--rebalance",
            "monthly",
            "--test-window-years",
            str(int(args.test_window_years)),
            "--train-window-years",
            str(int(args.train_window_years)),
            "--step-years",
            str(int(args.step_years)),
            "--seed",
            str(int(args.seed)),
            "--run-dir",
            str(run_dir),
        ]

        master_path = _resolve_path(PROJECT_ROOT, args.master_path)
        prices_path = _resolve_path(PROJECT_ROOT, args.prices_path)
        if master_path:
            cmd.extend(["--master-path", str(master_path)])
        if prices_path:
            cmd.extend(["--prices-path", str(prices_path)])

        env = os.environ.copy()
        env["WFT_MOS_GRID"] = str(combo.mos_threshold)
        env["WFT_MAD_GRID"] = str(combo.mad_min)
        env["WFT_WEAKNESS_VARIANTS"] = str(combo.weakness_rule_variant)

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
        )

        base_row = {
            "config_id": config_id,
            "params": combo.as_params_json(),
            "mos_threshold": float(combo.mos_threshold),
            "mad_min": float(combo.mad_min),
            "weakness_rule_variant": str(combo.weakness_rule_variant),
            "run_dir": str(run_dir),
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "stderr": (proc.stderr or "").strip()[-1000:],
        }

        results_path = run_dir / "wft_results.csv"
        if proc.returncode == 0 and results_path.exists():
            metrics = _aggregate_wft_results(results_path=results_path, cost_rate=cost_rate)
            base_row.update(metrics)
        else:
            base_row.update(
                {
                    "folds": 0,
                    "oos_return_mean": 0.0,
                    "oos_return_median": 0.0,
                    "oos_worst_year": 0.0,
                    "oos_return_mean_gross": 0.0,
                    "oos_return_median_gross": 0.0,
                    "oos_worst_year_gross": 0.0,
                    "oos_maxdd": 0.0,
                    "oos_turnover": 0.0,
                    "oos_pct_cash": 1.0,
                    "oos_stability": 1.0,
                    "oos_cagr": 0.0,
                    "oos_cagr_gross": 0.0,
                    "oos_return_mean_net": 0.0,
                    "oos_return_median_net": 0.0,
                    "oos_worst_year_net": 0.0,
                    "oos_cagr_net": 0.0,
                }
            )
        rows.append(base_row)

    out = pd.DataFrame(rows)
    ok = out[out["status"] == "ok"].copy()
    if not ok.empty:
        ok = ok.sort_values(
            by=["oos_maxdd", "oos_worst_year_gross", "oos_return_median_gross", "oos_turnover"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ok["rank"] = ok.index + 1
        out = out.merge(ok[["config_id", "rank"]], on="config_id", how="left")
    else:
        out["rank"] = pd.NA

    results_csv = sweep_root / "sweep_results.csv"
    out.to_csv(results_csv, index=False)

    top5 = out[out["status"] == "ok"].sort_values("rank").head(5).copy()
    chosen = top5.iloc[0] if not top5.empty else None

    summary_lines = [
        "# WFT Sweep Summary",
        "",
        f"- Output dir: `{sweep_root}`",
        f"- Config: `{config_path}`",
        f"- Period: {int(args.start)}-{int(args.end)}",
        f"- Rebalance: {args.rebalance}",
        f"- Train/Test windows: {int(args.train_window_years)}y / {int(args.test_window_years)}y",
        f"- Cost assumption: {float(args.cost_bps):.1f} bps per 100% annual turnover",
        f"- Total configs: {len(out)}",
        f"- Successful configs: {int((out['status'] == 'ok').sum())}",
        "",
    ]
    if notes:
        summary_lines.append("## Notes")
        for note in notes:
            summary_lines.append(f"- {note}")
        summary_lines.append("")

    summary_lines.extend(
        [
            "## Ranking Logic (Conservative)",
            "Priority order:",
            "1. Best worst max drawdown (least negative).",
            "2. Best worst-year return.",
            "3. Best median OOS return.",
            "4. Lowest turnover.",
            "",
            "## Top 5 Configs",
        ]
    )

    if top5.empty:
        summary_lines.append("No successful WFT runs in sweep.")
    else:
        top5_view = top5[
            [
                "rank",
                "config_id",
                "mos_threshold",
                "mad_min",
                "weakness_rule_variant",
                "oos_cagr",
                "oos_cagr_net",
                "oos_maxdd",
                "oos_worst_year_gross",
                "oos_turnover",
                "oos_pct_cash",
                "oos_stability",
                "run_dir",
            ]
        ].copy()
        summary_lines.append(_md_table(top5_view))

    summary_lines.append("")
    summary_lines.append("## Selected Best Config")
    if chosen is None:
        summary_lines.append("No configuration selected (all runs failed).")
    else:
        summary_lines.append(f"- Selected: `{chosen['config_id']}`")
        summary_lines.append(f"- Params: `{chosen['params']}`")
        summary_lines.append(f"- Run dir: `{chosen['run_dir']}`")
        summary_lines.append(
            "- Reason: selected by conservative priority with focus on drawdown and worst-year return before return and turnover."
        )
        summary_lines.append(
            f"- Key metrics: maxDD={float(chosen['oos_maxdd']):.2%}, "
            f"worst_year={float(chosen['oos_worst_year_gross']):.2%}, "
            f"median_return={float(chosen['oos_return_median_gross']):.2%}, "
            f"turnover={float(chosen['oos_turnover']):.2%}, "
            f"cagr_gross={float(chosen['oos_cagr']):.2%}, "
            f"cagr_net={float(chosen['oos_cagr_net']):.2%}"
        )

    summary_md = sweep_root / "sweep_summary.md"
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"OK: {results_csv}")
    print(f"OK: {summary_md}")

    return 0 if chosen is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
