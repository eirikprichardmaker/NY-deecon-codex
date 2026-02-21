from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config, resolve_paths


GO_REASON_RISK_KEYS = {"MoS", ">200d", "kvalitetsscore"}


def _parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("Seed list cannot be empty.")
    return seeds


def _resolve_path(base_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (base_root / p).resolve()


def _detect_git_commit(project_root: Path) -> str:
    for env_name in ("GIT_COMMIT", "CI_COMMIT_SHA", "GITHUB_SHA"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"

    if proc.returncode != 0:
        return "unknown"
    out = (proc.stdout or "").strip()
    return out if out else "unknown"


def _safe_float(v) -> float:
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) else 0.0


def _annualized_sharpe(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd * (n ** 0.5))


def _max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _annual_cagr(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(r)
    if n == 0:
        return 0.0
    nav = float((1.0 + r).prod())
    if nav <= 0:
        return -1.0
    return float(nav ** (1.0 / n) - 1.0)


def _split_reasons(text: str) -> set[str]:
    out: set[str] = set()
    for part in str(text).split("|"):
        token = part.strip()
        if token:
            out.add(token)
    return out


def _reason_distribution(cash_reason_series: pd.Series) -> tuple[dict[str, float], float, float]:
    s = cash_reason_series.fillna("").astype(str)
    n = int(len(s))
    if n == 0:
        return {}, 0.0, 0.0

    counts: dict[str, int] = {}
    risk_rows = 0
    datamangler_rows = 0
    for raw in s:
        tokens = _split_reasons(raw)
        if not tokens:
            continue
        if tokens.intersection(GO_REASON_RISK_KEYS):
            risk_rows += 1
        if "datamangler" in tokens:
            datamangler_rows += 1
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

    dist = {k: float(v) / float(n) for k, v in sorted(counts.items())}
    return dist, (float(risk_rows) / float(n)), (float(datamangler_rows) / float(n))


def _aggregate_seed_metrics(
    results_path: Path,
    cost_rate: float,
    *,
    go_min_excess_cagr: float,
    go_min_info_ratio: float,
    go_maxdd_gap_pp: float,
    go_max_cash_share: float,
    go_high_cash_risk_reason_min_share: float,
    go_high_cash_max_datamangler_share: float,
) -> dict:
    df = pd.read_csv(results_path)
    if "mode" in df.columns:
        tuned = df[df["mode"].astype(str).str.lower().eq("tuned")].copy()
        if tuned.empty:
            tuned = df.copy()
    else:
        tuned = df.copy()

    tuned["test_year"] = pd.to_numeric(tuned.get("test_year"), errors="coerce")
    for col in ("return", "max_dd", "turnover", "benchmark_return", "benchmark_max_dd"):
        tuned[col] = pd.to_numeric(tuned.get(col), errors="coerce")
    tuned = tuned.dropna(subset=["test_year"]).copy()
    if tuned.empty:
        return {
            "folds": 0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "hitrate": 0.0,
            "turnover": 0.0,
            "costs": 0.0,
            "cagr_net": 0.0,
            "benchmark_cagr": 0.0,
            "benchmark_maxdd": 0.0,
            "excess_cagr": 0.0,
            "info_ratio": 0.0,
            "cash_share": 0.0,
            "cash_risk_reason_share": 0.0,
            "cash_datamangler_share": 0.0,
            "cash_reason_distribution": "{}",
            "maxdd_gap_vs_benchmark": 0.0,
            "rule_excess_cagr": False,
            "rule_info_ratio": False,
            "rule_maxdd_vs_benchmark": False,
            "rule_cash_share_reasons": True,
            "go_no_go": "NO_GO",
            "go_no_go_reasons": "no_folds",
        }

    by_year = (
        tuned.groupby("test_year", as_index=False)
        .agg(
            ret=("return", "mean"),
            max_dd=("max_dd", "min"),
            turnover=("turnover", "mean"),
            benchmark_ret=("benchmark_return", "mean"),
            benchmark_max_dd=("benchmark_max_dd", "min"),
        )
        .sort_values("test_year", kind="mergesort")
        .reset_index(drop=True)
    )
    by_year["ret_net"] = by_year["ret"] - (by_year["turnover"] * float(cost_rate))
    by_year["excess_ret"] = by_year["ret_net"] - by_year["benchmark_ret"]
    cagr = _annual_cagr(by_year["ret"])
    cagr_net = _annual_cagr(by_year["ret_net"])
    benchmark_cagr = _annual_cagr(by_year["benchmark_ret"])
    benchmark_maxdd_series = _max_drawdown(by_year["benchmark_ret"])
    benchmark_maxdd_col = pd.to_numeric(by_year["benchmark_max_dd"], errors="coerce").dropna()
    benchmark_maxdd = float(benchmark_maxdd_col.min()) if not benchmark_maxdd_col.empty else benchmark_maxdd_series
    excess_cagr = float(cagr_net - benchmark_cagr)
    info_ratio = _annualized_sharpe(by_year["excess_ret"])
    maxdd_gap_vs_benchmark = float(abs(float(by_year["max_dd"].min())) - abs(float(benchmark_maxdd)))

    cash_mask = tuned.get("candidate_or_cash", pd.Series("", index=tuned.index)).astype(str).str.upper().eq("CASH")
    cash_share = float(cash_mask.mean()) if len(cash_mask) else 0.0
    cash_reason_series = tuned.loc[cash_mask, "decision_reasons"] if "decision_reasons" in tuned.columns else pd.Series(dtype=str)
    reason_dist, cash_risk_reason_share, cash_datamangler_share = _reason_distribution(cash_reason_series)

    rule_excess_cagr = bool(excess_cagr > float(go_min_excess_cagr))
    rule_info_ratio = bool(info_ratio > float(go_min_info_ratio))
    rule_maxdd_vs_benchmark = bool(maxdd_gap_vs_benchmark <= (float(go_maxdd_gap_pp) / 100.0))
    if cash_share <= float(go_max_cash_share):
        rule_cash_share_reasons = True
    else:
        rule_cash_share_reasons = bool(
            (cash_risk_reason_share >= float(go_high_cash_risk_reason_min_share)) and
            (cash_datamangler_share <= float(go_high_cash_max_datamangler_share))
        )

    failures: list[str] = []
    if not rule_excess_cagr:
        failures.append("excess_cagr")
    if not rule_info_ratio:
        failures.append("info_ratio")
    if not rule_maxdd_vs_benchmark:
        failures.append("maxdd_vs_benchmark")
    if not rule_cash_share_reasons:
        failures.append("cash_share_reasons")
    go_no_go = "GO" if not failures else "NO_GO"

    return {
        "folds": int(len(by_year)),
        "cagr": cagr,
        "sharpe": _annualized_sharpe(by_year["ret"]),
        "max_dd": float(by_year["max_dd"].min()),
        "hitrate": float((by_year["ret"] > 0).mean()),
        "turnover": float(by_year["turnover"].mean()),
        "costs": float(by_year["turnover"].mean() * float(cost_rate)),
        "cagr_net": cagr_net,
        "benchmark_cagr": benchmark_cagr,
        "benchmark_maxdd": benchmark_maxdd,
        "excess_cagr": excess_cagr,
        "info_ratio": info_ratio,
        "cash_share": cash_share,
        "cash_risk_reason_share": cash_risk_reason_share,
        "cash_datamangler_share": cash_datamangler_share,
        "cash_reason_distribution": json.dumps(reason_dist, sort_keys=True, ensure_ascii=True),
        "maxdd_gap_vs_benchmark": maxdd_gap_vs_benchmark,
        "rule_excess_cagr": bool(rule_excess_cagr),
        "rule_info_ratio": bool(rule_info_ratio),
        "rule_maxdd_vs_benchmark": bool(rule_maxdd_vs_benchmark),
        "rule_cash_share_reasons": bool(rule_cash_share_reasons),
        "go_no_go": go_no_go,
        "go_no_go_reasons": "|".join(failures),
    }


def _write_metadata(
    metadata_path: Path,
    *,
    asof: str | None,
    start: int,
    end: int,
    rebalance: str,
    train_years: int,
    test_years: int,
    seed: int,
    git_commit: str,
    config_path: Path,
    run_dir: Path,
    cost_bps: float,
    cost_model_note: str,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    payload = {
        "asof": asof,
        "start": int(start),
        "end": int(end),
        "rebalance": str(rebalance),
        "train_years": int(train_years),
        "test_years": int(test_years),
        "seed": int(seed),
        "git_commit": str(git_commit),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "cost_bps_per_100pct_turnover": float(cost_bps),
        "cost_model_note": str(cost_model_note),
        "command": command,
        "returncode": int(returncode),
        "stdout_tail": (stdout or "")[-1500:],
        "stderr_tail": (stderr or "")[-1500:],
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_numeric_dtype(d[col]):
            d[col] = d[col].map(lambda v: f"{float(v):.6f}" if pd.notna(v) else "")
        else:
            d[col] = d[col].astype(str)
    header = "| " + " | ".join(d.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    lines = [header, sep]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run walk-forward test sweep over explicit seeds.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--asof", default=None, help="Snapshot date YYYY-MM-DD")
    p.add_argument("--rebalance", default="monthly", choices=["monthly"])
    p.add_argument("--test-window-years", type=int, default=1)
    p.add_argument("--train-window-years", type=int, default=8)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--go-min-excess-cagr", type=float, default=0.0)
    p.add_argument("--go-min-info-ratio", type=float, default=0.20)
    p.add_argument("--go-maxdd-gap-pp", type=float, default=7.5, help="Max allowed strategy drawdown gap vs benchmark in percentage points.")
    p.add_argument("--go-max-cash-share", type=float, default=0.50)
    p.add_argument("--go-high-cash-risk-reason-min-share", type=float, default=0.60)
    p.add_argument("--go-high-cash-max-datamangler-share", type=float, default=0.25)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--master-path", default=None)
    p.add_argument("--prices-path", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.start) > int(args.end):
        raise ValueError("--start must be <= --end")
    if int(args.test_window_years) != 1:
        raise ValueError("--test-window-years must be 1 for annual test folds")
    if str(args.rebalance).lower() != "monthly":
        raise ValueError("--rebalance must be monthly")
    if float(args.go_maxdd_gap_pp) < 0:
        raise ValueError("--go-maxdd-gap-pp must be >= 0")
    for name in (
        "go_max_cash_share",
        "go_high_cash_risk_reason_min_share",
        "go_high_cash_max_datamangler_share",
    ):
        val = float(getattr(args, name))
        if val < 0 or val > 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")

    config_path = _resolve_path(PROJECT_ROOT, args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(config_path)
    paths = resolve_paths(cfg, PROJECT_ROOT)
    runs_dir = paths["runs_dir"]

    output_dir = _resolve_path(PROJECT_ROOT, args.output_dir)
    sweep_root = output_dir if output_dir else (runs_dir / f"wft_sweep_{time.strftime('%Y%m%d_%H%M%S')}")
    sweep_root.mkdir(parents=True, exist_ok=True)
    train_root = sweep_root / f"train_{int(args.train_window_years)}"
    train_root.mkdir(parents=True, exist_ok=True)

    master_path = _resolve_path(PROJECT_ROOT, args.master_path)
    prices_path = _resolve_path(PROJECT_ROOT, args.prices_path)
    seeds = _parse_seed_list(args.seeds)
    git_commit = _detect_git_commit(PROJECT_ROOT)
    cost_rate = float(args.cost_bps) / 10000.0
    cost_model_note = (
        "Placeholder transaction cost model: fixed bps applied per 100% turnover "
        "(until Nordea-specific model is available)."
    )

    rows: list[dict] = []
    for seed in seeds:
        seed_dir = train_root / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)

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
            str(int(seed)),
            "--run-dir",
            str(seed_dir),
        ]
        if args.asof:
            cmd.extend(["--asof", str(args.asof)])
        if master_path:
            cmd.extend(["--master-path", str(master_path)])
        if prices_path:
            cmd.extend(["--prices-path", str(prices_path)])

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(int(seed))

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        metadata_path = seed_dir / "metadata.json"
        _write_metadata(
            metadata_path=metadata_path,
            asof=str(args.asof) if args.asof else None,
            start=int(args.start),
            end=int(args.end),
            rebalance="monthly",
            train_years=int(args.train_window_years),
            test_years=int(args.test_window_years),
            seed=int(seed),
            git_commit=git_commit,
            config_path=config_path,
            run_dir=seed_dir,
            cost_bps=float(args.cost_bps),
            cost_model_note=cost_model_note,
            command=cmd,
            returncode=int(proc.returncode),
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

        base_row = {
            "seed": int(seed),
            "status": "ok" if int(proc.returncode) == 0 else "failed",
            "returncode": int(proc.returncode),
            "run_dir": str(seed_dir),
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "hitrate": 0.0,
            "turnover": 0.0,
            "costs": 0.0,
            "cagr_net": 0.0,
            "benchmark_cagr": 0.0,
            "benchmark_maxdd": 0.0,
            "excess_cagr": 0.0,
            "info_ratio": 0.0,
            "cash_share": 0.0,
            "cash_risk_reason_share": 0.0,
            "cash_datamangler_share": 0.0,
            "cash_reason_distribution": "{}",
            "maxdd_gap_vs_benchmark": 0.0,
            "rule_excess_cagr": False,
            "rule_info_ratio": False,
            "rule_maxdd_vs_benchmark": False,
            "rule_cash_share_reasons": True,
            "go_no_go": "NO_GO",
            "go_no_go_reasons": "run_failed",
            "folds": 0,
            "stderr_tail": (proc.stderr or "").strip()[-1000:],
        }

        results_path = seed_dir / "wft_results.csv"
        if int(proc.returncode) == 0 and results_path.exists():
            metrics = _aggregate_seed_metrics(
                results_path,
                cost_rate=cost_rate,
                go_min_excess_cagr=float(args.go_min_excess_cagr),
                go_min_info_ratio=float(args.go_min_info_ratio),
                go_maxdd_gap_pp=float(args.go_maxdd_gap_pp),
                go_max_cash_share=float(args.go_max_cash_share),
                go_high_cash_risk_reason_min_share=float(args.go_high_cash_risk_reason_min_share),
                go_high_cash_max_datamangler_share=float(args.go_high_cash_max_datamangler_share),
            )
            base_row.update(metrics)

        rows.append(base_row)

    summary_df = pd.DataFrame(rows).sort_values("seed", kind="mergesort").reset_index(drop=True)
    ok_df = summary_df[summary_df["status"] == "ok"].copy()
    if not ok_df.empty:
        ok_df = ok_df.sort_values(
            by=["sharpe", "excess_cagr", "max_dd", "turnover", "costs"],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ok_df["risk_adjusted_rank"] = ok_df.index + 1
        summary_df = summary_df.merge(ok_df[["seed", "risk_adjusted_rank"]], on="seed", how="left")
    else:
        summary_df["risk_adjusted_rank"] = pd.NA

    summary_csv = sweep_root / "sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    md_lines = [
        "# WFT Seed Sweep Summary",
        "",
        f"- Output dir: `{sweep_root}`",
        f"- Train dir: `{train_root}`",
        f"- Config: `{config_path}`",
        f"- Asof: `{args.asof}`",
        f"- Period: {int(args.start)}-{int(args.end)}",
        f"- Rebalance: monthly",
        f"- Train/Test windows: {int(args.train_window_years)}y / {int(args.test_window_years)}y",
        f"- Seeds: {', '.join(str(int(s)) for s in seeds)}",
        f"- Cost assumption: {float(args.cost_bps):.1f} bps per 100% turnover (placeholder model)",
        f"- Cost model note: {cost_model_note}",
        (
            "- GO/NO-GO rule: excess_cagr > "
            f"{float(args.go_min_excess_cagr):.2%}, info_ratio > {float(args.go_min_info_ratio):.2f}, "
            f"maxdd gap <= {float(args.go_maxdd_gap_pp):.2f}pp, "
            f"CASH share <= {float(args.go_max_cash_share):.0%} or "
            f"(risk-reasons >= {float(args.go_high_cash_risk_reason_min_share):.0%} and "
            f"datamangler <= {float(args.go_high_cash_max_datamangler_share):.0%})."
        ),
        "- Risk-adjusted ranking: Sharpe, tie-break by excess_cagr, max_dd, turnover, costs.",
        "",
        "## Per Seed",
    ]

    table_cols = [
        "seed",
        "status",
        "risk_adjusted_rank",
        "cagr",
        "sharpe",
        "max_dd",
        "hitrate",
        "turnover",
        "costs",
        "cagr_net",
        "benchmark_cagr",
        "benchmark_maxdd",
        "excess_cagr",
        "info_ratio",
        "cash_share",
        "maxdd_gap_vs_benchmark",
        "go_no_go",
        "go_no_go_reasons",
        "folds",
        "run_dir",
    ]
    md_lines.append(_markdown_table(summary_df[table_cols]))
    md_lines.append("")
    md_lines.append("## Best Risk-Adjusted Seed")
    if ok_df.empty:
        md_lines.append("No successful seed run.")
    else:
        best = ok_df.iloc[0]
        md_lines.append(f"- Seed: `{int(best['seed'])}`")
        md_lines.append(
            f"- Metrics: Sharpe={_safe_float(best['sharpe']):.4f}, "
            f"CAGR net={_safe_float(best['cagr_net']):.2%}, "
            f"Benchmark CAGR={_safe_float(best['benchmark_cagr']):.2%}, "
            f"Excess CAGR={_safe_float(best['excess_cagr']):.2%}, "
            f"InfoRatio={_safe_float(best['info_ratio']):.4f}, "
            f"MaxDD={_safe_float(best['max_dd']):.2%}, "
            f"DD gap vs benchmark={_safe_float(best['maxdd_gap_vs_benchmark']):.2%}, "
            f"CASH share={_safe_float(best['cash_share']):.2%}, "
            f"Hitrate={_safe_float(best['hitrate']):.2%}, "
            f"Turnover={_safe_float(best['turnover']):.2%}, "
            f"Costs={_safe_float(best['costs']):.2%}"
        )
        md_lines.append(
            f"- GO/NO-GO: `{best['go_no_go']}` (reasons: `{best['go_no_go_reasons']}`)"
        )
        md_lines.append(
            f"- CASH decision reasons distribution: `{best['cash_reason_distribution']}`"
        )
        md_lines.append(f"- Run dir: `{best['run_dir']}`")

    summary_md = sweep_root / "sweep_summary.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"OK: {summary_csv}")
    print(f"OK: {summary_md}")
    return 0 if not ok_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
