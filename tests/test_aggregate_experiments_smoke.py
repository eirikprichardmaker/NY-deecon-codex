from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _write_stub_run(
    runs_dir: Path,
    run_id: str,
    seed: int,
    holdout_excess: float,
    holdout_max_dd: float,
    holdout_turnover: float,
    holdout_pct_cash: float,
    n_trials: int = 50,
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_cfg = {
        "optimizer": {
            "method": "nested_walk_forward_strategy_a",
            "seed": int(seed),
            "n_trials": int(n_trials),
            "start": 2010,
            "end": 2025,
            "train_window_years": 12,
            "test_window_years": 1,
            "step_years": 1,
            "rebalance": "monthly",
            "holdout_start": 2021,
            "holdout_end": 2022,
            "selection_folds": 3,
            "holdout_folds": 2,
        },
        "strategy_a": {
            "weights_reg_lambda": 0.5,
        },
        "selected_params": {
            "mos_threshold": 0.35,
            "mad_min": -0.02,
            "mad_penalty_k": 0.0,
            "min_hold_months": 3,
            "score_gap": 0.25,
            "weights": {
                "quality": 0.4,
                "value": 0.3,
                "lowrisk": 0.2,
                "balance": 0.1,
            },
        },
        "selection_metrics": {
            "trial_id": "trial_0001",
            "median_excess_return_net": 0.01,
            "worst_fold_excess_return_net": -0.01,
            "median_test_max_dd": -0.10,
            "mean_test_turnover": 0.20,
            "mean_test_pct_cash": 0.40,
        },
        "holdout_metrics": {
            "cumulative_return_net": holdout_excess + 0.01,
            "excess_return_net": holdout_excess,
            "max_dd": holdout_max_dd,
            "turnover": holdout_turnover,
            "pct_cash": holdout_pct_cash,
            "switches": 4,
        },
    }
    (run_dir / "best_config.yaml").write_text(yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8")

    trials = pd.DataFrame(
        [
            {
                "trial_id": "trial_0001",
                "is_selected_train": True,
                "excess_return_net": 0.02,
                "test_max_dd": -0.12,
                "test_turnover": 0.20,
                "test_pct_cash": 0.40,
            },
            {
                "trial_id": "trial_0001",
                "is_selected_train": True,
                "excess_return_net": 0.01,
                "test_max_dd": -0.10,
                "test_turnover": 0.18,
                "test_pct_cash": 0.35,
            },
            {
                "trial_id": "trial_0001",
                "is_selected_train": True,
                "excess_return_net": -0.01,
                "test_max_dd": -0.15,
                "test_turnover": 0.23,
                "test_pct_cash": 0.42,
            },
        ]
    )
    trials.to_csv(run_dir / "trials.csv", index=False)

    holdout = pd.DataFrame(
        [
            {
                "mode": "holdout_best",
                "train_start": "2009-12-31",
                "train_end": "2020-12-31",
                "test_year": 2021,
                "chosen_params": "{}",
                "candidate_or_cash": "CANDIDATE",
                "return": holdout_excess / 2.0,
                "max_dd": holdout_max_dd,
                "turnover": holdout_turnover,
                "pct_cash": holdout_pct_cash,
            },
            {
                "mode": "holdout_best",
                "train_start": "2010-12-31",
                "train_end": "2021-12-31",
                "test_year": 2022,
                "chosen_params": "{}",
                "candidate_or_cash": "CANDIDATE",
                "return": holdout_excess / 2.0,
                "max_dd": holdout_max_dd,
                "turnover": holdout_turnover,
                "pct_cash": holdout_pct_cash,
            },
        ]
    )
    holdout.to_csv(run_dir / "holdout_results.csv", index=False)

    (run_dir / "optimize_summary.md").write_text("# Stub summary\n", encoding="utf-8")
    return run_dir


def test_aggregate_experiments_smoke(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "experiments"
    runs_dir.mkdir(parents=True, exist_ok=True)

    _write_stub_run(runs_dir, "wft_opt_20260219_200001", 1, 0.03, -0.10, 0.18, 0.35)
    _write_stub_run(runs_dir, "wft_opt_20260219_200002", 2, 0.02, -0.11, 0.22, 0.45)
    _write_stub_run(runs_dir, "wft_opt_20260219_200003", 3, -0.01, -0.13, 0.25, 0.50)
    _write_stub_run(runs_dir, "wft_opt_20260219_200004", 4, 0.04, -0.09, 0.19, 0.30)
    _write_stub_run(runs_dir, "wft_opt_20260219_200005", 5, 0.01, -0.12, 0.21, 0.42)
    baseline_run = _write_stub_run(runs_dir, "wft_opt_20260219_199900", 999, 0.00, -0.12, 0.20, 0.40, n_trials=1)

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "aggregate_experiments.py"),
        "--mode",
        "strategyA",
        "--runs-dir",
        str(runs_dir),
        "--output-dir",
        str(out_dir),
        "--holdout-start",
        "2021",
        "--holdout-end",
        "2022",
        "--seeds",
        "1,2,3,4,5",
        "--latest",
        "10",
        "--baseline-run-id",
        baseline_run.name,
    ]
    first = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert first.returncode == 0, first.stderr

    csv_path = out_dir / "strategyA_seed_summary.csv"
    md_path = out_dir / "strategyA_seed_summary.md"
    rec_path = out_dir / "strategyA_recommendation.md"
    assert csv_path.exists()
    assert md_path.exists()
    assert rec_path.exists()

    df = pd.read_csv(csv_path)
    assert list(df["seed"]) == [4, 1, 2, 5, 3]
    assert int(df.iloc[0]["rank"]) == 1
    assert int(df.iloc[0]["seed"]) == 4
    assert str(df.iloc[0]["baseline_run_id"]) == baseline_run.name
    assert "delta_holdout_excess_return_net" in set(df.columns)

    rec = rec_path.read_text(encoding="utf-8")
    assert "decision: **freeze config**" in rec

    csv_first = csv_path.read_text(encoding="utf-8")
    md_first = md_path.read_text(encoding="utf-8")
    rec_first = rec_path.read_text(encoding="utf-8")

    second = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert csv_path.read_text(encoding="utf-8") == csv_first
    assert md_path.read_text(encoding="utf-8") == md_first
    assert rec_path.read_text(encoding="utf-8") == rec_first

