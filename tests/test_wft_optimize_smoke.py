from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _write_smoke_dataset(base: Path) -> tuple[Path, Path, Path]:
    cfg_path = base / "config.yaml"
    cfg_path.write_text(
        "paths:\n"
        "  runs_dir: runs\n"
        "  data_dir: data\n"
        "  raw_dir: data/raw\n"
        "  processed_dir: data/processed\n",
        encoding="utf-8",
    )

    processed = base / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    master = pd.DataFrame(
        {
            "ticker": ["EQNR"],
            "yahoo_ticker": ["EQNR.OL"],
            "company": ["Equinor"],
            "model": ["DCF"],
            "market_cap_current": [100_000_000_000.0],
            "intrinsic_equity": [140_000_000_000.0],
            "roic": [0.12],
            "wacc_used": [0.07],
            "fcf_yield": [0.05],
            "nd_ebitda": [1.0],
            "ev_ebit": [10.0],
            "beta": [1.0],
        }
    )
    master_path = processed / "master_valued.parquet"
    master.to_parquet(master_path, index=False)

    months = pd.date_range("2001-01-31", "2008-12-31", freq="ME")
    rows = []
    for i, dt in enumerate(months):
        px = 100.0 + float(i)
        rows.append(
            {
                "ticker": "EQNR.OL",
                "date": dt,
                "adj_close": px,
                "ma21": px * 0.99,
                "ma200": px * 0.95,
                "mad": 0.03,
                "above_ma200": True,
            }
        )
        idx = 1000.0 + float(i) * 2.0
        rows.append(
            {
                "ticker": "^OSEAX",
                "date": dt,
                "adj_close": idx,
                "ma21": idx * 0.99,
                "ma200": idx * 0.95,
                "mad": 0.03,
                "above_ma200": True,
            }
        )

    prices_path = processed / "prices.parquet"
    pd.DataFrame(rows).to_parquet(prices_path, index=False)
    return cfg_path, master_path, prices_path


def test_wft_optimize_smoke_outputs_and_determinism(tmp_path):
    cfg_path, master_path, prices_path = _write_smoke_dataset(tmp_path)
    out1 = tmp_path / "opt_out_1"
    out2 = tmp_path / "opt_out_2"

    cmd_base = [
        sys.executable,
        str(ROOT / "tools" / "run_wft_optimize.py"),
        "--config",
        str(cfg_path),
        "--start",
        "2006",
        "--end",
        "2008",
        "--train-window-years",
        "2",
        "--test-window-years",
        "1",
        "--rebalance",
        "monthly",
        "--n-trials",
        "6",
        "--seed",
        "7",
        "--master-path",
        str(master_path),
        "--prices-path",
        str(prices_path),
    ]

    proc1 = subprocess.run(cmd_base + ["--output-dir", str(out1)], cwd=str(ROOT), capture_output=True, text=True)
    assert proc1.returncode == 0, proc1.stderr

    trials1 = out1 / "trials.csv"
    best1 = out1 / "best_config.yaml"
    summary1 = out1 / "optimize_summary.md"
    assert trials1.exists()
    assert best1.exists()
    assert summary1.exists()

    df = pd.read_csv(trials1)
    assert not df.empty
    expected_cols = {
        "trial_id",
        "fold_id",
        "test_year",
        "is_selected_train",
        "mos_threshold",
        "mad_min",
        "weakness_rule_variant",
        "train_objective",
        "train_return",
        "train_sharpe",
        "train_turnover",
        "train_pct_cash",
        "test_return_gross",
        "test_return_net",
        "benchmark_return",
        "excess_return_net",
        "test_max_dd",
        "test_turnover",
        "test_pct_cash",
        "candidate_or_cash",
        "benchmark_symbols_used",
        "benchmark_missing_months",
        "trade_count",
    }
    assert expected_cols.issubset(set(df.columns))

    proc2 = subprocess.run(cmd_base + ["--output-dir", str(out2)], cwd=str(ROOT), capture_output=True, text=True)
    assert proc2.returncode == 0, proc2.stderr

    cfg1 = yaml.safe_load(best1.read_text(encoding="utf-8"))
    cfg2 = yaml.safe_load((out2 / "best_config.yaml").read_text(encoding="utf-8"))
    assert cfg1["selected_params"] == cfg2["selected_params"]
