from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

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
            "intrinsic_equity": [160_000_000_000.0],
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

    months = pd.date_range("2001-01-31", "2013-12-31", freq="ME")
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
        idx = 1000.0 + float(i)
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


def test_wft_sweep_smoke(tmp_path):
    cfg_path, master_path, prices_path = _write_smoke_dataset(tmp_path)
    out_dir = tmp_path / "sweep_out"

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_wft_sweep.py"),
        "--config",
        str(cfg_path),
        "--start",
        "2013",
        "--end",
        "2013",
        "--rebalance",
        "monthly",
        "--test-window-years",
        "1",
        "--train-window-years",
        "12",
        "--cost-bps",
        "20",
        "--master-path",
        str(master_path),
        "--prices-path",
        str(prices_path),
        "--output-dir",
        str(out_dir),
        "--max-combos",
        "2",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    results_path = out_dir / "sweep_results.csv"
    summary_path = out_dir / "sweep_summary.md"

    assert results_path.exists()
    assert summary_path.exists()

    df = pd.read_csv(results_path)
    expected = {
        "config_id",
        "params",
        "oos_cagr",
        "oos_maxdd",
        "oos_worst_year",
        "oos_turnover",
        "oos_pct_cash",
        "oos_cagr_gross",
        "oos_cagr_net",
    }
    assert expected.issubset(set(df.columns))
