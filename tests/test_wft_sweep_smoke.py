from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_sweep_module():
    p = ROOT / "tools" / "run_wft_sweep.py"
    spec = importlib.util.spec_from_file_location("run_wft_sweep_mod", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_reason_distribution_counts_new_data_missing_reasons():
    mod = _load_sweep_module()
    dist, risk_share, data_missing_share = mod._reason_distribution(
        pd.Series(
            [
                "DATA_MISSING_MA200|ingen kandidat",
                "DATA_MISSING_BENCHMARK|ingen kandidat",
                ">200d|ingen kandidat",
            ]
        )
    )

    assert abs(float(risk_share) - (1.0 / 3.0)) < 1e-12
    assert abs(float(data_missing_share) - (2.0 / 3.0)) < 1e-12
    assert float(dist.get("DATA_MISSING_MA200", 0.0)) > 0.0
    assert float(dist.get("DATA_MISSING_BENCHMARK", 0.0)) > 0.0


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
        "--asof",
        "2013-12-31",
        "--rebalance",
        "monthly",
        "--test-window-years",
        "1",
        "--train-window-years",
        "12",
        "--seeds",
        "11",
        "--cost-bps",
        "20",
        "--master-path",
        str(master_path),
        "--prices-path",
        str(prices_path),
        "--output-dir",
        str(out_dir),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    seed_dir = out_dir / "train_12" / "seed_11"
    wft_results_path = seed_dir / "wft_results.csv"
    wft_summary_path = seed_dir / "wft_summary.md"
    tuned_cfg_path = seed_dir / "tuned_config.yaml"
    metadata_path = seed_dir / "metadata.json"
    summary_csv = out_dir / "sweep_summary.csv"
    summary_md = out_dir / "sweep_summary.md"

    assert wft_results_path.exists()
    assert wft_summary_path.exists()
    assert tuned_cfg_path.exists()
    assert metadata_path.exists()
    assert summary_csv.exists()
    assert summary_md.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_meta = {
        "asof",
        "start",
        "end",
        "rebalance",
        "train_years",
        "test_years",
        "seed",
        "git_commit",
        "config_path",
        "cost_bps_per_100pct_turnover",
        "cost_model_note",
    }
    assert required_meta.issubset(set(metadata.keys()))
    assert metadata["asof"] == "2013-12-31"
    assert int(metadata["seed"]) == 11
    assert int(metadata["start"]) == 2013
    assert int(metadata["end"]) == 2013

    df = pd.read_csv(summary_csv)
    expected = {
        "seed",
        "status",
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
        "cash_risk_reason_share",
        "cash_datamangler_share",
        "cash_reason_distribution",
        "maxdd_gap_vs_benchmark",
        "rule_excess_cagr",
        "rule_info_ratio",
        "rule_maxdd_vs_benchmark",
        "rule_cash_share_reasons",
        "go_no_go",
        "go_no_go_reasons",
        "run_dir",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) == 1
    assert int(df.iloc[0]["seed"]) == 11
    assert str(df.iloc[0]["go_no_go"]) in {"GO", "NO_GO"}
