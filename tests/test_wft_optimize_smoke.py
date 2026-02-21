from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_optimizer_module():
    p = ROOT / "tools" / "run_wft_optimize.py"
    spec = importlib.util.spec_from_file_location("run_wft_optimize_mod", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def _build_hysteresis_window() -> pd.DataFrame:
    months = pd.date_range("2020-01-31", "2020-11-30", freq="ME")
    rows = []
    for i, m in enumerate(months):
        favored = "AAA" if i % 2 == 0 else "BBB"
        for t in ["AAA", "BBB"]:
            mad = 0.05 if t == favored else -0.20
            px = 100.0 + float(i) + (0.0 if t == "AAA" else 1.0)
            rows.append(
                {
                    "month": m,
                    "ticker": f"{t}.OL",
                    "k": t,
                    "adj_close": px,
                    "ma200": px * 0.95,
                    "above_ma200": True,
                    "mad": mad,
                    "index_price": 1000.0 + float(i),
                    "index_ma200": (1000.0 + float(i)) * 0.95,
                    "index_data_ok": True,
                    "index_above_ma200": True,
                    "index_mad": 0.05,
                    "high_risk_flag": False,
                    "mos": 0.50,
                    "value_creation_ok_base": True,
                    "quality_weak_count": 0,
                    "roic": 0.10,
                    "fcf_yield": 0.05,
                    "beta": 1.0,
                    "nd_ebitda": 1.0,
                    "market_cap": 1_000_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_wft_optimize_smoke_holdout_and_determinism(tmp_path):
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
        "8",
        "--seed",
        "7",
        "--holdout-start",
        "2007",
        "--holdout-end",
        "2008",
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
    holdout1 = out1 / "holdout_results.csv"
    assert trials1.exists()
    assert best1.exists()
    assert summary1.exists()
    assert holdout1.exists()

    df1 = pd.read_csv(trials1)
    assert not df1.empty
    expected_cols = {
        "trial_id",
        "fold_id",
        "test_year",
        "is_selected_train",
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
        "weights_reg_penalty",
        "train_objective",
        "test_return_net",
        "benchmark_return",
        "excess_return_net",
        "test_turnover",
        "test_pct_cash",
        "trade_count",
    }
    assert expected_cols.issubset(set(df1.columns))

    holdout_df = pd.read_csv(holdout1)
    assert set(["mode", "test_year", "chosen_params", "return", "max_dd", "turnover", "pct_cash"]).issubset(set(holdout_df.columns))

    md = summary1.read_text(encoding="utf-8")
    assert "## Holdout Evaluation" in md

    proc2 = subprocess.run(cmd_base + ["--output-dir", str(out2)], cwd=str(ROOT), capture_output=True, text=True)
    assert proc2.returncode == 0, proc2.stderr

    cfg1 = yaml.safe_load(best1.read_text(encoding="utf-8"))
    cfg2 = yaml.safe_load((out2 / "best_config.yaml").read_text(encoding="utf-8"))
    assert cfg1 == cfg2

    df2 = pd.read_csv(out2 / "trials.csv")
    pdt.assert_frame_equal(df1, df2, check_dtype=False)


def test_hysteresis_min_hold_sanity():
    opt = _load_optimizer_module()
    window_df = _build_hysteresis_window()

    params_fast = opt.StrategyAParams(
        mos_threshold=0.30,
        mad_min=0.0,
        mad_penalty_k=2.0,
        min_hold_months=0,
        score_gap=0.0,
        weight_quality=0.4,
        weight_value=0.3,
        weight_lowrisk=0.2,
        weight_balance=0.1,
        weakness_rule_variant="baseline",
    )
    params_slow = opt.StrategyAParams(
        mos_threshold=0.30,
        mad_min=0.0,
        mad_penalty_k=2.0,
        min_hold_months=6,
        score_gap=0.0,
        weight_quality=0.4,
        weight_value=0.3,
        weight_lowrisk=0.2,
        weight_balance=0.1,
        weakness_rule_variant="baseline",
    )

    fast = opt._simulate_window_strategy_a(window_df, params_fast)
    slow = opt._simulate_window_strategy_a(window_df, params_slow)

    fast_switches = [int(i) for i in fast.index[fast["position_change"] == 1].tolist()]
    slow_switches = [int(i) for i in slow.index[slow["position_change"] == 1].tolist()]

    assert any(i < 6 for i in fast_switches)
    assert all(i >= 6 for i in slow_switches)


def test_score_month_uses_price_over_ma200_not_input_flags():
    opt = _load_optimizer_module()
    params = opt.StrategyAParams(
        mos_threshold=0.30,
        mad_min=0.0,
        mad_penalty_k=1.0,
        min_hold_months=0,
        score_gap=0.0,
        weight_quality=0.4,
        weight_value=0.3,
        weight_lowrisk=0.2,
        weight_balance=0.1,
        weakness_rule_variant="baseline",
    )
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "k": ["AAA"],
            "high_risk_flag": [False],
            "mos": [0.50],
            "value_creation_ok_base": [True],
            "quality_weak_count": [0],
            "adj_close": [110.0],
            "ma200": [100.0],
            "mad": [0.05],
            "above_ma200": [False],
            "index_price": [1100.0],
            "index_ma200": [1000.0],
            "index_mad": [0.05],
            "index_above_ma200": [False],
            "roic": [0.12],
            "fcf_yield": [0.06],
            "beta": [1.0],
            "nd_ebitda": [1.0],
            "market_cap": [1_000_000_000.0],
        }
    )

    scored = opt._score_month(month_df, params)
    assert bool(scored.loc[0, "technical_ok"]) is True


def test_score_month_two_of_three_allows_stock_trend_fail_when_index_and_mad_pass():
    opt = _load_optimizer_module()
    params = opt.StrategyAParams(
        mos_threshold=0.30,
        mad_min=0.0,
        mad_penalty_k=1.0,
        min_hold_months=0,
        score_gap=0.0,
        weight_quality=0.4,
        weight_value=0.3,
        weight_lowrisk=0.2,
        weight_balance=0.1,
        weakness_rule_variant="baseline",
    )
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "k": ["AAA"],
            "high_risk_flag": [False],
            "mos": [0.50],
            "value_creation_ok_base": [True],
            "quality_weak_count": [0],
            "adj_close": [95.0],
            "ma200": [100.0],
            "mad": [0.05],
            "index_price": [1100.0],
            "index_ma200": [1000.0],
            "index_mad": [0.05],
            "roic": [0.12],
            "fcf_yield": [0.06],
            "beta": [1.0],
            "nd_ebitda": [1.0],
            "market_cap": [1_000_000_000.0],
        }
    )

    scored = opt._score_month(month_df, params)
    assert int(scored.loc[0, "tech_signal_count"]) == 2
    assert bool(scored.loc[0, "technical_ok"]) is True
