from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import wft


def _write_smoke_dataset(root: Path) -> tuple[Path, Path]:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    (root / "config" / "config.yaml").write_text(
        "paths:\n"
        "  runs_dir: runs\n"
        "  data_dir: data\n"
        "  raw_dir: data/raw\n"
        "  processed_dir: data/processed\n",
        encoding="utf-8",
    )

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
    master_path = root / "data" / "processed" / "master_valued.parquet"
    master.to_parquet(master_path, index=False)

    months = pd.date_range("2001-01-31", "2013-12-31", freq="ME")
    rows = []
    for i, dt in enumerate(months):
        price = 100.0 + float(i)
        rows.append(
            {
                "ticker": "EQNR.OL",
                "date": dt,
                "adj_close": price,
                "ma21": price * 0.99,
                "ma200": price * 0.95,
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

    prices_path = root / "data" / "processed" / "prices.parquet"
    pd.DataFrame(rows).to_parquet(prices_path, index=False)
    return master_path, prices_path


def test_no_leakage_split():
    months = pd.date_range("2000-01-31", "2020-12-31", freq="ME")
    folds = wft._build_folds(pd.Series(months), train_years=12, test_years=1, step_years=1)

    assert folds
    years = []
    for f in folds:
        assert f["train_end"] < f["test_start"]
        assert f["train_start"].year == (f["test_year"] - 12)
        assert f["train_end"].year == (f["test_year"] - 1)
        assert f["test_start"].year == f["test_year"]
        assert f["test_end"].year == f["test_year"]
        years.append(f["test_year"])

    assert years == sorted(years)
    assert len(set(years)) == len(years)


def test_tuning_uses_train_only(monkeypatch):
    p_baseline = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    p_alt = wft.WFTParams(mos_threshold=0.45, mad_min=-0.05, weakness_rule_variant="baseline")
    grid = [p_baseline, p_alt]

    def _fake_simulate(window_df: pd.DataFrame, params: wft.WFTParams) -> pd.DataFrame:
        split = str(window_df.iloc[0]["split"])
        if split == "train":
            ret = 0.05 if params.mos_threshold == 0.30 else -0.02
        else:
            ret = 0.12 if params.mos_threshold == 0.45 else -0.03
        return pd.DataFrame(
            {
                "month": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
                "position": ["AAA", "AAA"],
                "ret": [ret, 0.0],
                "position_change": [0, 0],
                "is_cash": [False, False],
            }
        )

    monkeypatch.setattr(wft, "_simulate_window", _fake_simulate)

    train_df = pd.DataFrame({"split": ["train"]})
    test_df = pd.DataFrame({"split": ["test"]})

    chosen, tuned_metrics, _base_metrics, _tuned_trades, _base_trades, _diag = wft.run_fold(
        train_df=train_df,
        test_df=test_df,
        grid=grid,
        baseline=p_baseline,
        seed=42,
    )

    assert chosen == p_baseline

    alt_test_metrics = wft._window_metrics(_fake_simulate(test_df, p_alt))
    assert alt_test_metrics["return"] > tuned_metrics["return"]


def test_determinism_seed(monkeypatch):
    p0 = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    p1 = wft.WFTParams(mos_threshold=0.35, mad_min=-0.02, weakness_rule_variant="baseline")
    p2 = wft.WFTParams(mos_threshold=0.40, mad_min=0.00, weakness_rule_variant="stricter")
    grid = [p0, p1, p2]

    def _fake_simulate(window_df: pd.DataFrame, params: wft.WFTParams) -> pd.DataFrame:
        split = str(window_df.iloc[0]["split"])
        base = 0.02 if split == "train" else 0.01
        alpha = (0.50 - float(params.mos_threshold)) * 0.05
        beta = (-float(params.mad_min)) * 0.01
        gamma = -0.002 if params.weakness_rule_variant == "stricter" else 0.0
        ret = base + alpha + beta + gamma
        return pd.DataFrame(
            {
                "month": [pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-28")],
                "position": ["AAA", "AAA"],
                "ret": [ret, ret / 2.0],
                "position_change": [0, 0],
                "is_cash": [False, False],
            }
        )

    monkeypatch.setattr(wft, "_simulate_window", _fake_simulate)

    train_df = pd.DataFrame({"split": ["train"]})
    test_df = pd.DataFrame({"split": ["test"]})

    out1 = wft.run_fold(
        train_df=train_df,
        test_df=test_df,
        grid=grid,
        baseline=p0,
        seed=42,
    )
    out2 = wft.run_fold(
        train_df=train_df,
        test_df=test_df,
        grid=grid,
        baseline=p0,
        seed=42,
    )

    assert out1[0] == out2[0]
    assert out1[1] == out2[1]
    assert out1[2] == out2[2]
    pd.testing.assert_frame_equal(out1[3], out2[3], check_dtype=False)
    pd.testing.assert_frame_equal(out1[4], out2[4], check_dtype=False)
    pd.testing.assert_frame_equal(out1[5], out2[5], check_dtype=False)


def test_cash_reasons_include_quality_filter_failure():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL", "BBB.OL"],
            "k": ["AAA", "BBB"],
            "mos": [0.50, 0.55],
            "high_risk_flag": [False, False],
            "value_creation_ok_base": [True, True],
            "quality_weak_count": [3, 4],
            "adj_close": [110.0, 120.0],
            "above_ma200": [True, True],
            "ma200": [100.0, 100.0],
            "mad": [0.03, 0.02],
            "index_price": [1000.0, 1000.0],
            "index_ma200": [900.0, 900.0],
            "index_data_ok": [True, True],
            "index_above_ma200": [True, True],
            "index_mad": [0.03, 0.03],
            "roic": [0.10, 0.11],
            "fcf_yield": [0.05, 0.06],
            "market_cap": [10_000_000_000.0, 9_000_000_000.0],
        }
    )

    position, reasons = wft._pick_ticker_with_reason(month_df, params)

    assert position == "CASH"
    assert "kvalitetsscore" in reasons
    assert "ingen kandidat" in reasons


def test_cash_reasons_include_data_missing_ma200_on_warmup_without_200d():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL", "BBB.OL"],
            "k": ["AAA", "BBB"],
            "mos": [0.50, 0.55],
            "high_risk_flag": [False, False],
            "value_creation_ok_base": [True, True],
            "quality_weak_count": [0, 0],
            "adj_close": [110.0, 120.0],
            "ma200": [None, None],
            "mad": [None, None],
            "index_price": [1000.0, 1000.0],
            "index_ma200": [None, None],
            "index_mad": [None, None],
            "roic": [0.10, 0.11],
            "fcf_yield": [0.05, 0.06],
            "market_cap": [10_000_000_000.0, 9_000_000_000.0],
        }
    )

    position, reasons = wft._pick_ticker_with_reason(month_df, params)

    assert position == "CASH"
    assert "DATA_MISSING_MA200" in reasons
    assert ">200d" not in reasons


def test_cash_reasons_include_data_missing_benchmark_when_mapping_missing():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.XX", "BBB.XX"],
            "k": ["AAA", "BBB"],
            "relevant_index_key": ["", ""],
            "mos": [0.50, 0.55],
            "high_risk_flag": [False, False],
            "value_creation_ok_base": [True, True],
            "quality_weak_count": [0, 0],
            "adj_close": [110.0, 120.0],
            "ma200": [100.0, 100.0],
            "mad": [0.03, 0.02],
            "index_price": [None, None],
            "index_ma200": [None, None],
            "index_mad": [None, None],
            "roic": [0.10, 0.11],
            "fcf_yield": [0.05, 0.06],
            "market_cap": [10_000_000_000.0, 9_000_000_000.0],
        }
    )

    position, reasons = wft._pick_ticker_with_reason(month_df, params)

    assert position == "CASH"
    assert "DATA_MISSING_BENCHMARK" in reasons
    assert ">200d" not in reasons


def test_apply_filters_uses_price_over_ma200_for_stock_and_index():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "k": ["AAA"],
            "mos": [0.50],
            "high_risk_flag": [False],
            "value_creation_ok_base": [True],
            "quality_weak_count": [0],
            "adj_close": [110.0],
            "ma200": [100.0],
            "mad": [0.03],
            "above_ma200": [False],
            "index_price": [1100.0],
            "index_ma200": [1000.0],
            "index_mad": [0.02],
            "index_above_ma200": [False],
            "roic": [0.10],
            "fcf_yield": [0.05],
            "market_cap": [10_000_000_000.0],
        }
    )

    out = wft._apply_filters(month_df, params)
    assert bool(out.loc[0, "stock_technical_ok"]) is True
    assert bool(out.loc[0, "index_technical_ok"]) is True
    assert bool(out.loc[0, "technical_ok"]) is True


def test_apply_filters_two_of_three_allows_stock_trend_fail_when_index_and_mad_pass():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "k": ["AAA"],
            "mos": [0.50],
            "high_risk_flag": [False],
            "value_creation_ok_base": [True],
            "quality_weak_count": [0],
            "adj_close": [95.0],
            "ma200": [100.0],
            "mad": [0.02],
            "index_price": [1100.0],
            "index_ma200": [1000.0],
            "index_mad": [0.01],
            "roic": [0.10],
            "fcf_yield": [0.05],
            "market_cap": [10_000_000_000.0],
        }
    )

    out = wft._apply_filters(month_df, params)
    assert bool(out.loc[0, "stock_technical_ok"]) is False
    assert bool(out.loc[0, "tech_signal_mad"]) is True
    assert int(out.loc[0, "tech_signal_count"]) == 2
    assert bool(out.loc[0, "technical_ok"]) is True


def test_apply_filters_two_of_three_blocks_when_only_index_trend_passes():
    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    month_df = pd.DataFrame(
        {
            "ticker": ["AAA.OL"],
            "k": ["AAA"],
            "mos": [0.50],
            "high_risk_flag": [False],
            "value_creation_ok_base": [True],
            "quality_weak_count": [0],
            "adj_close": [95.0],
            "ma200": [100.0],
            "mad": [-0.20],
            "index_price": [1100.0],
            "index_ma200": [1000.0],
            "index_mad": [0.01],
            "roic": [0.10],
            "fcf_yield": [0.05],
            "market_cap": [10_000_000_000.0],
        }
    )

    out = wft._apply_filters(month_df, params)
    assert int(out.loc[0, "tech_signal_count"]) == 1
    assert bool(out.loc[0, "technical_ok"]) is False


def test_wft_cli_smoke_outputs_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    master_path, prices_path = _write_smoke_dataset(root)

    monkeypatch.setattr(wft, "project_root_from_file", lambda: root)
    monkeypatch.setattr(
        wft,
        "_iter_param_grid",
        lambda *_args, **_kwargs: [wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wft",
            "--config",
            "config/config.yaml",
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
            "--master-path",
            str(master_path),
            "--prices-path",
            str(prices_path),
        ],
    )

    rc = wft.main()
    assert rc == 0

    run_dirs = sorted((root / "runs").glob("wft_2013_2013_*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    results_path = run_dir / "wft_results.csv"
    summary_path = run_dir / "wft_summary.md"

    assert results_path.exists()
    assert summary_path.exists()

    out = pd.read_csv(results_path)
    assert not out.empty
    expected_cols = {
        "mode",
        "train_start",
        "train_end",
        "test_year",
        "chosen_params",
        "candidate_or_cash",
        "return",
        "max_dd",
        "turnover",
        "pct_cash",
    }
    assert expected_cols.issubset(set(out.columns))

    md = summary_path.read_text(encoding="utf-8")
    assert "CAGR" in md
    assert "MaxDD" in md
