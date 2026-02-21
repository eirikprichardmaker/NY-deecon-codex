from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import run_weekly
from src.build_master import _ensure_price_features, _norm_ticker, _normalize_prices_with_mapping
from src.pipeline import apply_screen, build_shortlist, validate_contract


class _Log:
    def info(self, *_args, **_kwargs):
        return None


def test_ticker_normalize_case_whitespace():
    assert _norm_ticker("  eqnr.ol ") == "EQNR"
    assert _norm_ticker("^oseax") == "OSEAX"


def test_merge_prices_with_missing_yahoo_ticker_uses_mapping():
    prices = pd.DataFrame(
        {
            "ticker": ["EQNR.OL", " MOWI.OL "],
            "date": ["2026-01-01", "2026-01-01"],
            "adj_close": [100, 200],
            "yahoo_ticker": ["", ""],
        }
    )
    mapping = pd.DataFrame({"ticker_norm": ["EQNR", "MOWI"], "yahoo_ticker": ["EQNR.OL", "MOWI.OL"]})
    out = _normalize_prices_with_mapping(prices, mapping, _Log())
    assert set(out["yahoo_ticker"].tolist()) == {"EQNR", "MOWI"} or set(out["yahoo_ticker"].tolist()) == {"EQNR.OL", "MOWI.OL"}
    assert out["missing_price_reason"].eq("").all()


def test_ma200_mad_computed_on_synthetic_series():
    n = 220
    prices = pd.DataFrame(
        {
            "ticker_norm": ["EQNR"] * n,
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "adj_close": np.arange(1, n + 1, dtype=float),
        }
    )
    out = _ensure_price_features(prices)
    last = out.iloc[-1]
    assert np.isfinite(last["ma200"])
    assert np.isfinite(last["mad"])
    assert bool(last["above_ma200"]) is True


def test_ma200_warmup_is_nan_for_first_199_observations():
    n = 220
    prices = pd.DataFrame(
        {
            "ticker_norm": ["EQNR"] * n,
            "date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "adj_close": np.arange(1, n + 1, dtype=float),
            "ma200": np.arange(n, dtype=float),
            "mad": np.arange(n, dtype=float),
            "above_ma200": [True] * n,
        }
    )
    out = _ensure_price_features(prices)
    assert pd.isna(out["ma200"].iloc[198])
    assert pd.notna(out["ma200"].iloc[199])


def test_fail_fast_rules_cash_when_no_candidate_passes():
    master = pd.DataFrame(
        {
            "yahoo_ticker": ["EQNR.OL", "MOWI.OL"],
            "Company": ["Eq", "Mo"],
            "ROIC - Current": [0.20, 0.01],
            "EV/EBIT - Current": [10.0, 12.0],
            "N.Debt/Ebitda - Current": [2.0, 5.0],
            "above_ma200": [False, False],
            "price_date": [pd.Timestamp("2026-01-01")] * 2,
            "price": [100.0, 50.0],
            "ma200": [120.0, 60.0],
            "mad": [-0.1, -0.1],
            "Info - Country": ["NO", "NO"],
            "Market Cap - Current": [10, 8],
            "OCF - Millions": [1, 1],
            "Capex - Millions": [1, 1],
            "FCF - Millions": [1, 1],
            "Net Debt - Current": [1, 1],
            "P/E - Current": [1, 1],
            "P/S - Current": [1, 1],
        }
    )
    thresholds = {"fundamentals": {"roic_min": 0.05, "ev_ebit_min": 5, "nd_ebitda_max": 4}}
    screened = apply_screen(master, thresholds)
    shortlist = build_shortlist(screened)
    assert shortlist.empty


def test_schema_checks_required_columns_master_latest():
    row = {
        "yahoo_ticker": "EQNR.OL",
        "Company": "Equinor",
        "Info - Country": "NO",
        "Market Cap - Current": 10,
        "OCF - Millions": 1,
        "Capex - Millions": 1,
        "FCF - Millions": 1,
        "Net Debt - Current": 1,
        "N.Debt/Ebitda - Current": 1,
        "ROIC - Current": 0.1,
        "EV/EBIT - Current": 9,
        "P/E - Current": 8,
        "P/S - Current": 2,
        "price_date": pd.Timestamp("2026-01-01"),
        "price": 100,
        "ma200": 90,
        "mad": 0.1,
        "above_ma200": True,
        "fundamental_ok": True,
        "technical_ok": True,
        "decision": "CANDIDATE",
    }
    df = pd.DataFrame([row])
    validate_contract(df)


def test_run_weekly_smoke_creates_run_dir_and_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "config" / "config.yaml").write_text("paths:\n  runs_dir: runs\n", encoding="utf-8")

    dummy = types.ModuleType("src.dummy_step")
    def _run(ctx, log):
        (ctx.run_dir / "valuation.csv").write_text("ticker,intrinsic_equity\nEQNR,1\n", encoding="utf-8")
        (ctx.run_dir / "decision.md").write_text("# Decision", encoding="utf-8")
        (ctx.run_dir / "quality.md").write_text("# Quality", encoding="utf-8")
        return 0
    dummy.run = _run
    sys.modules["src.dummy_step"] = dummy

    monkeypatch.setattr(run_weekly, "DEFAULT_STEPS", [("valuation", "src.dummy_step"), ("decision", "src.dummy_step")])
    monkeypatch.setattr(sys, "argv", ["run_weekly", "--asof", "2026-02-16", "--config", "config/config.yaml", "--steps", "valuation,decision"])

    from src.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "project_root_from_file", lambda: root)
    rc = run_weekly.main()
    assert rc == 0

    run_dirs = sorted((root / "runs").glob("20260216_*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    assert (run_dir / "valuation.csv").exists()
    assert (run_dir / "decision.md").exists()
    assert (run_dir / "quality.md").exists()
