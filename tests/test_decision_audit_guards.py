from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (
    _apply_index_technical_filter,
    _candidate_data_sufficiency,
    _decision_schema_rows,
    _run_media_red_flag_scan,
)


def test_candidate_data_sufficiency_flags_missing_fields() -> None:
    df = pd.DataFrame(
        {
            "intrinsic_value": [10.0],
            "market_cap": [8.0],
            "mos": [0.25],
            "quality_score": [0.2],
        }
    )
    out, req, min_count, min_ratio = _candidate_data_sufficiency(
        df,
        {
            "candidate_required_fields": ["intrinsic_value", "market_cap", "mos", "quality_score", "ma200"],
            "candidate_min_required_fields": 5,
            "candidate_min_required_ratio": 1.0,
        },
    )
    assert req == ["intrinsic_value", "market_cap", "mos", "quality_score", "ma200"]
    assert min_count == 5
    assert min_ratio == 1.0
    assert bool(out.loc[0, "candidate_data_ok"]) is False
    assert "ma200" in str(out.loc[0, "candidate_data_missing_fields"])


def test_index_filter_marks_stale_index_price() -> None:
    base = pd.DataFrame({"ticker": ["AAA"], "yahoo_ticker": ["AAA.OL"], "info_country": ["Norway"]})
    idx_prices = pd.DataFrame(
        {
            "ticker": ["^OSEAX"] * 260,
            "date": pd.date_range("2025-01-01", periods=260, freq="D"),
            "adj_close": [100.0 + i for i in range(260)],
        }
    )
    out = _apply_index_technical_filter(
        base,
        prices_df=idx_prices,
        asof="2026-03-02",
        dec_cfg={"max_price_age_days": 7, "require_index_ma200": True},
        mad_min=-0.05,
    )
    assert bool(out.loc[0, "index_price_stale"]) is True
    assert bool(out.loc[0, "index_data_ok"]) is False


def test_decision_schema_rows_contains_data_sufficiency_line() -> None:
    pick = pd.Series(
        {
            "mos": 0.5,
            "mos_req": 0.3,
            "mos_basis": "equity_vs_mcap",
            "high_risk_flag": 0,
            "stock_price_age_days": 0,
            "stock_price_stale": False,
            "index_price_age_days": 0,
            "index_price_stale": False,
            "value_creation_ok": True,
            "roic_wacc_spread": 0.04,
            "quality_weak_count": 0,
            "quality_gate_ok": True,
            "stock_ma200_ok": True,
            "index_ma200_ok": True,
            "stock_mad_ok": True,
            "technical_ok": True,
            "candidate_available_fields_count": 5,
            "candidate_required_fields_count": 10,
            "candidate_data_coverage_ratio": 0.5,
            "candidate_data_ok": False,
            "candidate_data_missing_fields": "ma200, index_price",
            "fundamental_ok": False,
            "eligible": False,
            "reason_fundamental_fail": "DATA_NOT_SUFFICIENT_FOR_EVALUATION",
            "reason_technical_fail": "",
        }
    )
    rows = _decision_schema_rows(
        pick=pick,
        mos_min=0.3,
        mos_high=0.4,
        mad_min=-0.05,
        max_price_age_days=7,
        required_fields=["intrinsic_value", "market_cap"],
        min_count=8,
        min_ratio=0.75,
    )
    hit = rows[rows["parameter"] == "Datatilstrekkelighet"]
    assert len(hit) == 1
    assert hit.iloc[0]["status"] == "FAIL"


def test_media_red_flag_scan_disabled_returns_without_network() -> None:
    out = _run_media_red_flag_scan(
        asof="2026-03-02",
        ticker="AAA",
        company="Acme",
        dec_cfg={"media_red_flags": {"enabled": False}},
    )
    assert out["enabled"] is False
    assert out["status"] == "disabled"
