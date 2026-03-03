from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (
    _apply_index_technical_filter,
    _analyze_candidate_value_qc,
    _build_value_qc_flags,
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


def test_value_qc_flags_detects_extreme_roic_outlier() -> None:
    df = pd.DataFrame(
        {
            "market_cap": [1e9, 2e9, 3e9, 4e9],
            "intrinsic_value": [2e9, 2.1e9, 3.1e9, 4.2e9],
            "mos": [1.0, 0.05, 0.03, 0.05],
            "roic_dec": [0.2, 0.15, 0.1, 34.0],
            "roic": [20.0, 15.0, 10.0, 3400.0],
            "roic_current": [20.0, 15.0, 10.0, 3400.0],
            "wacc_dec": [0.09, 0.09, 0.09, 0.09],
            "roic_wacc_spread": [0.11, 0.06, 0.01, 33.91],
            "quality_score": [0.1, 0.2, 0.3, 0.4],
            "ev_ebit": [10.0, 12.0, 9.0, 11.0],
            "nd_ebitda": [1.0, 2.0, 1.5, 1.2],
            "fcf_yield": [0.05, 0.06, 0.04, 0.07],
            "adj_close": [10.0, 11.0, 9.0, 10.5],
            "ma21": [10.1, 11.1, 9.1, 10.4],
            "ma200": [9.5, 10.5, 8.8, 10.1],
            "mad": [0.05, 0.057, 0.034, 0.03],
            "index_price": [1000.0, 1001.0, 999.0, 1002.0],
            "index_ma21": [990.0, 991.0, 989.0, 992.0],
            "index_ma200": [950.0, 951.0, 949.0, 952.0],
            "index_mad": [0.04, 0.042, 0.043, 0.042],
        }
    )
    qc, specs, summary = _build_value_qc_flags(
        df,
        {"value_qc": {"enabled": True, "min_samples_for_distribution": 2}},
    )
    assert "value_qc_roic_flag" in qc.columns
    assert bool(qc.loc[3, "value_qc_roic_flag"]) is True
    assert int(qc.loc[3, "value_qc_alert_count"]) >= 1
    assert "roic" in str(qc.loc[3, "value_qc_alert_metrics"])
    assert "roic" in summary["metric"].tolist()

    pick = pd.concat([df.loc[3], qc.loc[3]])
    details = _analyze_candidate_value_qc(pick, specs)
    roic_row = details[details["metric"] == "roic"].iloc[0]
    assert bool(roic_row["is_alert"]) is True
    assert bool(roic_row["resolved"]) is False
    assert "Uvanlig verdi" in str(roic_row["note"])
