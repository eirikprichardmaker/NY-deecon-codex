from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_master import (
    _apply_reports_y_anomaly_guard,
    _build_financials_feature_snapshot,
    _build_reports_y_feature_snapshot,
    _build_dividend_feature_snapshot,
    _derive_tier23_financial_metrics,
    _derive_dividend_features,
    _load_financials_tier_fields,
    _latest_raw_snapshot_dir,
    _latest_raw_snapshot_dir_with_data,
    _previous_raw_snapshot_dir_with_data,
)


def test_latest_raw_snapshot_dir_respects_asof(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "2026-02-16").mkdir(parents=True)
    (raw / "2026-02-23").mkdir(parents=True)
    (raw / "2026-02-25").mkdir(parents=True)

    got = _latest_raw_snapshot_dir(raw, "2026-02-24")
    assert got is not None
    assert got.name == "2026-02-23"

    none_case = _latest_raw_snapshot_dir(raw, "2026-02-10")
    assert none_case is None


def test_latest_raw_snapshot_dir_with_data_filters_on_subdirs(tmp_path: Path):
    raw = tmp_path / "raw"
    d1 = raw / "2026-02-16"
    d2 = raw / "2026-02-23"
    d3 = raw / "2026-02-25"
    (d1 / "reports_y").mkdir(parents=True)
    d2.mkdir(parents=True)
    (d3 / "reports_r12").mkdir(parents=True)

    got_y = _latest_raw_snapshot_dir_with_data(raw, "2026-02-26", ["reports_y"])
    assert got_y is not None
    assert got_y.name == "2026-02-16"

    got_any = _latest_raw_snapshot_dir_with_data(raw, "2026-02-26", ["reports_y", "reports_r12"])
    assert got_any is not None
    assert got_any.name == "2026-02-25"


def test_previous_raw_snapshot_dir_with_data_filters_on_subdirs(tmp_path: Path):
    raw = tmp_path / "raw"
    d1 = raw / "2026-02-16"
    d2 = raw / "2026-02-23"
    d3 = raw / "2026-02-25"
    (d1 / "reports_y").mkdir(parents=True)
    (d2 / "reports_y").mkdir(parents=True)
    (d3 / "reports_y").mkdir(parents=True)

    got_prev = _previous_raw_snapshot_dir_with_data(raw, "2026-02-25", ["reports_y"])
    assert got_prev is not None
    assert got_prev.name == "2026-02-23"

    no_prev = _previous_raw_snapshot_dir_with_data(raw, "2026-02-16", ["reports_y"])
    assert no_prev is None


def test_derive_dividend_features_from_reports():
    reports_y = pd.DataFrame(
        {
            "year": [2019, 2020, 2021, 2022, 2023, 2024],
            "period": [5, 5, 5, 5, 5, 5],
            "dividend": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "profit_To_Equity_Holders": [50.0, 55.0, 60.0, 70.0, 80.0, 95.0],
            "earnings_Per_Share": [2.0, 2.4, 2.8, 3.2, 3.6, 4.0],
            "number_Of_Shares": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "stock_Price_Average": [50.0, 52.0, 53.0, 54.0, 55.0, 50.0],
            "report_Date": [
                "2020-02-01",
                "2021-02-01",
                "2022-02-01",
                "2023-02-01",
                "2024-02-01",
                "2025-02-01",
            ],
        }
    )

    reports_r12 = pd.DataFrame(
        {
            "year": [2025],
            "period": [4],
            "revenues": [1000.0],
            "profit_To_Equity_Holders": [100.0],
            "cash_Flow_From_Operating_Activities": [120.0],
            "free_Cash_Flow": [80.0],
            "total_Assets": [2000.0],
            "report_Date": ["2026-02-01"],
        }
    )

    out = _derive_dividend_features(reports_y=reports_y, reports_r12=reports_r12, asof_dt=pd.Timestamp("2026-02-24"))

    assert int(out["dividend_history_years"]) == 6
    assert int(out["dividend_paid_years"]) == 6
    assert abs(float(out["payout_ratio"]) - 0.5) < 1e-12
    assert abs(float(out["dividend_yield"]) - 0.04) < 1e-12
    assert abs(float(out["fcf_margin"]) - 0.08) < 1e-12
    assert abs(float(out["ocf_margin"]) - 0.12) < 1e-12
    assert abs(float(out["profit_margin"]) - 0.10) < 1e-12
    assert abs(float(out["roa"]) - 0.05) < 1e-12
    assert np.isfinite(float(out["dividend_growth_5y"]))
    assert np.isfinite(float(out["profit_growth_5y"]))
    assert np.isfinite(float(out["share_count_growth_5y"]))


def test_build_dividend_feature_snapshot_reads_raw_reports(tmp_path: Path):
    raw = tmp_path / "raw" / "2026-02-23"
    y_dir = raw / "reports_y" / "market=NO"
    r12_dir = raw / "reports_r12" / "market=NO"
    y_dir.mkdir(parents=True)
    r12_dir.mkdir(parents=True)

    reports_y = pd.DataFrame(
        {
            "year": [2023, 2024],
            "period": [5, 5],
            "dividend": [1.5, 2.0],
            "earnings_Per_Share": [3.0, 4.0],
            "number_Of_Shares": [100.0, 101.0],
            "stock_Price_Average": [45.0, 50.0],
            "report_Date": ["2024-02-01", "2025-02-01"],
        }
    )
    reports_r12 = pd.DataFrame(
        {
            "year": [2025],
            "period": [4],
            "revenues": [1000.0],
            "profit_To_Equity_Holders": [80.0],
            "cash_Flow_From_Operating_Activities": [90.0],
            "free_Cash_Flow": [70.0],
            "total_Assets": [1600.0],
            "report_Date": ["2026-01-31"],
        }
    )

    reports_y.to_parquet(y_dir / "ins_id=1.parquet", index=False)
    reports_r12.to_parquet(r12_dir / "ins_id=1.parquet", index=False)

    master = pd.DataFrame({"ticker": ["AAA"], "yahoo_ticker": ["AAA.OL"]})
    ins_map = pd.DataFrame(
        {
            "yahoo_ticker": ["AAA.OL"],
            "yahoo_key": ["AAA.OL"],
            "ticker_norm": ["AAA"],
            "ins_id": [1],
            "market": ["NO"],
        }
    )

    snap = _build_dividend_feature_snapshot(
        master=master,
        raw_snapshot_dir=raw,
        asof_dt=pd.Timestamp("2026-02-24"),
        ins_map=ins_map,
    )

    assert len(snap) == 1
    row = snap.iloc[0]
    assert str(row["yahoo_ticker"]) == "AAA.OL"
    assert np.isfinite(float(row["payout_ratio"]))
    assert np.isfinite(float(row["dividend_yield"]))
    assert np.isfinite(float(row["profit_margin"]))


def test_load_financials_tier_fields_from_yaml(tmp_path: Path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    tiers = cfg_dir / "tiers.yaml"
    tiers.write_text(
        "version: 1\ncore_model_input_fields:\n  - revenue_total\nenrichment_optional_fields:\n  - ebitda\n",
        encoding="utf-8",
    )
    cfg = {"financials_enrichment": {"tiers_config": "config/tiers.yaml"}}
    core, enrich = _load_financials_tier_fields(tmp_path, cfg)
    assert core == ["revenue_total"]
    assert enrich == ["ebitda"]


def test_build_financials_feature_snapshot_maps_company_to_yahoo(tmp_path: Path):
    raw_root = tmp_path / "data" / "raw"
    processed_root = tmp_path / "data" / "processed"
    run_date = "2026-03-18"
    (raw_root / run_date).mkdir(parents=True)
    (processed_root / run_date).mkdir(parents=True)

    pd.DataFrame(
        [
            {"company_id": "TELENOR", "ticker": "TEL.OL", "year": 2024},
            {"company_id": "TELENOR", "ticker": "TEL.OL", "year": 2023},
        ]
    ).to_parquet(raw_root / run_date / "download_index.parquet", index=False)

    pd.DataFrame(
        [
            {"company_id": "TELENOR", "period_end": "2023-12-31", "revenue_total": 80452.0},
            {"company_id": "TELENOR", "period_end": "2024-12-31", "revenue_total": 79928.0},
        ]
    ).to_parquet(processed_root / run_date / "financials_wide.parquet", index=False)

    ticker_map = pd.DataFrame([{"ticker_norm": "TEL", "yahoo_ticker": "TEL.OL"}])
    out = _build_financials_feature_snapshot(
        project_root=tmp_path,
        raw_root=raw_root,
        processed_root=processed_root,
        asof="2026-03-18",
        ticker_map=ticker_map,
        fields=["revenue_total"],
    )
    assert len(out) == 1
    assert out.iloc[0]["yahoo_ticker"] == "TEL.OL"
    assert float(out.iloc[0]["revenue_total"]) == 79928.0
    assert out.iloc[0]["financials_source"] == "financials_agent"


def test_build_reports_y_feature_snapshot_maps_to_canonical(tmp_path: Path):
    raw = tmp_path / "data" / "raw" / "2026-03-18"
    (raw / "reports_y" / "market=NO").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "year": 2024,
                "period": 5,
                "revenues": 79928.0,
                "gross_Income": 62197.0,
                "operating_Income": 18623.0,
                "profit_Before_Tax": 24866.0,
                "profit_To_Equity_Holders": 19107.0,
                "number_Of_Shares": 1398.403,
                "cash_And_Equivalents": 10380.0,
                "total_Assets": 229580.0,
                "total_Equity": 82544.0,
                "non_Current_Liabilities": 100652.0,
                "current_Liabilities": 46384.0,
                "total_Liabilities_And_Equity": 229580.0,
                "cash_Flow_From_Operating_Activities": 31481.0,
                "cash_Flow_From_Investing_Activities": -11486.0,
                "cash_Flow_From_Financing_Activities": -29469.0,
                "cash_Flow_For_The_Year": -9474.0,
                "report_End_Date": "2024-12-31",
                "report_Date": "2025-02-06",
                "currency": "NOK",
            }
        ]
    ).to_parquet(raw / "reports_y" / "market=NO" / "ins_id=966.parquet", index=False)

    master = pd.DataFrame([{"ticker": "TEL", "yahoo_ticker": "TEL.OL"}])
    ins_map = pd.DataFrame(
        [{"yahoo_ticker": "TEL.OL", "yahoo_key": "TEL.OL", "ticker_norm": "TEL", "ins_id": 966, "market": "NO"}]
    )
    out = _build_reports_y_feature_snapshot(
        master=master,
        raw_snapshot_dir=raw,
        asof_dt=pd.Timestamp("2026-03-18"),
        ins_map=ins_map,
        fields=[
            "revenue_total",
            "gross_profit",
            "operating_income_ebit",
            "profit_before_tax",
            "net_income_attributable_to_parent",
            "shares_outstanding_basic",
            "shares_outstanding_diluted",
            "total_assets",
            "total_equity",
            "total_liabilities",
            "cash_flow_from_operations",
            "cash_flow_from_investing",
            "cash_flow_from_financing",
            "net_change_in_cash",
            "reporting_currency",
            "fiscal_year",
            "reporting_period_type",
        ],
    )
    assert len(out) == 1
    r = out.iloc[0]
    assert r["financials_source"] == "reports_y"
    assert r["financials_period_end"] == "2024-12-31"
    assert float(r["revenue_total"]) == 79928.0
    assert float(r["gross_profit"]) == 62197.0
    assert float(r["net_income_attributable_to_parent"]) == 19107.0
    assert float(r["shares_outstanding_basic"]) == 1398.403
    assert float(r["shares_outstanding_diluted"]) == 1398.403
    assert float(r["total_liabilities"]) == (229580.0 - 82544.0)
    assert str(r["reporting_currency"]) == "NOK"


def test_derive_tier23_financial_metrics_with_fallbacks():
    master = pd.DataFrame(
        [
            {
                "revenue_total": 100.0,
                "gross_profit": 40.0,
                "operating_income_ebit": 20.0,
                "profit_before_tax": 15.0,
                "income_tax_expense": 3.0,
                "net_income_attributable_to_parent": 10.0,
                "net_income": 11.0,
                "cash_flow_from_operations": 18.0,
                "capex_purchase_ppe": 4.0,
                "capex_purchase_intangibles": np.nan,
                "depreciation_expense": 5.0,
                "amortization_expense": 1.0,
                "finance_costs": -4.0,
                "short_term_debt": 8.0,
                "long_term_debt": 22.0,
                "lease_liabilities_current": 3.0,
                "lease_liabilities_noncurrent": 7.0,
                "cash_and_cash_equivalents": 5.0,
                "short_term_investments": 2.0,
                "accounts_receivable": 12.0,
                "inventory": 9.0,
                "accounts_payable": 6.0,
                "total_assets": 120.0,
                "total_equity": 50.0,
                "total_liabilities": 70.0,
            },
            {
                "revenue_total": 80.0,
                "gross_profit": 20.0,
                "operating_income_ebit": 8.0,
                "profit_before_tax": 6.0,
                "income_tax_expense": 1.2,
                "net_income": 4.0,
                "cash_flow_from_operations": 5.0,
                "interest_paid_cash": 2.0,
                "short_term_debt": 4.0,
                "long_term_debt": 6.0,
                "cash_and_cash_equivalents": 3.0,
                "accounts_receivable": 4.0,
                "inventory": 3.0,
                "accounts_payable": 2.0,
                "total_assets": 40.0,
                "total_equity": 20.0,
                "total_liabilities": 20.0,
            },
        ]
    )
    out, cov = _derive_tier23_financial_metrics(master)
    r1 = out.iloc[0]
    r2 = out.iloc[1]

    assert abs(float(r1["t2_effective_tax_rate"]) - 0.2) < 1e-12
    assert abs(float(r1["t2_ebitda_used"]) - 26.0) < 1e-12
    assert abs(float(r1["t2_gross_margin"]) - 0.4) < 1e-12
    assert abs(float(r1["t2_fcf_simple"]) - 14.0) < 1e-12
    assert abs(float(r1["t2_net_debt_excl_leases"]) - 23.0) < 1e-12
    assert abs(float(r1["t2_net_debt_incl_leases"]) - 33.0) < 1e-12
    assert abs(float(r1["t2_interest_coverage"]) - 5.0) < 1e-12
    assert abs(float(r1["t3_working_capital"]) - 15.0) < 1e-12
    assert abs(float(r1["t3_liabilities_to_equity"]) - 1.4) < 1e-12
    assert str(r1["t2_ebitda_used_derived_from"]).startswith("fallback:")

    assert abs(float(r2["t2_interest_coverage"]) - 4.0) < 1e-12
    assert str(r2["t2_interest_coverage_derived_from"]).startswith("fallback:")
    assert str(r2["t2_net_margin_derived_from"]).startswith("fallback:")

    assert "t2_effective_tax_rate" in cov
    assert cov["t2_effective_tax_rate"] > 0.0


def test_apply_reports_y_anomaly_guard_flags_scale_jump():
    current = pd.DataFrame(
        [
            {
                "yahoo_ticker": "TRMED.OL",
                "financials_period_end": "2024-12-31",
                "revenue_total": 120.0,
                "gross_profit": 60.0,
                "operating_income_ebit": -43.9,
                "profit_before_tax": -42.6,
                "net_income_attributable_to_parent": -42.6,
                "cash_and_cash_equivalents": 123.4,
                "total_assets": 414.4,
                "total_equity": 337.1,
                "total_liabilities": 77.3,
                "cash_flow_from_operations": -64.1,
                "cash_flow_from_investing": 6.3,
                "cash_flow_from_financing": 104.0,
                "net_change_in_cash": 81.6,
            },
            {
                "yahoo_ticker": "TEL.OL",
                "financials_period_end": "2024-12-31",
                "revenue_total": 100.0,
                "gross_profit": 40.0,
                "operating_income_ebit": 20.0,
                "profit_before_tax": 18.0,
                "net_income_attributable_to_parent": 14.0,
                "cash_and_cash_equivalents": 10.0,
                "total_assets": 200.0,
                "total_equity": 100.0,
                "total_liabilities": 100.0,
                "cash_flow_from_operations": 22.0,
                "cash_flow_from_investing": -8.0,
                "cash_flow_from_financing": -6.0,
                "net_change_in_cash": 8.0,
            },
        ]
    )
    previous = pd.DataFrame(
        [
            {
                "yahoo_ticker": "TRMED.OL",
                "financials_period_end": "2024-12-31",
                "revenue_total": 0.12,
                "gross_profit": 0.06,
                "operating_income_ebit": -0.0439,
                "profit_before_tax": -0.0426,
                "net_income_attributable_to_parent": -0.0426,
                "cash_and_cash_equivalents": 0.1234,
                "total_assets": 0.4144,
                "total_equity": 0.3371,
                "total_liabilities": 0.0773,
                "cash_flow_from_operations": -0.0641,
                "cash_flow_from_investing": 0.0063,
                "cash_flow_from_financing": 0.104,
                "net_change_in_cash": 0.0816,
            },
            {
                "yahoo_ticker": "TEL.OL",
                "financials_period_end": "2024-12-31",
                "revenue_total": 100.0,
                "gross_profit": 40.0,
                "operating_income_ebit": 20.0,
                "profit_before_tax": 18.0,
                "net_income_attributable_to_parent": 14.0,
                "cash_and_cash_equivalents": 10.0,
                "total_assets": 200.0,
                "total_equity": 100.0,
                "total_liabilities": 100.0,
                "cash_flow_from_operations": 22.0,
                "cash_flow_from_investing": -8.0,
                "cash_flow_from_financing": -6.0,
                "net_change_in_cash": 8.0,
            },
        ]
    )
    kept, flagged, stats = _apply_reports_y_anomaly_guard(
        current=current,
        previous=previous,
        candidate_fields=[
            "revenue_total",
            "gross_profit",
            "operating_income_ebit",
            "profit_before_tax",
            "net_income_attributable_to_parent",
            "cash_and_cash_equivalents",
            "total_assets",
            "total_equity",
            "total_liabilities",
            "cash_flow_from_operations",
            "cash_flow_from_investing",
            "cash_flow_from_financing",
            "net_change_in_cash",
        ],
        guard_cfg={"ratio_threshold": 100.0, "min_changed_fields": 5, "min_abs_delta": 1.0},
        current_snapshot_name="2026-02-26",
        previous_snapshot_name="2026-02-25",
    )

    assert len(kept) == 1
    assert kept.iloc[0]["yahoo_ticker"] == "TEL.OL"
    assert len(flagged) == 1
    assert flagged.iloc[0]["yahoo_ticker"] == "TRMED.OL"
    assert bool(flagged.iloc[0]["financials_guard_anomaly"]) is True
    assert "reports_y_scale_jump_vs_prev_snapshot:2026-02-25->2026-02-26" in str(
        flagged.iloc[0]["reason_financials_guard"]
    )
    assert int(stats["flagged_rows"]) == 1
    assert int(stats["excluded_rows"]) == 1
