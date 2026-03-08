from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import (
    _build_decision_report_html,
    _build_candidate_product_demand_forecast,
    _build_candidate_market_position,
    _estimate_market_forecast_from_prices,
    _extract_candidate_products,
    _media_assessment_from_scan,
    _media_headlines_frame,
    _render_candidate_product_demand_md,
)


def _universe_stub() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "ticker": f"T{i}",
                "company": f"C{i}",
                "profit_growth_5y": 0.02 + 0.005 * i,
                "fcf_yield": 0.01 + 0.002 * i,
                "t2_ebit_margin": 0.08 + 0.004 * i,
                "t2_gross_margin": 0.25 + 0.006 * i,
                "roic_wacc_spread": 0.01 + 0.003 * i,
                "quality_score": -0.5 + 0.1 * i,
            }
        )
    return pd.DataFrame(rows)


def test_extract_candidate_products_prefers_structured_fields():
    pick = pd.Series(
        {
            "ticker": "AAA",
            "company": "Acme",
            "revenue_goods": 60.0,
            "revenue_services": 40.0,
        }
    )
    products = _extract_candidate_products(pick)

    assert len(products) == 2
    assert set(products["product_id"].tolist()) == {"PRODUCT_GOODS", "PRODUCT_SERVICES"}
    assert abs(float(products["value_share"].sum()) - 1.0) < 1e-9


def test_extract_candidate_products_fallback_when_no_structured_fields():
    pick = pd.Series({"ticker": "AAA", "company": "Acme"})
    products = _extract_candidate_products(pick)

    assert len(products) == 1
    assert str(products.iloc[0]["product_id"]) == "PRODUCT_UNSPECIFIED"
    assert "fallback" in str(products.iloc[0]["detail"])


def test_build_candidate_product_demand_forecast_outputs_scenarios_and_md():
    universe = _universe_stub()
    pick = universe.iloc[-1].copy()
    pick["ticker"] = "AAA"
    pick["company"] = "Acme"
    pick["revenue_goods"] = 70.0
    pick["revenue_services"] = 30.0

    products = _extract_candidate_products(pick)
    forecast, summary = _build_candidate_product_demand_forecast(
        pick=pick,
        universe_df=universe,
        asof="2026-02-16",
        products=products,
        horizon_years=3,
    )

    assert not forecast.empty
    assert set(forecast["scenario"].unique().tolist()) == {"bear", "base", "bull"}
    assert set(forecast["year_offset"].unique().tolist()) == {1, 2, 3}
    assert {"base_demand_cagr", "model_confidence", "driver_metrics_used"}.issubset(set(summary.columns))

    md_lines = _render_candidate_product_demand_md(products, forecast, summary)
    md_txt = "\n".join(md_lines)
    assert "Products and Demand Outlook" in md_txt
    assert "Base Scenario (Demand Index)" in md_txt


def test_estimate_market_forecast_from_prices_uses_index_history():
    dates = pd.date_range("2024-01-01", periods=320, freq="B")
    idx_price = 100.0
    rows = []
    for i, d in enumerate(dates):
        idx_price *= (1.0005 if i % 2 == 0 else 1.0002)
        rows.append({"ticker": "^OSEAX", "date": d, "adj_close": idx_price})
    prices_df = pd.DataFrame(rows)
    pick = pd.Series({"relevant_index_symbol": "^OSEAX"})

    out = _estimate_market_forecast_from_prices(prices_df=prices_df, pick=pick, asof="2026-02-16")

    assert out.get("source") == "market_data"
    assert out.get("index_symbol") == "^OSEAX"
    assert float(out.get("base_cagr", 0.0)) > -0.25


def test_render_candidate_product_demand_md_includes_market_position():
    products = pd.DataFrame(
        [{"product_label": "Goods sales", "source_field": "revenue_goods", "value": 70.0, "value_share": 0.7, "detail": "structured_field"}]
    )
    forecast = pd.DataFrame(
        [{"scenario": "base", "product_label": "Goods sales", "forecast_year": 2027, "demand_index": 108.0, "product_demand_index": 75.6, "demand_cagr_assumption": 0.08}]
    )
    summary = pd.DataFrame(
        [{"base_demand_cagr": 0.08, "bear_demand_cagr": 0.03, "bull_demand_cagr": 0.13, "model_confidence": "HIGH", "forecast_source": "market_data"}]
    )
    position = pd.DataFrame(
        [{"market_position_metric": "Market cap percentile (univers)", "value": "82.0%", "comment": "Hoyere er bedre"}]
    )

    md_lines = _render_candidate_product_demand_md(products, forecast, summary, market_position=position)
    md_txt = "\n".join(md_lines)
    assert "Markedsposisjon" in md_txt
    assert "Forecast source: market_data" in md_txt


def test_build_candidate_market_position_has_percentile_rows():
    universe = _universe_stub()
    universe["market_cap"] = [1e9 + i * 1e8 for i in range(len(universe))]
    universe["mos"] = [0.1 + i * 0.01 for i in range(len(universe))]
    universe["sector"] = "Tech"
    universe["country"] = "Norway"
    pick = universe.iloc[-1].copy()
    pick["ticker"] = "AAA"
    pick["company"] = "Acme"

    out = _build_candidate_market_position(pick=pick, universe_df=universe)
    assert not out.empty
    assert "market_position_metric" in out.columns
    assert out["market_position_metric"].astype(str).str.contains("percentile", case=False).any()


def test_media_assessment_from_scan_levels():
    low = _media_assessment_from_scan({"status": "ok", "red_flag_count": 0})
    mod = _media_assessment_from_scan({"status": "ok", "red_flag_count": 1})
    high = _media_assessment_from_scan({"status": "ok", "red_flag_count": 3})
    err = _media_assessment_from_scan({"status": "error", "red_flag_count": 0})

    assert low[0] == "Lav risiko"
    assert mod[0] == "Moderat risiko"
    assert high[0] == "Hoy risiko"
    assert err[0] == "Ukjent"


def test_media_headlines_frame_merges_red_flags_and_sample():
    media_scan = {
        "red_flags": [
            {
                "title": "Company gets warning",
                "link": "https://x/a",
                "source": "X",
                "pub_date": "Mon, 01 Jan 2026 10:00:00 GMT",
                "matched_terms": "warning",
            }
        ],
        "headlines_sample": [
            {
                "title": "Company gets warning",
                "link": "https://x/a",
                "source": "X",
                "pub_date": "Mon, 01 Jan 2026 10:00:00 GMT",
            },
            {
                "title": "Company launches product",
                "link": "https://x/b",
                "source": "Y",
                "pub_date": "Tue, 02 Jan 2026 10:00:00 GMT",
            },
        ],
    }
    out = _media_headlines_frame(media_scan, max_rows=10)
    assert len(out) == 2
    assert "RED_FLAG" in set(out["classification"].astype(str))
    assert "INFO" in set(out["classification"].astype(str))


def test_decision_report_html_value_qa_uses_unresolved_alert_count():
    pick = pd.Series(
        {
            "ticker": "AAA",
            "company": "Acme",
            "mos": 0.40,
            "mos_req": 0.30,
            "technical_ok": True,
            "dq_blocked": False,
            "data_quality_fail": False,
            "value_qc_unresolved_alert_count": 2,
        }
    )
    fundamental_df = pd.DataFrame(
        [
            {"Nokkel": "Value QA unresolved", "Verdi": "2", "Kommentar": "Uforklarte avvik"},
            {"Nokkel": "MoS", "Verdi": "40.0%", "Kommentar": "Krav 30%"},
        ]
    )
    stock_df = pd.DataFrame([{"Nokkel": "Teknisk gate", "Verdi": "True", "Kommentar": "ok"}])
    products_df = pd.DataFrame([{"product_label": "P1"}])
    demand_df = pd.DataFrame([{"forecast_year": 2027, "demand_index": 110.0}, {"forecast_year": 2028, "demand_index": 120.0}])
    media_df = pd.DataFrame([{"headline": "h1", "classification": "INFO", "assessment": "Ingen red-flag term matchet"}])

    out = _build_decision_report_html(
        asof="2026-03-03",
        pick=pick,
        fundamental_df=fundamental_df,
        stock_df=stock_df,
        products_df=products_df,
        demand_df=demand_df,
        demand_chart_lines=[],
        media_df=media_df,
        media_notes=[],
    )
    assert "Value QA" in out
    assert "stat-value'>2" in out
    assert "stat tone-bad" in out
