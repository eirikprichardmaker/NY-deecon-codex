from __future__ import annotations

import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.financials_agent_core import (
    AgentConfig,
    _extract_ixbrl_facts_from_text,
    _parse_esef_file,
    augment_wide_with_derived_fields,
    export_outputs,
    filter_sources_by_mode,
    load_merged_mapping,
    map_to_canonical,
    validate_financials,
)


def _cfg() -> AgentConfig:
    return AgentConfig(
        report_sources_csv=Path("config/report_sources.csv"),
        canonical_schema=Path("config/canonical_schema.yaml"),
        mapping_default=Path("config/mappings/esef_ifrs_default.yaml"),
        mapping_overrides_dir=Path("config/mappings/company_overrides"),
        tolerance_balance=0.02,
        tolerance_cashflow=0.08,
        timeout_sec=10,
        user_agent="test-agent",
    )


def test_schema_loads_with_required_field_keys():
    data = yaml.safe_load(Path("config/canonical_schema.yaml").read_text(encoding="utf-8"))
    fields = data["fields"]
    assert 80 <= len(fields) <= 120
    for f in fields[:10]:
        assert {"field_id", "name", "statement", "sign_convention", "required_core", "sector", "description"}.issubset(f)


def test_mapping_merge_override_correctly(tmp_path: Path):
    override = tmp_path / "sample.yaml"
    override.write_text(
        "version: 1\nmappings:\n  - concept: ifrs-full:Revenue\n    field_id: revenue_services\n    confidence: high\n",
        encoding="utf-8",
    )
    merged = load_merged_mapping(Path("config/mappings/esef_ifrs_default.yaml"), tmp_path, ticker="sample")
    revenue_row = merged[merged["concept"] == "ifrs-full:Revenue"].iloc[0]
    assert revenue_row["field_id"] == "revenue_services"


def test_load_merged_mapping_nan_ticker_uses_default():
    merged = load_merged_mapping(Path("config/mappings/esef_ifrs_default.yaml"), Path("config/mappings/company_overrides"), ticker=float("nan"))
    assert "ifrs-full:Revenue" in set(merged["concept"])


def test_filter_sources_mode_esef_includes_all_esef_variants():
    src = pd.DataFrame(
        [
            {"company_id": "A", "ticker": "A", "year": 2024, "source_type": "esef", "report_url": "x"},
            {"company_id": "B", "ticker": "B", "year": 2024, "source_type": "esef_zip", "report_url": "x"},
            {"company_id": "C", "ticker": "C", "year": 2024, "source_type": "esef_xhtml", "report_url": "x"},
            {"company_id": "D", "ticker": "D", "year": 2024, "source_type": "pdf", "report_url": "x"},
        ]
    )
    out = filter_sources_by_mode(src, "esef")
    assert set(out["source_type"]) == {"esef", "esef_zip", "esef_xhtml"}


def test_validation_rules_on_synthetic_dataset():
    cfg = _cfg()
    long_df = pd.DataFrame(
        [
            {"company_id": "A", "period_end": "2024-12-31", "field_id": "total_assets", "value": 100.0},
            {"company_id": "A", "period_end": "2024-12-31", "field_id": "total_equity", "value": 20.0},
            {"company_id": "A", "period_end": "2024-12-31", "field_id": "total_liabilities", "value": 20.0},
        ]
    )
    wide = pd.DataFrame(
        [
            {
                "company_id": "A",
                "period_end": "2024-12-31",
                "total_assets": 100.0,
                "total_equity": 20.0,
                "total_liabilities": 20.0,
                "revenue_total": -1.0,
                "cash_flow_from_operations": 1.0,
                "cash_flow_from_investing": 1.0,
                "cash_flow_from_financing": 1.0,
                "net_change_in_cash": 10.0,
            }
        ]
    )
    issues = validate_financials(long_df, wide, cfg)
    assert set(issues["issue_type"]).issuperset({"balance_sheet_not_balanced", "cashflow_bridge_mismatch", "sign_sanity_failed"})


def test_export_creates_expected_excel_sheets(tmp_path: Path):
    long_df = pd.DataFrame([{"company_id": "A", "period_end": "2024-12-31", "statement": "IS", "field_id": "revenue_total", "value": 1.0, "unit": "", "currency": "NOK", "source_doc_id": "d1", "confidence": "high", "raw_tag": "ifrs-full:Revenue", "raw_label": "Revenue"}])
    wide_df = pd.DataFrame([{"company_id": "A", "period_end": "2024-12-31", "revenue_total": 1.0}])
    issues_df = pd.DataFrame([{"severity": "warning", "issue_type": "x", "company_id": "A", "period_end": "2024-12-31", "field_id": "revenue_total", "details": "d"}])

    export_outputs("2026-01-01", tmp_path / "processed", tmp_path / "exports", long_df, wide_df, issues_df)
    xlsx = tmp_path / "exports" / "financials.xlsx"
    wb = pd.ExcelFile(xlsx)
    assert set(wb.sheet_names) == {"financials_long", "financials_wide", "issues"}


def test_parse_esef_mock_zip_and_map_to_canonical(tmp_path: Path):
    xhtml = Path("tests/fixtures/sample_esef.xhtml")
    zpath = tmp_path / "sample.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(xhtml, arcname="report.xhtml")

    facts = _parse_esef_file(zpath)
    assert any(f["concept"] == "ifrs-full:Revenue" for f in facts)

    facts_df = pd.DataFrame(
        [
            {
                "source_doc_id": "doc1",
                "company_id": "SAMPLE_CO",
                "ticker": "SAMPLE",
                "period_end": "2024-12-31",
                "concept": f["concept"],
                "value": f["value"],
                "decimals": f["decimals"],
                "unit": f["unit"],
                "currency": "NOK",
                "context_ref": f["context_ref"],
                "entity": "SAMPLE_CO",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": f["raw_label"],
                "source_type": "esef",
            }
            for f in facts
        ]
    )
    long_df, mapping_issues = map_to_canonical(facts_df, _cfg())
    assert mapping_issues == []
    assert "revenue_total" in set(long_df["field_id"])


def test_extract_ixbrl_uses_context_period_end():
    txt = """
    <xbrli:context id="c-1">
      <xbrli:entity><xbrli:identifier scheme="x">LEI1</xbrli:identifier></xbrli:entity>
      <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
    </xbrli:context>
    <xbrli:context id="c-2">
      <xbrli:entity><xbrli:identifier scheme="x">LEI1</xbrli:identifier></xbrli:entity>
      <xbrli:period><xbrli:instant>2023-12-31</xbrli:instant></xbrli:period>
      <xbrli:segment><xbrldi:explicitMember dimension="d:Axis">d:Member</xbrldi:explicitMember></xbrli:segment>
    </xbrli:context>
    <ix:nonFraction name="ifrs-full:Revenue" contextRef="c-1" unitRef="u">100</ix:nonFraction>
    <ix:nonFraction name="ifrs-full:Assets" contextRef="c-2" unitRef="u">200</ix:nonFraction>
    """
    facts = _extract_ixbrl_facts_from_text(txt)
    assert len(facts) == 2
    r = next(x for x in facts if x["concept"] == "ifrs-full:Revenue")
    a = next(x for x in facts if x["concept"] == "ifrs-full:Assets")
    assert r["period_end"] == "2024-12-31"
    assert a["period_end"] == "2023-12-31"
    assert a["dimensions"] == "has_dimensions"


def test_extract_ixbrl_resolves_units_and_numeric_sign_scale():
    txt = """
    <xbrli:unit id="u-EUR"><xbrli:measure>iso4217:EUR</xbrli:measure></xbrli:unit>
    <xbrli:context id="c-1">
      <xbrli:entity><xbrli:identifier scheme="x">LEI1</xbrli:identifier></xbrli:entity>
      <xbrli:period><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period>
    </xbrli:context>
    <ix:nonFraction name="ifrs-full:Revenue" contextRef="c-1" unitRef="u-EUR" scale="3">79,928</ix:nonFraction>
    <ix:nonFraction name="ifrs-full:FinanceCosts" contextRef="c-1" unitRef="u-EUR" sign="-">1,234</ix:nonFraction>
    """
    facts = _extract_ixbrl_facts_from_text(txt)
    rev = next(x for x in facts if x["concept"] == "ifrs-full:Revenue")
    fin = next(x for x in facts if x["concept"] == "ifrs-full:FinanceCosts")
    assert rev["unit"] == "iso4217:EUR"
    assert rev["value"] == 79928.0
    assert fin["value"] == -1234.0


def test_augment_wide_derives_required_core_from_raw_facts():
    wide = pd.DataFrame(
        [
            {
                "company_id": "TEL",
                "period_end": "2024-12-31",
                "revenue_total": 79928.0,
                "net_income_attributable_to_parent": 18336.0,
                "net_income": 20109.0,
            }
        ]
    )
    long_df = pd.DataFrame(
        [
            {
                "company_id": "TEL",
                "period_end": "2024-12-31",
                "field_id": "revenue_total",
                "value": 79928.0,
                "currency": "NOK",
                "unit": "iso4217:NOK",
                "period_type": "annual",
            }
        ]
    )
    raw_facts = pd.DataFrame(
        [
            {
                "source_doc_id": "d1",
                "company_id": "TEL",
                "ticker": "TEL",
                "period_end": "2024-12-31",
                "concept": "tel:RawMaterialsAndConsumablesUsedAndTrafficCharges",
                "value": -17731.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c1",
                "entity": "TEL",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "",
                "source_type": "esef_zip",
            },
            {
                "source_doc_id": "d1",
                "company_id": "TEL",
                "ticker": "TEL",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:DepreciationAndAmortisationExpense",
                "value": 16871.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c1",
                "entity": "TEL",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "",
                "source_type": "esef_zip",
            },
            {
                "source_doc_id": "d1",
                "company_id": "TEL",
                "ticker": "TEL",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:EmployeeBenefitsExpense",
                "value": 10005.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c1",
                "entity": "TEL",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "",
                "source_type": "esef_zip",
            },
            {
                "source_doc_id": "d1",
                "company_id": "TEL",
                "ticker": "TEL",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:OtherExpenseByNature",
                "value": 17212.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c1",
                "entity": "TEL",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "",
                "source_type": "esef_zip",
            },
            {
                "source_doc_id": "d1",
                "company_id": "TEL",
                "ticker": "TEL",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:MiscellaneousOtherOperatingExpense",
                "value": 898.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c1",
                "entity": "TEL",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "",
                "source_type": "esef_zip",
            },
            {
                "source_doc_id": "d1",
                "company_id": "TEL",
                "ticker": "TEL",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:BasicEarningsLossPerShare",
                "value": 13.32,
                "decimals": "2",
                "unit": "iso4217:NOK*xbrli:shares",
                "currency": "NOK",
                "context_ref": "c1",
                "entity": "TEL",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "",
                "source_type": "esef_zip",
            },
        ]
    )

    out = augment_wide_with_derived_fields(wide, long_df, raw_facts_df=raw_facts)
    r = out.iloc[0]
    assert r["cost_of_goods_sold"] == -17731.0
    assert r["gross_profit"] == 62197.0
    assert r["sga_expense"] == 28115.0
    assert r["depreciation_expense"] == 16871.0
    assert r["amortization_expense"] == 0.0
    assert abs(r["shares_outstanding_basic"] - (18336.0 / 13.32)) < 1e-9
    assert r["shares_outstanding_diluted"] == r["shares_outstanding_basic"]


def test_map_to_canonical_does_not_sum_duplicate_facts():
    cfg = _cfg()
    facts_df = pd.DataFrame(
        [
            {
                "source_doc_id": "doc1",
                "company_id": "SAMPLE_CO",
                "ticker": "SAMPLE",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:Revenue",
                "value": 100.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c-1",
                "entity": "SAMPLE_CO",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "Revenue",
                "source_type": "esef",
            },
            {
                "source_doc_id": "doc1",
                "company_id": "SAMPLE_CO",
                "ticker": "SAMPLE",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:Revenue",
                "value": 100.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c-1",
                "entity": "SAMPLE_CO",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "Revenue",
                "source_type": "esef",
            },
            {
                "source_doc_id": "doc1",
                "company_id": "SAMPLE_CO",
                "ticker": "SAMPLE",
                "period_end": "2024-12-31",
                "concept": "ifrs-full:Revenue",
                "value": 999.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c-2",
                "entity": "SAMPLE_CO",
                "dimensions": "has_dimensions",
                "taxonomy_ref": "",
                "raw_label": "Revenue",
                "source_type": "esef",
            },
            {
                "source_doc_id": "doc1",
                "company_id": "SAMPLE_CO",
                "ticker": "SAMPLE",
                "period_end": "2023-12-31",
                "concept": "ifrs-full:Revenue",
                "value": 80.0,
                "decimals": "0",
                "unit": "iso4217:NOK",
                "currency": "NOK",
                "context_ref": "c-3",
                "entity": "SAMPLE_CO",
                "dimensions": "",
                "taxonomy_ref": "",
                "raw_label": "Revenue",
                "source_type": "esef",
            },
        ]
    )
    long_df, mapping_issues = map_to_canonical(facts_df, cfg)
    assert mapping_issues == []
    rev = long_df[long_df["field_id"] == "revenue_total"].copy()
    assert len(rev) == 2
    by_period = dict(zip(rev["period_end"], rev["value"]))
    assert by_period["2024-12-31"] == 100.0
    assert by_period["2023-12-31"] == 80.0
