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
    _parse_esef_file,
    export_outputs,
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
