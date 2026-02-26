from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.financials_agent_core import (
    AgentConfig,
    parse_reports,
    resolve_conflicting_filings,
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


def test_arelle_mode_graceful_when_missing_dependency(monkeypatch):
    monkeypatch.setattr("src.financials_agent_core._arelle_available", lambda: False)
    docs_df = pd.DataFrame(
        [
            {
                "doc_id": "doc1",
                "company_id": "CO1",
                "ticker": "AAA",
                "year": 2024,
                "source_type": "esef",
                "status": "cached",
                "file_path": "tests/fixtures/sample_esef.xhtml",
            }
        ]
    )
    facts, issues, val = parse_reports(docs_df, arelle_mode="parse")
    assert not facts.empty
    assert any(x["issue_type"] == "arelle_parse_failed_fallback" for x in issues)
    assert any("Arelle dependency is not installed" in str(x["details"]) for x in issues)
    assert val.empty


def test_filing_resolution_winner_logic():
    long_df = pd.read_csv("tests/fixtures/conflicting_filings.csv")
    long_df["statement"] = "IS"
    long_df["unit"] = ""
    long_df["confidence"] = "high"
    long_df["raw_tag"] = "ifrs-full:Revenue"
    long_df["raw_label"] = "Revenue"
    long_df["original_value"] = long_df["value"]
    long_df["original_unit"] = ""
    long_df["normalized_value"] = long_df["value"]
    long_df["normalized_unit"] = "currency"

    docs_df = pd.DataFrame(
        [
            {"doc_id": "doc_old", "year": 2024, "source_type": "esef", "filing_datetime": "2025-01-01"},
            {"doc_id": "doc_new", "year": 2024, "source_type": "esef_zip", "filing_datetime": "2025-01-02"},
        ]
    )

    resolution_df, winner_long = resolve_conflicting_filings(long_df, docs_df, _cfg())
    assert resolution_df.iloc[0]["doc_id_winner"] == "doc_new"
    assert set(winner_long["source_doc_id"]) == {"doc_new"}


def test_filing_resolution_handles_missing_docs_schema():
    long_df = pd.DataFrame(
        [
            {
                "company_id": "CO1",
                "period_end": "2024-12-31",
                "period_type": "annual",
                "source_doc_id": "doc_a",
                "field_id": "revenue_total",
                "value": 100.0,
            },
            {
                "company_id": "CO1",
                "period_end": "2024-12-31",
                "period_type": "annual",
                "source_doc_id": "doc_b",
                "field_id": "revenue_total",
                "value": 101.0,
            },
        ]
    )
    docs_df = pd.DataFrame()
    resolution_df, winner_long = resolve_conflicting_filings(long_df, docs_df, _cfg())
    assert len(resolution_df) == 1
    assert len(winner_long) == 1
    assert winner_long.iloc[0]["source_doc_id"] in {"doc_a", "doc_b"}


def test_currency_unit_checks_raise_issues():
    long_df = pd.DataFrame(
        [
            {
                "company_id": "CO1",
                "period_end": "2024-12-31",
                "statement": "IS",
                "field_id": "revenue_total",
                "value": 10.0,
                "unit": "iso4217:EUR",
                "currency": "EUR",
                "source_doc_id": "doc1",
                "confidence": "high",
                "raw_tag": "ifrs-full:Revenue",
                "raw_label": "Revenue",
                "period_type": "annual",
                "original_value": 10.0,
                "original_unit": "iso4217:EUR",
                "normalized_value": 10.0,
                "normalized_unit": "currency",
            },
            {
                "company_id": "CO1",
                "period_end": "2024-12-31",
                "statement": "IS",
                "field_id": "revenue_total",
                "value": 11.0,
                "unit": "iso4217:USD",
                "currency": "",
                "source_doc_id": "doc1",
                "confidence": "high",
                "raw_tag": "ifrs-full:Revenue",
                "raw_label": "Revenue",
                "period_type": "annual",
                "original_value": 11.0,
                "original_unit": "iso4217:USD",
                "normalized_value": 11.0,
                "normalized_unit": "currency",
            },
        ]
    )
    wide_df = pd.DataFrame([{"company_id": "CO1", "period_end": "2024-12-31", "revenue_total": 21.0}])
    issues = validate_financials(long_df, wide_df, _cfg())
    assert {"mixed_currencies", "missing_currency"}.issubset(set(issues["issue_type"]))
