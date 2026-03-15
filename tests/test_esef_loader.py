"""
tests/test_esef_loader.py

Tests for src/esef_loader.py.

Uses tests/fixtures/mock_arelle_extracted_facts.json as test data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.esef_loader import load_facts, _load_mapping, _find_facts_file

FIXTURE = ROOT / "tests" / "fixtures" / "mock_arelle_extracted_facts.json"
MAPPING = ROOT / "config" / "mappings" / "esef_ifrs_default.yaml"


# ---------------------------------------------------------------------------
# _load_mapping
# ---------------------------------------------------------------------------


def test_load_mapping_returns_concept_dict():
    """Mapping file loads and contains expected IFRS concept."""
    mapping = _load_mapping(MAPPING)
    assert isinstance(mapping, dict)
    # The mapping must contain Revenue -> revenue_total
    assert "ifrs-full:Revenue" in mapping
    assert mapping["ifrs-full:Revenue"]["field_id"] == "revenue_total"
    assert mapping["ifrs-full:Revenue"]["confidence"] == "high"


def test_load_mapping_missing_file_returns_empty(tmp_path):
    """Missing mapping file returns empty dict, does not raise."""
    result = _load_mapping(tmp_path / "nonexistent.yaml")
    assert result == {}


# ---------------------------------------------------------------------------
# _find_facts_file
# ---------------------------------------------------------------------------


def test_find_facts_file_ticker_subdir(tmp_path):
    """Finds latest file <= asof in <facts_dir>/<ticker>/ layout."""
    ticker_dir = tmp_path / "EQNR"
    ticker_dir.mkdir()
    (ticker_dir / "2024-12-31.json").write_text("{}")
    (ticker_dir / "2023-06-30.json").write_text("{}")

    found = _find_facts_file("EQNR", "2024-01-01", tmp_path)
    assert found is not None
    assert found.name == "2023-06-30.json"


def test_find_facts_file_flat_layout(tmp_path):
    """Falls back to flat <facts_dir>/<ticker>.json layout."""
    flat = tmp_path / "EQNR.json"
    flat.write_text("{}")
    found = _find_facts_file("EQNR", "2024-12-31", tmp_path)
    assert found == flat


def test_find_facts_file_missing_returns_none(tmp_path):
    """Returns None when no file found — not an error."""
    found = _find_facts_file("MISSING", "2024-01-01", tmp_path)
    assert found is None


# ---------------------------------------------------------------------------
# load_facts — positive cases
# ---------------------------------------------------------------------------


def test_load_facts_returns_revenue_from_fixture(tmp_path):
    """
    POSITIVE: load_facts maps ifrs-full:Revenue to revenue_total with trust=high.
    Uses the mock_arelle_extracted_facts.json fixture.
    """
    assert FIXTURE.exists(), f"Fixture missing: {FIXTURE}"
    ticker_dir = tmp_path / "SAMPLE_CO"
    ticker_dir.mkdir()
    (ticker_dir / "2024-12-31.json").write_bytes(FIXTURE.read_bytes())

    result = load_facts("SAMPLE_CO", "2025-01-01", facts_dir=tmp_path, mapping_path=MAPPING)

    assert result is not None
    assert "revenue_total" in result
    assert result["revenue_total"]["value"] == 1234.0
    assert result["revenue_total"]["source"] == "esef_quarterly"
    assert result["revenue_total"]["trust"] == "high"


def test_load_facts_respects_asof_cutoff(tmp_path):
    """
    POSITIVE: file with period_end > asof is skipped.
    A file dated 2025-03-31 should not be returned for asof=2024-12-31.
    """
    ticker_dir = tmp_path / "SAMPLE_CO"
    ticker_dir.mkdir()
    # Modify fixture: set period_end to future date
    fixture_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture_data["facts"][0]["period_end"] = "2025-03-31"
    (ticker_dir / "2024-12-31.json").write_text(
        json.dumps(fixture_data), encoding="utf-8"
    )

    result = load_facts("SAMPLE_CO", "2024-12-31", facts_dir=tmp_path, mapping_path=MAPPING)

    # The only fact has period_end > asof, so result should be None or not contain revenue_total
    assert result is None or "revenue_total" not in result


# ---------------------------------------------------------------------------
# load_facts — negative / fallback cases
# ---------------------------------------------------------------------------


def test_load_facts_returns_none_when_no_file(tmp_path):
    """
    NEGATIVE: no facts file → returns None (not an exception).
    """
    result = load_facts("NONEXISTENT", "2024-12-31", facts_dir=tmp_path, mapping_path=MAPPING)
    assert result is None


def test_load_facts_returns_none_on_empty_facts(tmp_path):
    """
    NEGATIVE: facts file exists but facts list is empty → returns None.
    """
    ticker_dir = tmp_path / "EMPTY_CO"
    ticker_dir.mkdir()
    (ticker_dir / "2024-12-31.json").write_text(
        json.dumps({"facts": [], "validation": []}), encoding="utf-8"
    )
    result = load_facts("EMPTY_CO", "2025-01-01", facts_dir=tmp_path, mapping_path=MAPPING)
    assert result is None


def test_load_facts_returns_none_on_corrupt_json(tmp_path):
    """
    NEGATIVE: corrupt JSON file → returns None (not an exception).
    """
    ticker_dir = tmp_path / "BAD_CO"
    ticker_dir.mkdir()
    (ticker_dir / "2024-12-31.json").write_text("NOT VALID JSON", encoding="utf-8")
    result = load_facts("BAD_CO", "2025-01-01", facts_dir=tmp_path, mapping_path=MAPPING)
    assert result is None


def test_load_facts_unknown_concepts_are_ignored(tmp_path):
    """
    NEGATIVE: facts with non-IFRS concepts (e.g. company extensions) are silently skipped.
    """
    ticker_dir = tmp_path / "EXT_CO"
    ticker_dir.mkdir()
    data = {
        "facts": [
            {
                "concept": "acme:SomeProprietaryMetric",
                "value": 999.0,
                "period_end": "2024-12-31",
                "entity": "EXT_CO",
                "currency": "EUR",
                "context_ref": "c1",
                "dimensions": "{}",
                "decimals": "0",
                "unit": "iso4217:EUR",
                "raw_label": "Proprietary",
            }
        ]
    }
    (ticker_dir / "2024-12-31.json").write_text(json.dumps(data), encoding="utf-8")
    result = load_facts("EXT_CO", "2025-01-01", facts_dir=tmp_path, mapping_path=MAPPING)
    assert result is None  # no DEECON fields mapped
