"""
src/esef_loader.py

Loads ESEF/XBRL quarterly facts for a given ticker and asof date,
mapping IFRS concept names to DEECON field IDs using the ESEF default mapping.

Key function:
    load_facts(ticker, asof, facts_dir, mapping_path) -> dict | None

Return format (per field):
    {
        "revenue_total":           {"value": 1234.0, "source": "esef_quarterly", "trust": "high"},
        "operating_income_ebit":   {"value": 89.0,   "source": "esef_quarterly", "trust": "high"},
        ...
    }

If no data file is found, returns None (not an error) and logs a message.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

logger = logging.getLogger(__name__)

# Fields DEECON cares about (field_id from esef_ifrs_default.yaml)
_DEECON_FIELDS = {
    "revenue_total",
    "operating_income_ebit",
    "free_cash_flow",
    "cash_flow_from_operations",
    "capital_expenditures",
    "net_debt",
    "shares_outstanding_basic",
    "total_equity",
}

_DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[1] / "config" / "mappings" / "esef_ifrs_default.yaml"
_DEFAULT_FACTS_DIR = Path(__file__).resolve().parents[1] / "data" / "esef_facts"


def _load_mapping(mapping_path: Path) -> dict[str, dict[str, str]]:
    """Return {concept -> {"field_id": ..., "confidence": ...}}."""
    if not _HAS_YAML:
        logger.warning("esef_loader: pyyaml not installed, using empty mapping.")
        return {}
    if not mapping_path.exists():
        logger.warning("esef_loader: mapping file not found: %s", mapping_path)
        return {}
    with mapping_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    result: dict[str, dict[str, str]] = {}
    for entry in data.get("mappings", []):
        concept = str(entry.get("concept", "")).strip()
        field_id = str(entry.get("field_id", "")).strip()
        confidence = str(entry.get("confidence", "medium")).strip()
        if concept and field_id:
            # Don't overwrite a high-confidence mapping with a medium one
            if concept not in result or confidence == "high":
                result[concept] = {"field_id": field_id, "confidence": confidence}
    return result


def _find_facts_file(ticker: str, asof: str, facts_dir: Path) -> Path | None:
    """
    Find the most recent ESEF facts file for `ticker` that is <= asof.

    Expected layout:
        <facts_dir>/<ticker>/<YYYY-MM-DD>.json   (period_end date)
    or
        <facts_dir>/<YYYY-MM-DD>/<ticker>.json   (snapshot date)

    Returns None if no suitable file is found.
    """
    ticker_dir = facts_dir / ticker
    if ticker_dir.exists():
        candidates = sorted(ticker_dir.glob("*.json"), reverse=True)
        for f in candidates:
            if f.stem <= asof:
                return f
    # Flat layout: <facts_dir>/<ticker>.json
    flat = facts_dir / f"{ticker}.json"
    if flat.exists():
        return flat
    return None


def load_facts(
    ticker: str,
    asof: str,
    facts_dir: Path | None = None,
    mapping_path: Path | None = None,
) -> dict[str, dict[str, Any]] | None:
    """
    Load ESEF quarterly facts for `ticker` as of `asof` date.

    Parameters
    ----------
    ticker : str
        Base ticker (e.g. "EQNR", not "EQNR.OL").
    asof : str
        ISO date string (YYYY-MM-DD). Returns facts for the most recent
        period <= asof.
    facts_dir : Path, optional
        Directory containing ESEF facts JSON files.
        Defaults to data/esef_facts/.
    mapping_path : Path, optional
        Path to IFRS-ESEF concept mapping YAML.
        Defaults to config/mappings/esef_ifrs_default.yaml.

    Returns
    -------
    dict mapping field_id -> {"value": float, "source": "esef_quarterly", "trust": str}
    or None if no data file is found.
    """
    if facts_dir is None:
        facts_dir = _DEFAULT_FACTS_DIR
    if mapping_path is None:
        mapping_path = _DEFAULT_MAPPING_PATH

    facts_file = _find_facts_file(ticker, asof, Path(facts_dir))
    if facts_file is None:
        logger.debug("esef_loader: no facts file for %s asof %s in %s", ticker, asof, facts_dir)
        return None

    try:
        raw = json.loads(facts_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("esef_loader: failed to read %s: %s", facts_file, exc)
        return None

    mapping = _load_mapping(Path(mapping_path))
    facts_list = raw.get("facts", [])
    if not facts_list:
        logger.debug("esef_loader: empty facts list in %s", facts_file)
        return None

    # Collect best value per field_id (prefer period closest to asof, high confidence)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for fact in facts_list:
        concept = str(fact.get("concept", "")).strip()
        if concept not in mapping:
            continue
        field_id = mapping[concept]["field_id"]
        if field_id not in _DEECON_FIELDS:
            continue
        try:
            value = float(fact["value"])
        except (KeyError, TypeError, ValueError):
            continue
        period_end = str(fact.get("period_end", ""))
        if period_end > asof:
            continue  # future data — skip
        candidates.setdefault(field_id, []).append({
            "value": value,
            "period_end": period_end,
            "confidence": mapping[concept]["confidence"],
        })

    if not candidates:
        return None

    result: dict[str, dict[str, Any]] = {}
    for field_id, entries in candidates.items():
        # Sort: prefer high confidence, then most recent period
        def _sort_key(e: dict) -> tuple:
            conf_rank = 0 if e["confidence"] == "high" else 1
            return (conf_rank, e["period_end"])

        best = sorted(entries, key=_sort_key)[0]
        result[field_id] = {
            "value": best["value"],
            "source": "esef_quarterly",
            "trust": best["confidence"],
        }

    logger.info(
        "esef_loader: loaded %d fields for %s asof %s from %s",
        len(result), ticker, asof, facts_file.name,
    )
    return result or None
