# ESEF/XBRL Quarterly Facts — Status

**Date**: 2026-03-15

---

## Status: Stub → Loader Created (Option A)

`src/financials_agent_core.py` contains full Arelle XBRL extraction logic, but it is an
optional IR-document step and is **not connected to `valuation.py`**.

From the valuation pipeline's perspective, ESEF data was a stub as of 2026-03-15.

---

## What Was Done

### 1. `src/esef_loader.py` — new module

Loads ESEF/XBRL quarterly facts for a ticker and asof date, mapping IFRS concept names to
DEECON field IDs using the `config/mappings/esef_ifrs_default.yaml` mapping file.

**Key function**:
```python
load_facts(ticker, asof, facts_dir=None, mapping_path=None) -> dict | None
```

Returns:
```python
{
    "revenue_total":         {"value": 1234.0, "source": "esef_quarterly", "trust": "high"},
    "operating_income_ebit": {"value": 89.0,   "source": "esef_quarterly", "trust": "high"},
    ...
}
```
or `None` if no file found, JSON is corrupt, facts list is empty, or no IFRS concepts match.

**Supported fields** (`_DEECON_FIELDS`):
- `revenue_total`, `operating_income_ebit`, `free_cash_flow`, `cash_flow_from_operations`
- `capital_expenditures`, `net_debt`, `shares_outstanding_basic`, `total_equity`

**File layout supported**:
- `data/esef_facts/<TICKER>/<YYYY-MM-DD>.json` — per-ticker subdirectory (most recent <= asof)
- `data/esef_facts/<TICKER>.json` — flat layout fallback

**Graceful fallbacks**:
- Missing file → `None` (debug log)
- Corrupt JSON → `None` (warning log)
- Empty facts → `None` (debug log)
- Unknown concepts (company extensions) → silently skipped
- Future-dated facts (period_end > asof) → skipped

### 2. `tests/test_esef_loader.py` — 11 tests, all passing

Covers `_load_mapping`, `_find_facts_file`, `load_facts` (positive + negative paths).
Uses `tests/fixtures/mock_arelle_extracted_facts.json` as test fixture.

### 3. `valuation.py` integration — optional TTM supplement

Gated behind `valuation.use_esef_facts: false` (default off, opt-in).

When enabled:
- Iterates over all tickers in the pipeline universe
- Calls `load_facts()` per ticker using `data/esef_facts/` as the facts directory
- Supplements `fcf_m` (free cash flow in millions) where it is currently NaN
- Uses ESEF FCF only from `trust=high` mappings
- **Never replaces existing FCF data** — pure fill-in
- Logs coverage: count of tickers with ESEF FCF found and count used as fill
- Writes audit to `valuation_input_audit.json["esef_facts"]`

---

## How to Activate

Add to `config/config.yaml`:
```yaml
valuation:
  use_esef_facts: true
```

And populate `data/esef_facts/<TICKER>/` with Arelle-extracted JSON files (format matches
`tests/fixtures/mock_arelle_extracted_facts.json`).

---

## Current Limitation

`data/esef_facts/` does not yet contain real data. The loader returns `None` for all tickers
until quarterly XBRL facts are extracted via `src/financials_agent_core.py` and stored in
the expected directory layout.

---

## Mapping File

`config/mappings/esef_ifrs_default.yaml` maps IFRS-full concept names to DEECON field IDs.
Key mappings (high confidence):
- `ifrs-full:Revenue` → `revenue_total`
- `ifrs-full:OperatingIncome` → `operating_income_ebit`
- `ifrs-full:CashFlowsFromUsedInOperatingActivities` → `cash_flow_from_operations`
- `ifrs-full:PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsAndOtherLongTermAssets` → `capital_expenditures`
