# Financials Agent Runbook

## Prerequisites
- Python 3.11+ (Windows + PowerShell friendly).
- Install dependencies:
  ```powershell
  pip install -r requirements.txt
  ```
- Optional for full ESEF validation: Arelle Python package available in your environment.

## Run command
```powershell
python -m src.run_financials_agent --asof 2026-12-31 --config .\config\agent.yaml --steps download,parse,map,validate,export --source all
```

## Add a new company-year report URL
1. Edit `config/report_sources.csv`.
2. Add row with:
   - `company_id`
   - `ticker`
   - `year`
   - `report_url` (prefer official ESEF ZIP/XHTML)
   - `source_type` (`esef` preferred, `pdf` fallback)
3. Re-run with a new `--asof` date.

## Add mapping overrides
1. Create `config/mappings/company_overrides/<ticker>.yaml` (lowercase ticker recommended).
2. Use same format as default mapping file (`mappings` list with `concept`, `field_id`, `confidence`).
3. Override wins by concept key.

## Interpret issues queue
`data/processed/<asof>/issues.parquet` includes:
- `download_failed`: source could not be fetched.
- `parse_failed`: parse error for a document.
- `pdf_review_required`: PDF fallback queued for review.
- `mapping_gap`: raw concept has no canonical mapping.
- `required_core_missing`: required canonical field absent in period.
- `balance_sheet_not_balanced`, `cashflow_bridge_mismatch`, `sign_sanity_failed`, `duplicate_period_field`.

## Re-run and reproducibility
- Pipeline is date-partitioned by `--asof` and writes immutable artifacts in:
  - `data/raw/<asof>/...`
  - `data/processed/<asof>/...`
  - `exports/<asof>/financials.xlsx`
- Idempotency behavior:
  - Existing raw file with same location is reused unless `--force`.
  - Re-running same inputs yields same normalized outputs.
- Auditability:
  - `metadata.json` per document with source URL and hash.
  - `raw_facts.parquet` + mapped facts + validations + `manifest.json`.
