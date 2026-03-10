# CLAUDE.md — NY-deecon-codex (DEECON)

This file provides guidance for AI assistants working in this repository.

---

## Project Overview

DEECON is a **deterministic Nordic equity investment analysis pipeline** that selects at most one stock (or CASH) per weekly run. It combines fundamental valuation (DCF) with technical analysis gating.

**Core invariants:**
- Intrinsic value is computed from **fundamentals + risk inputs only** — never from price data.
- Technical analysis (MA200/MAD + index regime) is used for **gating/timing only**, never as valuation input.
- Every exclusion must carry an explicit `reason_*` field. Silent row drops are forbidden.
- The final decision output is **exactly one stock or CASH**.
- All runs are **as-of dated** (`--asof YYYY-MM-DD`) for reproducibility and backtesting.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Data | pandas, numpy, pyarrow |
| Config | PyYAML |
| Testing | pytest 7.4+ |
| Schema validation | pandera |
| GUI | PySide6 (Qt6) primary, Tkinter fallback |
| ESEF/XBRL | Arelle |
| Data sources | Børsdata API, yfinance |
| Excel output | openpyxl |
| Packaging | PyInstaller (Windows .exe) |
| CI/CD | GitHub Actions (Windows, Python 3.12) |

---

## Repository Structure

```
NY-deecon-codex/
├── src/                        # All pipeline source code
│   ├── run_weekly.py           # Main entry point / orchestrator
│   ├── ingest_validate.py      # Step 1: Load & validate Børsdata fundamentals
│   ├── transform_fundamentals.py # Step 2: Enrich fundamentals (R12 rolling bridge)
│   ├── freeze_golden_fundamentals.py # Step 3: Snapshot frozen fundamentals
│   ├── transform_prices.py     # Step 4: MA200/MAD technical indicators
│   ├── build_master.py         # Step 5: Merge fundamentals + prices
│   ├── compute_factors.py      # Step 6: Quality/value/momentum factor scores
│   ├── cost_of_capital.py      # Step 7: WACC (cost of equity + debt)
│   ├── valuation.py            # Step 8: DCF valuation + MoS + sensitivity
│   ├── decision.py             # Step 9: Gating + DQ checks + final pick
│   ├── ir_reports.py           # Optional: Investor relations report processing
│   ├── report_watch.py         # Monitor quarterly report release dates
│   ├── refresh_prices_daily.py # Daily price refresh (idempotent)
│   ├── healthcheck.py          # Pipeline health monitoring
│   ├── wft.py                  # Walk-forward backtesting (54 KB)
│   ├── financials_agent_core.py # ESEF/XBRL parsing and mapping (67 KB)
│   ├── gui.py                  # Tkinter GUI (112 KB)
│   ├── gui_qt.py               # PySide6/Qt6 GUI (82 KB)
│   ├── gui_windows_entry.py    # Windows executable entry point
│   ├── agents/                 # Agent modules
│   ├── common/
│   │   ├── config.py           # Config loading, path resolution, run context
│   │   ├── io.py               # Parquet/CSV I/O helpers
│   │   ├── log.py              # Logging setup
│   │   ├── errors.py           # AppError and custom exceptions
│   │   ├── schema.py           # Schema utilities
│   │   └── utils.py            # General helpers (safe_div, zscore)
│   └── data_quality/
│       ├── rules.py            # 67+ DQ rules (DQ-F001…DQ-W*) with FAIL/WARN/PASS
│       └── schema_contracts.py # Pandera schema enforcement
├── tests/                      # 37 test files
│   └── fixtures/               # Sample data and mock objects
├── config/
│   ├── config.yaml             # Primary config (baseline strategy)
│   ├── config_dividend.yaml    # Dividend quality strategy variant
│   ├── config_graham.yaml      # Graham defensive strategy variant
│   ├── canonical_schema.yaml   # ESEF/XBRL field mapping (107 fields)
│   ├── computed_metrics.yaml   # Derived KPI definitions
│   ├── financials_field_tiers.yaml # Field priority tiers
│   ├── agent.yaml              # Financials agent config
│   └── agent_verified.yaml     # Verified agent config
├── configs/
│   ├── sources.yaml            # Børsdata data source mappings
│   └── thresholds.yaml         # MA200/valuation gate thresholds
├── docs/
│   ├── AGENT_RUNBOOK.md        # Financials agent operations manual
│   ├── CANONICAL_SCHEMA.md     # ESEF/XBRL schema specification
│   ├── KPI_SHORTLIST_AND_TESTPLAN.md
│   └── ablation_baseline_vs_v2.md
├── scripts/                    # PowerShell automation (Windows Task Scheduler)
├── tools/                      # WFT optimization, Børsdata utilities, repair scripts
├── build/                      # PyInstaller config
├── .github/workflows/
│   └── gui-smoke.yml           # CI: GUI smoke test on Windows
├── requirements.txt
├── README.md
├── agents.md                   # Non-negotiable operational rules
├── Project Knowledge Index.md  # How to use project files
└── TUNING_PARAMS.md            # WFT tunable parameters
```

---

## Module Contract

Every pipeline step module **must** expose:

```python
def run(ctx: dict, log: logging.Logger) -> None:
    ...
```

`ctx` is the run context built by `src.common.config.build_run_context`. It carries paths, config values, and the `asof` date. This contract is enforced by `run_weekly.py` at import time.

---

## Pipeline Steps

Run order in `src/run_weekly.py`:

| Step name | Module | Output artifact |
|-----------|--------|----------------|
| `ingest` | `src.ingest_validate` | `master.parquet` |
| `transform_fundamentals` | `src.transform_fundamentals` | Enriched fundamentals with R12 |
| `freeze_golden` | `src.freeze_golden_fundamentals` | `fundamentals_golden.parquet` |
| `transform_prices` | `src.transform_prices` | `prices.parquet` (with MA200/MAD) |
| `master` | `src.build_master` | Merged master dataset |
| `factors` | `src.compute_factors` | Factor scores (quality/value/momentum) |
| `cost_of_capital` | `src.cost_of_capital` | WACC per ticker |
| `valuation` | `src.valuation` | `valuation.csv`, `valuation_sensitivity.csv`, audit JSON |
| `decision` | `src.decision` | `decision.csv`, `decision.md` |

Optional: `ir_reports` → `src.ir_reports`

**Data flow:**
```
Ingest (Børsdata) → Transform Fundamentals (R12) → Freeze Golden
  → Transform Prices (MA200/MAD) → Build Master → Compute Factors
  → Cost of Capital (WACC) → Valuation (DCF + MoS) → Decision (1 stock or CASH)
  → Exports (CSV, JSON, audit trails in runs/<run_id>/)
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run all steps
python -m src.run_weekly --asof 2026-03-10 --config ./config/config.yaml

# Run specific steps only
python -m src.run_weekly --asof 2026-03-10 --config ./config/config.yaml --steps valuation,decision

# Dry run (init run-dir + manifest + log only)
python -m src.run_weekly --asof 2026-03-10 --config ./config/config.yaml --dry-run

# List available steps
python -m src.run_weekly --list-steps

# Run tests
pytest -q

# Run a specific test file
pytest -q tests/test_dq_rules.py

# Run with verbose output
pytest tests/test_decision_data_quality.py -v
```

**Verify run artifacts** at `runs/<run_id>/`:
- `log.txt`
- `manifest.json`
- `valuation.csv` and `valuation_sensitivity.csv`
- `decision.csv` and `decision.md`

---

## Investment Strategies

Three strategies selectable via config file:

| Strategy | Config file | Approach |
|----------|-------------|----------|
| **Baseline** (default) | `config/config.yaml` | ROIC + FCF yield z-score quality |
| **Dividend Quality** | `config/config_dividend.yaml` | 11 dividend criteria |
| **Graham Defensive** | `config/config_graham.yaml` | 7 Graham defensive criteria |

---

## Data Quality System

Located in `src/data_quality/`:

- **`rules.py`** — 67+ DQ rules with structured rule IDs (`DQ-F001`, `DQ-W*`, etc.) and severity levels: `FAIL`, `WARN`, `PASS`.
- **`schema_contracts.py`** — Pandera schema enforcement for:
  - `MasterSchema` — master dataset
  - `ValuationOutputSchema` — valuation step output
  - `DecisionOutputSchema` — decision step output

**DQ philosophy:**
- `FAIL` rules block a candidate from selection
- `WARN` rules flag issues but allow candidate through
- All exclusions require explicit `reason_*` fields
- Outlier detection: MAD threshold 6.0, minimum 10 samples

---

## Key Conventions

### Joins
- The join key for all merges is **`yahoo_ticker`**
- One-to-one joins are enforced — fail fast on duplicates

### Valuation guard
- `intrinsic_value` and `intrinsic_*` fields **must not** use price data as input
- Allowed inputs: fundamentals, WACC, growth assumptions from config

### Technical analysis scope
- `ma200`, `mad`, `index_price`, `index_ma200` are for gating only
- Market index: `^OSEAX` (Oslo Stock Exchange All Share)

### Audit trail
- Every decision candidate must carry `reason_*` fields explaining any exclusion
- Run artifacts are stored in date-partitioned `runs/<run_id>/` directories

### As-of dating
- All runs take `--asof YYYY-MM-DD`; never use today's date implicitly
- Enables reproducible backtesting and walk-forward testing

---

## Tests as Specification

**Tests are the contract** — always consult test files first when implementing or debugging:

| Test file | What it specifies |
|-----------|-----------------|
| `tests/test_dq_rules.py` | 67+ DQ rule expected outcomes (FAIL/WARN/PASS) |
| `tests/test_decision_data_quality.py` | DQ gate contract (fail vs warn; blocked; outlier policy) |
| `tests/test_schema_contracts.py` | Pandera schema enforcement |
| `tests/test_contracts.py` | Module contract verification |
| `tests/test_valuation_input_audit.py` | Valuation audit + quarterly prefer/require |
| `tests/test_wft*.py` | Walk-forward optimization and cross-validation |
| `tests/test_gui_smoke_ci.py` | GUI initialization (CI only) |

---

## Configuration Reference

Key fields in `config/config.yaml`:

```yaml
paths:
  runs_dir: runs
  data_dir: data
  raw_dir: data/raw
  processed_dir: data/processed

decision:
  quality_strategy: baseline       # or: dividend, graham
  max_price_age_days: 7
  min_fresh_price_coverage: 0.50
  candidate_min_required_fields: 8
  candidate_min_required_ratio: 0.75
  data_quality:
    stale_fundamentals_days: 450
    outlier_mad_threshold: 6.0
  media_red_flags:
    enabled: true
    days_back: 30

valuation:
  prefer_quarterly_reports: true
  require_quarterly_reports: true
  max_quarterly_report_age_days: 220
```

---

## Environment

```bash
# Required secret — never commit
BORSDATA_AUTHKEY=<your_api_key>   # in .env (git-ignored)
```

---

## CI/CD

**File:** `.github/workflows/gui-smoke.yml`
- Platform: `windows-latest`, Python 3.12
- Trigger: push to main, pull requests
- Environment: `QT_QPA_PLATFORM: offscreen`
- Runs: `pytest tests/test_gui_smoke_ci.py`

---

## Git Rules

- **Do not** add or modify anything under `data/`, `runs/`, `logs/` — these are git-ignored.
- **Never commit** secrets or API keys. Use `.env` (git-ignored).
- **Do not** commit large binary exports (`exports/*/*.xlsx`) or full universe ticker lists.
- Feature branches follow the pattern `claude/<feature>` or `fase-<N>/<description>`.

---

## File Reference by Topic

| Topic | Primary file(s) |
|-------|----------------|
| Pipeline order & step contract | `src/run_weekly.py` |
| Valuation engine | `src/valuation.py`, `tests/test_valuation_input_audit.py` |
| Decision engine + DQ gates | `src/decision.py`, `tests/test_decision_data_quality.py` |
| DQ rules catalogue | `src/data_quality/rules.py`, `tests/test_dq_rules.py` |
| Schema contracts | `src/data_quality/schema_contracts.py`, `tests/test_schema_contracts.py` |
| ESEF/XBRL field mapping | `docs/CANONICAL_SCHEMA.md`, `config/canonical_schema.yaml` |
| ESEF integration | `config/mappings/esef_ifrs_default.yaml`, `tests/fixtures/` |
| WFT tuning | `TUNING_PARAMS.md`, `tools/run_wft_optimize.py` |
| Config reference | `config/config.yaml`, `configs/thresholds.yaml` |
| Common utilities | `src/common/` |

---

## Do Not Upload to Claude Projects

- `.env` (contains secrets)
- `exports/*/*.xlsx` (binary output artifacts)
- Large universe ticker lists (`tickers*.csv`) — use a small golden sample instead
- `data/`, `runs/`, `logs/` directories (runtime artifacts)
