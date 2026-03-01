# KPI Shortlist And Testplan

## Why 21, not 26
- `21` is the currently captured global set (`current_21`).
- `26` refers to `legacy_16 + 10 new non-overlapping IDs` (`core_26`).
- Difference vs current:
  - Missing from current if you want `core_26`: `10,23,33,54,68,71,73`
  - Also choose whether to keep `98` (included in `current_21`, excluded in `core_26`).

Reference sets: [kpi_candidate_sets.yaml](/g:/Min disk/NEW DEECON/config/kpi_candidate_sets.yaml)

## Recommended shortlist
- Primary (`core_26`): balanced between legacy Nordic set and new fundamentals/growth.
- Secondary (`extended_27`): same as `core_26` plus `98`.

## KPI history parameters (API/freeze)
- `kpi_id`: metric identifier.
- `period`: `year`, `r12`, optionally `quarter`.
- `value_type`: typically `mean` in this repo.
- `instList`: batched instrument list.
- `maxCount`: optional history depth cap.

## Execution plan
1. Ensure coverage for target set (`year` and `r12`) on global ids.
2. Validate completeness from `state.json`.
3. Run baseline vs candidate model evaluation (ablation/WFT).
4. Keep only KPI additions that improve OOS robustness.

## Commands
### A) Freeze missing IDs for core_26 (year/r12)
Use global ids captured at:
`data/freeze/borsdata_proplus/2026-02-28/meta_global_capture/global_ids.csv`

```powershell
python tools/borsdata_freeze.py `
  --asof 2026-02-28 `
  --ids-csv data/freeze/borsdata_proplus/2026-02-28/meta_global_capture/global_ids.csv `
  --ids-col ins_id `
  --out-root data/freeze/borsdata_proplus_freeze `
  --only kpi `
  --kpi-ids 10,23,33,54,68,71,73 `
  --kpi-period year `
  --kpi-value-type mean `
  --batch-size 50 `
  --pace-s 0.03
```

```powershell
python tools/borsdata_freeze.py `
  --asof 2026-02-28 `
  --ids-csv data/freeze/borsdata_proplus/2026-02-28/meta_global_capture/global_ids.csv `
  --ids-col ins_id `
  --out-root data/freeze/borsdata_proplus_freeze `
  --only kpi `
  --kpi-ids 10,23,33,54,68,71,73 `
  --kpi-period r12 `
  --kpi-value-type mean `
  --batch-size 50 `
  --pace-s 0.03
```

### B) Coverage report for a set
```powershell
python tools/kpi_coverage_report.py `
  --state-path data/freeze/borsdata_proplus_freeze/2026-02-28/state.json `
  --set-file config/kpi_candidate_sets.yaml `
  --set-name core_26 `
  --periods year,r12 `
  --value-type mean `
  --ids-csv data/freeze/borsdata_proplus/2026-02-28/meta_global_capture/global_ids.csv `
  --batch-size 50
```

### C) Pipeline sanity check
```powershell
python -m src.run_weekly --asof 2026-02-28 --config ./config/config.yaml --steps valuation,decision
python -m pytest -q
```

### D) Evaluation runs (existing tooling)
```powershell
python tools/run_wft_sweep.py --config config/config.yaml --start 2010 --end 2025 --rebalance monthly --test-window-years 1 --train-window-years 12 --cost-bps 20
python tools/ablation_decision_rules.py
```

## Decision criteria (practical)
- Keep KPI expansion only if all are true:
  - OOS return and/or Sharpe improves without higher max drawdown.
  - Turnover does not deteriorate materially.
  - Cash share does not rise excessively without risk benefit.
  - Coverage remains stable (near-full batch completion per KPI/period).

