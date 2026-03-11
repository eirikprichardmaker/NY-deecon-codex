# NY-deecon-codex

Dette repoet er klargjort for import av kjernefiler fra Børsdata uten å inkludere tunge/binære artefakter.

**Viktig:** kjør hele pipelinen på Windows der dataene ligger (f.eks. `G:\Min disk\NEW DEECON\data`). Codespaces brukes kun til redigering og testing.

**Oppsett:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt   # inkluderer python-dotenv nå
```

==> Hvis du bruker WSL kan du montere G: som `/mnt/g/`.

## Status
- Kilderepo `Borsdata` ble ikke funnet i arbeidsområdet i denne sesjonen.
- Target-repo `NY-deecon-codex` finnes og inneholder n? pipeline-kode (`src/`, `tools/`, `tests/`, `config/`).

## Planlagt smoke test
Kj?r smoke test med n?v?rende kodebase:

```bash
python -m src.run_weekly --asof 2026-02-16 --config ./config/config.yaml --steps valuation,decision
```

Alternativ strategi (dividend quality) er tilgjengelig via egen config:

```bash
python -m src.run_weekly --asof 2026-02-16 --config ./config/config_dividend.yaml --steps valuation,decision
```

Alternativ strategi (graham strategy) er tilgjengelig via egen config:

```bash
python -m src.run_weekly --asof 2026-02-16 --config ./config/config_graham.yaml --steps valuation,decision
```

## Automatisk overvåkning av kvartalsrapporter
Kjor lett watcher som:
- leser forventede rapportdatoer fra `config/report_watch_sources.csv`
- oppdager datoendringer og logger disse i SQLite
- sjekker IR-rapportside rundt forventet dato
- laster ned nye rapportfiler automatisk (idempotent)

```bash
python -m src.report_watch --sources-csv config/report_watch_sources.csv
```

Nyttige argumenter:
- `--db data/processed/report_watch/report_watch.db`
- `--downloads-dir data/raw/ir_auto`
- `--watch-days-before 5 --watch-days-after 5`
- `--no-download` (kun oppdagelse + logging)

Valider og foresla bedre kilder:

```bash
python -m src.report_watch_validate --input config/report_watch_sources.csv --output tmp/report_watch_sources_validated.csv
```

## Daglig oppdatering av aksjekurser (hovedmodell)
Oppdater `data/processed/prices.parquet` og as-of kopi i `data/raw/<asof>/prices.parquet`:

```bash
python -m src.refresh_prices_daily --asof 2026-02-27 --config config/config.yaml --require-fresh-days 3
```

Anbefalt daglig (Windows Task Scheduler):

```powershell
python -m src.refresh_prices_daily --asof (Get-Date -Format yyyy-MM-dd) --config config/config.yaml --require-fresh-days 3
```

Ferdig daglig jobbscript (priser + report watcher, valgfritt weekly decision):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_daily_automation.ps1 -RunWeeklyDecision
```

Registrer to daglige Task Scheduler-jobber (morgen + kveld):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/register_daily_tasks.ps1 -RunWeeklyDecisionInEvening
```

## Import-manifest (forventet innhold fra Børsdata)

> **Nota bene:** Windows GUI / PyInstaller-exe bygg har blitt deprioritert. Filene under `build/pyinstaller` og `dist/` er fjernet og ikke nødvendige for utvikling.

## Import-manifest (forventet innhold fra Børsdata)
- `src/`
- `tools/`
- `tests/`
- `config/` / `configs/`
- `README.md`
- `pyproject.toml` eller `requirements.txt`
- `setup.cfg`
- `.env.example`
- eventuelt `.github/`, `Makefile`, `.pre-commit-config.yaml`

## Viktig
`archive/`, `runs/`, `logs/`, `data/raw/`, `data/freeze/`, virtuelle miljøer og cache-filer skal ikke commit'es.

## WFT Sweep
Kjor konservativ parameter-sweep (egen eval-jobb, ikke pytest-output):

```bash
python tools/run_wft_sweep.py --config config/config.yaml --start 2010 --end 2025 --rebalance monthly --test-window-years 1 --train-window-years 12 --cost-bps 20
```

Scriptet skriver:
- `runs/wft_sweep_<timestamp>/sweep_results.csv`
- `runs/wft_sweep_<timestamp>/sweep_summary.md`

## WFT Optimizer (Nested)
Kjor nested walk-forward optimizer (Strategy A, NO-primary default):

```bash
python tools/run_wft_optimize.py --config config/config.yaml --start 2010 --end 2025 --train-window-years 12 --test-window-years 1 --rebalance monthly --n-trials 50 --seed 42 --holdout-start 2023 --holdout-end 2025
```

Strategy A knobs (lav frihetsgrad):
- faktorvekter over 4 grupper (`quality`, `value`, `lowrisk`, `balance`) med sum=1 og bounds fra `config/config.yaml` (`wft_opt_strategy_a`)
- regularisering mot priorvekter via `weights_reg_lambda`
- `mos_threshold` grid (default `0.30,0.35,0.40`)
- MAD soft-penalty: `mad_min` grid + `mad_penalty_k` grid
- hysterese: `min_hold_months` grid + `score_gap` grid
- holdout: `--holdout-start/--holdout-end` (default 2023-2025, aldri brukt i tuning/select)

Andre defaults:
- `--universe NO` (NORDIC kan brukes ved behov)
- benchmark fallback per land:
  - NO: `^OSEBX -> ^OBX -> ^OSEAX`
  - SE: `^OMXS30 -> ^OMXS`
  - DK: `^OMXC25`
  - FI: `^OMXH25 -> ^HEX`
- kostmodell:
  - kurtasje `0.04%` (`--commission-rate-bps 4`)
  - min `39`, maks `149` per kjop/salg (`--commission-min`, `--commission-max`)
  - ordrestorrelse `100000` lokal valuta (`--order-notional`)
  - FX-kost `0` bps default (`--fx-bps-non-nok`)

Scriptet skriver:
- `runs/wft_opt_<timestamp>/trials.csv`
- `runs/wft_opt_<timestamp>/best_config.yaml`
- `runs/wft_opt_<timestamp>/optimize_summary.md`
- `runs/wft_opt_<timestamp>/holdout_results.csv`

`trials.csv` inkluderer blant annet:
- params: `mos_threshold`, `mad_min`, `mad_penalty_k`, `min_hold_months`, `score_gap`, `weakness_rule_variant`
- vekter/reg: `weight_quality`, `weight_value`, `weight_lowrisk`, `weight_balance`, `weights_reg_lambda`, `weights_reg_penalty`
- train-metrikker: `train_objective`, `train_return`, `train_sharpe`, `train_turnover`, `train_pct_cash`
- OOS-metrikker: `test_return_gross`, `test_return_net`, `test_max_dd`, `test_turnover`, `test_pct_cash`
- benchmark/excess: `benchmark_return`, `excess_return_net`, `benchmark_symbols_used`, `benchmark_missing_months`

## Multi-seed Verification (Strategy A)
Aggreger eksisterende optimize-runs for seed-stabilitet + holdout sammenligning mot baseline:

```bash
python tools/aggregate_experiments.py --mode strategyA --holdout-start 2021 --holdout-end 2022 --seeds 1,2,3,4,5 --latest 10
```

Bruk eksplisitt baseline-run:

```bash
python tools/aggregate_experiments.py --mode strategyA --holdout-start 2021 --holdout-end 2022 --seeds 1,2,3,4,5 --latest 10 --baseline-run-id <run_id>
```

Kjor baseline automatisk (n-trials=1, faste baseline-knobs):

```bash
python tools/aggregate_experiments.py --mode strategyA --holdout-start 2021 --holdout-end 2022 --seeds 1,2,3,4,5 --latest 10 --run-baseline
```

Output i `experiments/`:
- `strategyA_seed_summary.csv` (en rad per seed/run, med run_id + filstier for audit)
- `strategyA_seed_summary.md` (rangert tabell + kommentarer)
- `strategyA_recommendation.md` (beslutning: `freeze config` vs `tighten regularization/penalties`)
