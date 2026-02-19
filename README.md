# NY-deecon-codex

Dette repoet er klargjort for import av kjernefiler fra Børsdata uten å inkludere tunge/binære artefakter.

## Status
- Kilderepo `Borsdata` ble ikke funnet i arbeidsområdet i denne sesjonen.
- Target-repo `NY-deecon-codex` finnes, men inneholder foreløpig ikke pipeline-kode (`src/` mangler).

## Planlagt smoke test
Når `src/` og `config/` er importert, kjør:

```bash
python -m src.run_weekly --asof 2026-02-16 --config ./config/config.yaml --steps valuation,decision
```

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
Kjor nested walk-forward optimizer (NO-primary default):

```bash
python tools/run_wft_optimize.py --config config/config.yaml --start 2010 --end 2025 --train-window-years 12 --test-window-years 1 --rebalance monthly --n-trials 50 --seed 42
```

Viktige defaults:
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

`trials.csv` inkluderer blant annet:
- params: `mos_threshold`, `mad_min`, `weakness_rule_variant`
- train-metrikker: `train_objective`, `train_return`, `train_sharpe`, `train_turnover`, `train_pct_cash`
- OOS-metrikker: `test_return_gross`, `test_return_net`, `test_max_dd`, `test_turnover`, `test_pct_cash`
- benchmark/excess: `benchmark_return`, `excess_return_net`, `benchmark_symbols_used`, `benchmark_missing_months`
