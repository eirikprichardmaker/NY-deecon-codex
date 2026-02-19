# Borsdata - Nordic Single-Stock Model

Nordic single-stock pipeline med maks 1 aksje eller CASH per kjoring.
Malet er forklarbar beslutning basert pa fundamentals + tekniske risikofiltre.

## Oppsett (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Kjoring

### Full ukentlig/manedlig kjoring
```powershell
python -m src.run_weekly --asof 2026-02-16 --config .\config\config.yaml
```

### Kun valuation + decision (DoD-check)
```powershell
python -m src.run_weekly --asof 2026-02-16 --config .\config\config.yaml --steps valuation,decision
```

### Liste steg
```powershell
python -m src.run_weekly --list-steps
```

### Optional IR step (Task B)
`ir_reports` er optional og kjores kun hvis steget velges eksplisitt.

```powershell
python -m src.run_weekly --asof 2026-02-16 --config .\config\config.yaml --steps ir_reports
```

## IR-konfigurasjon (ticker -> IR URL)

1. Kopier eksempel:
```powershell
Copy-Item .\config\ir_sources.example.csv .\config\ir_sources.csv
```
2. Oppdater `config\ir_sources.csv` med egne IR-kilder.
3. `config\config.yaml` kan bruke standard:
```yaml
ir_reports:
  mapping_csv: config/ir_sources.csv
  rate_limit_sec: 1.0
  timeout_sec: 20
```

CSV-format:
- `ticker`: intern ticker (f.eks. `EQNR`)
- `url`: IR-side eller rapport-URL
- `source`: valgfri kilde-tag
- `period`: valgfri periode-tag (`Q1`, `Q2`, `Q3`, `Q4`, `FY`)
- `report_date`: valgfri rapportdato (`YYYY-MM-DD`)

## Artefakter

Kjoringer skrives til `runs/<run_id>/`:

- `manifest.json` - run metadata
- `quality.md` - data quality og merge-status (ticker/yahoo_ticker/prisdekning)
- `screen_basic.csv` - screening med `fundamental_ok`, `technical_ok`, reason-kolonner
- `shortlist.csv` - kandidater som passer filter
- `valuation.csv` - intrinsic beregninger
- `decision.csv` / `decision.md` - endelig anbefaling (ticker eller CASH)

I tillegg brukes mellomfiler i `data/raw/`, `data/processed/` og `data/golden/`.

IR-artefakter:
- `data/raw/ir/<ticker>/<date>/...` (raw filer / cache)
- `data/raw/ir/index.parquet`
  - kolonner: `ticker, url, report_date, period, source, status, error_code`
- `data/golden/ir_facts.parquet`
  - kolonner: `ticker, url, report_date, period, source, status, error_code, fact_revenue, fact_ebit, fact_eps, text_len`

## Data-kontrakt (viktig)

- Portefolje-prinsipp: alltid maks 1 aksje, ellers CASH.
- `yahoo_ticker` handteres robust:
  1. infereres fra ticker nar mulig,
  2. mapes fra `config/tickers*.csv`,
  3. ellers markeres raden eksplisitt med `missing_price` + reason (ingen stille drop).
- Screening-output skal inkludere forklaringskolonner (`reason_fundamental_fail`, `reason_technical_fail`).

## Tester

```powershell
python -m pytest -q
```

## PowerShell-kommandoer (Task B)

Deps:
```powershell
python -m pip install -r requirements.txt
```

Kun IR-step:
```powershell
python -m src.run_weekly --asof 2026-02-16 --config .\config\config.yaml --steps ir_reports --run-dir .\runs\ir_only_20260216
```

Full run (uten IR):
```powershell
python -m src.run_weekly --asof 2026-02-16 --config .\config\config.yaml
```

Tester:
```powershell
python -m pytest -q
```

## Feilsoking

### `prices.parquet` mangler `yahoo_ticker`
- Pipeline forsoker infer + mapping automatisk.
- Se `runs/<run_id>/quality.md` for antall rader uten mapping.

### Ingen kandidater i decision
- Dette er forventet fail-fast-adferd: output blir `CASH`.
- Sjekk `screen_basic.csv` og `reason_*` kolonner for hvorfor hver ticker ble forkastet.

### Google Drive path (`G:\Min disk\...`)
- Bruk absolutte stier i `config/config.yaml` / `configs/sources.yaml`.
- Pipeline bruker `Path(...)` og stotter Windows-stier direkte.
