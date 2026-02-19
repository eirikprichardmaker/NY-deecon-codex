# Børsdata – Nordic Single-Stock Model

Nordic single-stock pipeline med **maks 1 aksje eller CASH** per kjøring.
Målet er forklarbar beslutning basert på fundamentals + tekniske risikofiltre.

## Oppsett (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Kjøring

### Full ukentlig/månedlig kjøring
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

## Artefakter

Kjøringer skrives til `runs/<run_id>/`:

- `manifest.json` – run metadata
- `quality.md` – data quality og merge-status (ticker/yahoo_ticker/prisdekning)
- `screen_basic.csv` – screening med `fundamental_ok`, `technical_ok`, reason-kolonner
- `shortlist.csv` – kandidater som passer filter
- `valuation.csv` – intrinsic beregninger
- `decision.csv` / `decision.md` – endelig anbefaling (ticker eller CASH)

I tillegg brukes mellomfiler i `data/raw/`, `data/processed/` og `data/golden/`.

## Data-kontrakt (viktig)

- Portefølje-prinsipp: alltid maks 1 aksje, ellers CASH.
- `yahoo_ticker` håndteres robust:
  1. infereres fra ticker når mulig,
  2. mapes fra `config/tickers*.csv`,
  3. ellers markeres raden eksplisitt med `missing_price` + reason (ingen stille drop).
- Screening-output skal inkludere forklaringskolonner (`reason_fundamental_fail`, `reason_technical_fail`).

## Tester

```powershell
pytest -q
```

## Feilsøking

### `prices.parquet` mangler `yahoo_ticker`
- Pipeline forsøker infer + mapping automatisk.
- Se `runs/<run_id>/quality.md` for antall rader uten mapping.

### Ingen kandidater i decision
- Dette er forventet fail-fast-adferd: output blir `CASH`.
- Sjekk `screen_basic.csv` og `reason_*` kolonner for hvorfor hver ticker ble forkastet.

### Google Drive path (`G:\Min disk\...`)
- Bruk absolutte stier i `config/config.yaml` / `configs/sources.yaml`.
- Pipeline bruker `Path(...)` og støtter Windows-stier direkte.
