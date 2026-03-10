# NY-deecon-codex — Kodebaseoversikt

Deterministisk nordisk aksjeinvesteringssystem som automatisk velger maksimalt én nordisk aksje (NO/SE/DK/FI) eller anbefaler CASH. Alle kjøringer produserer revisjonsklare artefakter med strenge datakvalitetskontrakter og full beslutningssporbarhet.

---

## Arkitektur

### Hoved-pipeline (`src/run_weekly.py`)

Stegene kjøres sekvensielt og deterministisk:

| Steg | Modul | Formål |
|------|-------|--------|
| `ingest` | `src.ingest_validate` | Validerer og laster fundamental-data |
| `transform_fundamentals` | `src.transform_fundamentals` | Transformerer ESEF/XBRL-data |
| `freeze_golden` | `src.freeze_golden_fundamentals` | Låser gylden kopi av fundamentaler |
| `transform_prices` | `src.transform_prices` | Priser og indeksdata |
| `master` | `src.build_master` | Bygger master-datasett (aksjer + indekser + verdsettingsinput) |
| `factors` | `src.compute_factors` | Kvalitet, verdi og risikoberegninger |
| `cost_of_capital` | `src.cost_of_capital` | WACC og egenkapitalkostnad |
| `valuation` | `src.valuation` | DCF-basert intrinsic value + sensitivitetsanalyse |
| `decision` | `src.decision` | Multi-gate beslutningsmotor |

Hvert modul implementerer kontrakten: `run(ctx, log) -> None/int`

---

## Nøkkelkomponenter

### Verdsettelsesmotor (`src/valuation.py`)
- DCF-basert intrinsic value fra fundamentaler kun (ingen pris som input — ikke-forhandlbart)
- Quarterly R12-bro for fremtidig kontantstrømsestimat
- Output: `valuation.csv` (ticker, intrinsic_value, mos, wacc_used, sensitivity) og `valuation_sensitivity.csv`
- Audit-JSON for full sporbarhet

### Beslutningsmotor (`src/decision.py`)
Multi-gate filtreringspipeline:

**Datakvalitetsporter** (`src/data_quality/rules.py`):
- 67 unike regler med regel-IDer
- FAIL-regler: hard stopp, tvinger CASH
- WARN-regler: logges, ikke-blokkerende
- Sektorspesifikke unntak (banker fritatt fra EV-sjekker)

**Tekniske porter** (kun timing, påvirker aldri verdsettelse):
- MA200-filter (aksje må være over 200-dagers glidende snitt)
- MAD (mean absolute deviation) terskel
- Indeksregime-validering (benchmark må også være over MA200)

**Fundamentale porter**:
- Baseline: ROIC + FCF yield z-scores
- Alternativ: Dividend Quality (11 kriterier)
- Alternativ: Graham Strategy (7 defensive kriterier)

**Margin of Safety (MoS) terskler**:
- `mos_min` (standard: 0.30)
- `mos_high_uncertainty` (standard: 0.40 for høy-risiko selskaper)

**Output** (`decision.csv`):
- Valgt ticker eller "CASH"
- `reason_*`-felter (hvorfor ekskludert / hvorfor valgt)
- Alle portstatus (fundamental_ok, technical_ok)
- Snapshot av verdsettingsmetrikker

### Agent-rammeverk (`src/agents/`)

**Agent B — Investment Skeptic** (`investment_skeptic.py`):
- LLM-basert (GPT-4o) risikoanalyse av topp-N kandidater
- Identifiserer verdsettingsrisikoer: terminal value-dominans, sensitivitet, datakvalitet, marginer/vekstantagelser, sykliske topper osv.
- Output: PASS, VETO_CASH, eller REQUEST_REVIEW (ingen BUY-mulighet)
- Fallback deterministisk pre-sjekk
- Kostnadskontroll: maks 5 tickere analysert per kjøring

**Agent C — Dossier Writer** (`dossier_writer.py`):
- Genererer narrativ markdown-rapport for valgt aksje
- Dokumenterer portelogikk, nøkkelrisikoer, verdsettingsoppsummering
- Ikke-beslutning (kan ikke overstyre pipeline-output)

**Agent Runner** (`runner.py`):
- Orkestrerer agent-kjøring mot shortlist
- Håndterer OpenAI-klient, retries og feilhåndtering
- Skriver `skeptic_results.json`

### Datakvalitet (`src/data_quality/`)

**rules.py** — 67 revisjonsklare DQ-regler:
- Hvert regel har: rule_id, description, severity (FAIL/WARN/PASS), fields, check_fn
- Eksempler: DQ-F001 (pris > 0), DQ-F002 (market_cap > 0), DQ-W015 (utdaterte fundamentaler)

### Felles verktøy (`src/common/`)

- **config.py**: `RunContext` dataclass (asof, config_path, run_id, run_dir, cfg)
- **log.py**: Sentralisert logging
- **io.py**: Parquet I/O-hjelpere
- **errors.py**: Tilpasset unntakshierarki

### Walk-Forward Testing (`src/wft.py`)
Nestede walk-forward optimaliseringsverktøy for backtesting av parametere som `mos_threshold`, `mad_min` og faktorvekter.

---

## Konfigurasjonssystem

| Fil | Strategi |
|-----|---------|
| `config/config.yaml` | Standard baseline (ROIC+FCF) |
| `config/config_dividend.yaml` | Utbytte-fokusert (11 kriterier) |
| `config/config_graham.yaml` | Graham defensiv (7 kriterier) |

Støtter konfigurarvsarvning via `includes:` (sources.yaml, thresholds.yaml).

---

## Dataflyt

```
Börsdata API (fundamentaler)
Yahoo Finance (priser/indekser)        →  data/raw/{asof}/
IR-rapporter (auto-nedlastet)

Pipeline-prosessering                  →  data/processed/

Per-kjøring artefakter:               →  runs/{run_id}/
  log.txt
  manifest.json
  valuation.csv + valuation_sensitivity.csv
  decision.csv + decision.md
  skeptic_results.json
  decision_agent.md + decision_agent.json
```

---

## Kjøring

```bash
# Alle steg
python -m src.run_weekly --asof 2026-02-16 --config ./config/config.yaml --steps all

# Spesifikke steg
python -m src.run_weekly --asof 2026-02-16 --config ./config/config.yaml --steps valuation,decision

# List tilgjengelige steg
python -m src.run_weekly --list-steps
```

**Windows-automatisering:**
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_daily_automation.ps1 -RunWeeklyDecision
```

---

## Testdekning

45+ testfiler med kontraktstester for alle nøkkelkomponenter:

| Testfil | Hva testes |
|---------|-----------|
| `test_valuation_input_audit.py` | Quarterly prefer/require, auditfelter |
| `test_decision_data_quality.py` | DQ-regelinformasjon, outlier-policy |
| `test_dq_rules.py` | 67 regelimplementasjoner |
| `test_can_invest.py` | Juridisk/tilgjengelighetssjekker |
| `test_investment_skeptic.py` | Agent risikofinningskategorier |
| `test_runner.py` | Agent-orkestrering |
| `test_dossier_writer.py` | Dossier-skjema og -generering |
| `test_pipeline_robustness.py` | Ende-til-ende pipeline |

---

## Teknologistakk

- **Kjerne**: Python 3.8+
- **Data**: pandas, numpy, pyarrow (parquet), yaml
- **Testing**: pytest
- **UI**: PySide6 (Qt), tkinter
- **XBRL/ESEF**: arelle, beautifulsoup4
- **Finans-API**: yfinance, Börsdata SDK
- **LLM** (valgfritt): OpenAI GPT-4o

---

## Ikke-forhandlbare prinsipper

1. **Deterministisk as-of pipeline**: Alle kjøringer med `--asof` produserer revisjonsklare artefakter
2. **Ingen stille fjerninger**: Hvert ekskludert selskap har et `reason_*`-felt
3. **Intrinsic value fra fundamentaler kun**: Ingen pris brukt som verdsettingsinput
4. **Teknisk analyse kun for timing**: MA200/MAD påvirker ikke `intrinsic_value`
5. **En-til-en joins**: Join-nøkkel er `yahoo_ticker`; feiler ved duplikater
6. **Én beslutning**: Velger maks én aksje, ellers CASH
