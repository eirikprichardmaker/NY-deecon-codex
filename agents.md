# AGENTS.md — NEW DEECON (Deecon)

## 0) Formål (kort)
Bygg en reproduserbar, auditérbar modell som velger **maks 1 nordisk aksje ellers CASH**, der:
- **Fundamental verdsettelse (intrinsic) er eneste prisingsgrunnlag**
- Teknisk analyse brukes **kun** til timing/risikogating (entry/exit), aldri som input til intrinsic
- Backtest og benchmark bruker **total return (reinvesterte utbytter)** som standard
- Rebalanse: **siste børsdag i kvartalet**, men **exit/entry er flytende** (re-entry tillatt)

## 1) Non-negotiables (må alltid respekteres)
1. Portefølje = **0 eller 1 posisjon** (single-stock-or-cash). CASH er et eksplisitt output.
2. **Intrinsic beregnes kun fra fundamentals**. Ikke bruk pris i verdsettelsen.
3. TA = **gating** (MA200/MAD + indeksregime), ikke rangering og ikke verdsettelse.
4. All logikk må være **forklarbar og auditérbar**:
   - `screen_basic.csv` må ha reason-kolonner/tags for hvorfor ticker falt ut
   - `decision.md` skal være sendbar (begrunnelse + hva som slo inn)
5. Ingen lookahead: fundamentals må merges **as-of tilgjengelighetsdato** der det finnes.
6. Data/runs/freeze er store: **ikke commit** store data/artefakter.

## 2) Fast policy (frosset blueprint)
### Universe
- Hele Norden (NO/SE/DK/FI). Ingen “NO-only” særregler.

### Return og dividender
- Standard = **Total Return (TR)** med reinvestering av utbytter.
- Hvis TR-serie ikke finnes: konstruer TR eksplisitt fra pris + utbytte (reinvest).

### Kalender og tradingregler
- Kvartalsdato = siste børsdag i kvartalet (Q1/Q2/Q3/Q4).
- Intra-kvartal:
  - Hvis investert og teknisk gate feiler => SELL til CASH.
  - Hvis i CASH og teknisk gate blir OK igjen => re-entry er tillatt.
- Start implementasjon med “låst kandidat per kvartal” (samme kvartalsrangering), for å redusere frihetsgrader.
  - Kandidat oppdatering intra-kvartal kan legges til senere via config.

### Manual override (tillatt, men begrenset)
- Overrides kan kun **blokkere/stramme inn** eller **justere antakelser** (WACC/COE/terminalvekst/scenario).
- Overrides kan aldri “force buy” eller omgå hard gates.
- Alle overrides må logges til `runs/<run_id>/override_audit.json` og gjenspeiles i `decision.md`.

## 3) Prioritet: hva agenten skal bygge først (rekkefølge)
> Ikke start med mer tuning før simulator + funnel-logging fungerer.

### PR #1 (først): Quarterly calendar + state machine + event-driven entry/exit
Mål: få en korrekt kvartals-simulator som støtter flytende entry/exit og re-entry.

Leveranse:
- `src/deecon/backtest/calendar.py`:
  - funksjon `quarter_end_trading_days(exchange_calendar, start, end) -> list[date]`
  - siste børsdag i kvartalet
- `src/deecon/backtest/state_machine.py`:
  - `CASH` / `INVESTED(ticker)`
  - event: `rebalance_date` (kvartal) + `risk_check_date` (daglig/ukentlig)
  - re-entry tillatt
- `tests/test_backtest_quarterly.py`:
  - tester at kvartalsdatoer er korrekt generert
  - tester at state machine gjør SELL ved teknisk fail, og kan re-enter ved OK

Akseptkriterier:
- `pytest -q` grønn
- En minimal “toy backtest” gir ikke 100% cash hvis gates er deaktivert i testconfig.

### PR #2: Total Return simulator (reinvest dividends)
Mål: standardiser TR som sannhet.

Leveranse:
- `src/deecon/backtest/simulator.py`:
  - single-stock-or-cash, kvartalsrebalance + intra-kvartal events
  - TR beregning (serie eller konstruert)
- `src/deecon/backtest/metrics.py`:
  - CAGR_TR, maxDD, turnover, pct_cash, benchmark TR, excess metrics
- tester for reinvestert utbytte (en syntetisk dividendestrøm)

### PR #3: Funnel logging (“why cash?”)
Mål: gjøre “always cash” diagnostiserbart.

Leveranse:
- `runs/<run_id>/gate_funnel.csv` + `gate_funnel.md`
- For hver rebalance_date: antall tickere som faller på hver gate:
  - data coverage, MOS, value_creation, quality_weak, stock_ma200, stock_mad, index_present, index_ma200, index_mad

## 4) Kommandoer (Windows/PowerShell)
- Kjør tester:
  - `python -m pytest -q`
- Kjør WFT (eksempel; juster til repoets CLI):
  - `python -m src.wft --config config/config.yaml --start 2006 --end 2025 --rebalance quarterly`
- Kjør kvartals-simulator-smoke:
  - `python -m src.deecon.cli.run --asof 2026-02-16 --steps backtest`

## 5) Repo-konvensjoner
- Output-kontrakt per run:
  - `runs/<run_id>/metadata.json`
  - `runs/<run_id>/quality.md`
  - `runs/<run_id>/screen_basic.csv`
  - `runs/<run_id>/shortlist.csv`
  - `runs/<run_id>/valuation.csv` + `valuation_sensitivity.csv`
  - `runs/<run_id>/decision.csv` + `decision.md`
- Ikke skriv til `data/raw` utenfor “freeze”-steg. Bruk immutable snapshots.

## 6) Guardrails for agenten
DO:
- Lag små, testbare commits
- Oppdater/legg til tester når du endrer logikk
- Logg beslutningsgrunner eksplisitt

DON’T:
- Ikke endre verdsettelseslogikk til å bruke markedspris som input
- Ikke legg inn nye tunable parametre uten at de er dokumentert i `TUNING_PARAMS.md`
- Ikke øk frihetsgrader (intra-kvartal re-ranking) før baseline er stabil og har funnel-logging

## 7) Når du er i tvil
- Velg forklarbarhet + reproduksjon foran “smartness”.
- Hvis en endring kan øke risiko for lookahead, stopp og legg inn as-of/available_date støtte først.