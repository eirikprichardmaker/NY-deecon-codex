# Komplett systemarkitektur for Deecon-prosjektet
## Executive Summary
Du har allerede en tydelig Deecon-baseline: deterministiske kjøringer (--asof) som velger maks én nordisk aksje eller CASH, med forklarbare artefakter per run.
Den raskeste veien til en «modell som virker» er å låse en hard separasjon: fundamental verdsettelse (intrinsic) = eneste prisingsgrunnlag, mens teknisk analyse kun er timing-/risikogating og aldri påvirker intrinsic beregning. Dette er allerede reflektert i WFT/decision-reglene der MOS, value-creation og kvalitet er primære, mens MA200/MAD og indekskrav er tekniske filtre.

Kritisk funn: “always cash” oppstår i dine seed/holdout-sammendrag (holdout_pct_cash=1.0 og alle metrikker=0), som betyr at validering/tuning ikke er meningsfull før datadekning og/eller filterlogikk er verifisert gjennom en «kan investere»-test.
I logs finnes et sterkt signal om roten til problemet: svært lav prisdekning i master (“price coverage adj_close=0.041”) i en tidlig run, som typisk vil trigge tekniske krav og/eller gi missing_price-effekter. 

Hovedleveranse i denne rapporten: en full, modulær arkitektur (med mermaid-diagram og repo-/mappe-layout), datakontrakt, Pro+-“freeze”-liste, robust tuningplan (kvartalsvis rebalansering), og en konkret diagnoseprotokoll mot «always cash», pluss en 3-ukers plan tilrettelagt for maksimal uttelling av Börsdata Pro+. 

## Kildegrunnlag (lokale repo-kilder)
- `AGENTS.md`
- `README_borsdata.md`
- `README.md`
- `TUNING_PARAMS.md`
- `experiments/strategyA_seed_summary.md`
- `experiments/strategyA_recommendation.md`
- `src/wft.py`
- `src/decision.py`
- `tools/run_wft_sweep.py`
- `tools/run_wft_optimize.py`

## Intern filkatalog og baselineantakelser
### Katalogiserte n?kkelfiler fra repoet
Følgende filer er identifisert som kjernegrunnlag for dagens Deecon-baseline og for videre Codex-implementering:

AGENTS.md – definerer “non-negotiable contracts” (maks 1 aksje ellers CASH; eksplisitte skjema; yahoo_ticker som merge-nøkkel; forklarbarhet; Windows + Google Drive-stier). Den skisserer også modulnavn som finnes/forventes i src/ (pipeline orchestrator, build_panel, build_master).
README_borsdata.md / README.md (Børsdata) – dokumenterer artefaktstruktur i runs/<run_id>/ og datakontrakt (ingen stille dropp; missing_price + reason; screen_basic.csv med reason-kolonner), samt optional IR-step og IR-artefakter.
- README.md (NY-deecon-codex) ? beskriver importmanifest og WFT sweep/optimizer-kommandoer og "Strategy A knobs".
TUNING_PARAMS.md – spesifiserer hvilke parametre som faktisk tunas i WFT i dag (mos_threshold, mad_min, weakness_rule_variant) og eksakt hvilke filtre som avgjør CASH vs kandidat.
- experiments/strategyA_seed_summary.md / experiments/strategyA_recommendation.md ? gir konkrete seed-resultater, vinduer og GO/NO-GO-beslutningsgrunnlag.
strategyA_seed_summary.md / strategyA_recommendation.md – viser holdout-oppsett, baseline-regler, “robust utility”-formel og bekrefter «always cash» i holdout for flere seeds/runs.
- runs/<run_id>/quality.md, screen_basic.csv og decision.md brukes som prim?re audit-kilder for datadekning, filterutfall og beslutningsgrunnlag.
### Baselineantakelser (eksplisitt merket)
Dette legges til grunn fordi det er dokumentert i interne filer; hvis noe avviker fra din Deecon-modell, må det korrigeres:

Portefølje = maks 1 posisjon (aksje) ellers CASH.
Kjøringer er deterministiske via --asof og skriver audit-artefakter per run.
Screening/decision skal være forklarbar med reason_*-kolonner.
WFT-filter for “eligible” inkluderer MOS, value-creation-base, quality weak count og tekniske krav (MA200/MAD for aksje + indeks).
Dagens WFT tuning-parameterrom er med vilje lav-frihetsgrad (i WFTParams) og skiller mellom WFT-tunede parametre og config-drevne terskler i decision.
## Systemarkitektur og repo-/mappeoppsett
### Arkitekturdiagram (mermaid)
```mermaid
flowchart TD
  A[Börsdata Pro+ API / export] --> B[Ingest & Validate]
  B --> C[data/raw + data/freeze (immutable snapshots)]
  C --> D[Normalize / Transform Fundamentals]
  C --> E[Normalize / Transform Prices]
  D --> F[Master Merge (as-of) + Quality Report]
  E --> F
  F --> G[Factor & Quality Engine]
  F --> H[Cost of Capital + Risk Flags]
  G --> I[Valuation Engine (DCF / DDM / RIM)]
  H --> I
  I --> J[Scenario & Sensitivity Engine]
  I --> K[Screening + MOS + Value Creation Gate]
  E --> L[Technical Risk Filter (MA200/MAD + index regime)]
  K --> M[Decision Engine (rank, pick 1 else CASH)]
  L --> M
  N[Manual Override UI + Audit Log] --> M
  M --> O[runs/<run_id>/ artifacts: manifest, quality, screen, shortlist, valuation, decision]
  P[Backtest/WFT/Tuning (quarterly)] --> F
  P --> G
  P --> H
  P --> I
  P --> M
  Q[Optional: IR Reports & Fact Extraction] --> O
```
### Anbefalt repo-/mappe-layout (Codex-vennlig)
Dette er en struktur som matcher artefakt- og steglogikken i dine interne docs (runs/, data/raw/processed/golden) og samtidig skiller “immutable data” fra kode.

```text
NEW DEECON/
  config/
    config.yaml
    sources.yaml
    ir_sources.csv
    tickers_no.csv
    tickers_se.csv
    ...
  src/
    deecon/
      __init__.py
      cli/
        run.py                # --asof, --steps, --run-dir
      ingest/
        borsdata_api.py       # Pro+ API client + rate limit + retries
        export_freeze.py      # freeze til data/freeze/...
        validate_schema.py    # schema validators (fail-fast)
      normalize/
        fundamentals.py       # periodisering, TTM, as-of tilgjengelighet
        prices.py             # justeringer + tekniske features
        corporate_actions.py  # splits/dividends/buyback-normalisering
      master/
        build_master.py       # joins + as-of merge + quality flags
        quality_report.py     # writes quality.md + coverage metrics
      factors/
        compute.py            # quality/value/lowrisk/balance features
        ml_weights.py         # valgfri black-box vekting (se governance)
      risk/
        cost_of_capital.py    # beta/rf proxy + COE/WACC + high_risk_flag
        technical.py          # MA200/MAD + indeksregime gating
      valuation/
        dcf.py
        ddm.py
        rim.py
        engine.py             # velger modelltype pr selskap
        sensitivity.py        # valuation_sensitivity.csv
      decision/
        screening.py          # mos + value_creation + quality gate
        ranking.py            # score + tie-break
        overrides.py          # manual override rules + audit
        write_decision.py     # decision.csv + decision.md
      backtest/
        calendar.py           # quarterly rebalance schedule
        simulator.py          # single-stock-or-cash portfolio simulator
        wft.py                # walk-forward driver
        optimizer.py          # nested optimizer + holdout handling
        metrics.py            # total return, drawdown, turnover, pct_cash
      schemas/
        *.json                # data contract schemas
      utils/
        dates.py
        io.py
        logging.py
  tools/
    run_wft_sweep.py
    run_wft_optimize.py
    aggregate_experiments.py
  tests/
    test_schemas.py
    test_asof_merge.py
    test_valuation_models.py
    test_decision_gates.py
    test_backtest_quarterly.py
  data/
    raw/
    freeze/
    processed/
    golden/
  runs/
    <run_id>/
      manifest.json
      quality.md
      screen_basic.csv
      shortlist.csv
      valuation.csv
      valuation_sensitivity.csv
      decision.csv
      decision.md
  experiments/
    strategyA_seed_summary.*
    strategyA_recommendation.*
Viktig: Du har allerede en “ikke commit tunge artefakter”-policy i interne docs; behold den: archive/, runs/, logs/, data/raw/, data/freeze/ skal ikke i git.

## Modulspecifikasjon (komplett) med klare grensesnitt
Nedenfor er modulspesifikasjonene du kan gi til Codex som “kontrakter”: hver modul har input, output, invariants og testkrav. (Detaljer du ikke har spesifisert er markert som uspesifisert.)

### Data ingestion
Formål: hente/eksportere Pro+ data og fryse til lokale snapshots for reproduserbare backtester.

Kilde: Börsdata API kan levere “share prices, company data, report data and key figures” i JSON; REST-API er flyttet til Pro+ (1 feb 2025). 
Kontrakt: Alle uttrekk må være versjonert (export-dato + API-tilstand + univers-liste), og frosset som immutable snapshots i data/freeze/.
Output (minimum):
data/freeze/<export_date>/instruments.parquet
data/freeze/<export_date>/prices_daily.parquet
data/freeze/<export_date>/reports_quarterly.parquet
data/freeze/<export_date>/reports_annual.parquet
data/freeze/<export_date>/ratios.parquet (om du bruker ferdige KPIer)
data/freeze/<export_date>/corporate_actions.parquet (hvis tilgjengelig; ellers bygg fra pris/dividend data)
Policy/govenance: Börsdata beskriver eksplisitte bruksbegrensninger (for privat analyse; ikke dele API-data eller bygge eksterne widgets som viser API-data). Dette må reflekteres i arkitektur: freeze ligger lokalt, ikke publiseres. 
### Normalisering
Formål: gjøre regnskaps- og prisdata sammenlignbare og safe for faktor-/verdsettingslogikk.

Rapportlogikk/TTM: Börsdata forklarer at nøkkeltall bør bygges på 12 mnd, og at TTM bygges som sum av siste fire kvartaler for å kunne oppdatere raskt etter kvartalsrapporter. Dette er et naturlig normaliseringssteg for deg: kalkuler TTM panel per as-of. 
Bruk “as-of availability”: rapporter har faktisk publiserings-/tilgjengelighetsdato; Börsdata beskriver også at det kan være tidslagg på mindre selskaper. Dette betyr at fundamental panel må inkludere available_date og merges as-of for å unngå lookahead. 
Prisnormalisering: generér MA21/MA200/MAD og over_MA200 som eksplisitte features (matcher dine interne referanser).
### Valuation engine (DCF/DDM/RIM)
Formål: produsere intrinsic value eksklusivt fra fundamentals (A-kravet ditt), med sensitivitet.

DCF (FCFF/FCFE): Aswath Damodaran beskriver DCF som nåverdi av forventede kontantstrømmer, og at intrinsic value estimeres fra kontantstrøm, vekst og risiko (diskonteringsrate). Dette støtter at markedskurs ikke skal være input til verdsettelse. 
DDM/FCFE: Damodaran har egne forelesningsnotater for dividend discount-modeller; i arkitektur bør DDM/FCFE brukes når utbyttepolitikk er stabil og/eller bank/finanscases krever equity-fokus. 
RIM (Residual Income Model): (ikke dokumentert i dine interne filer) – foreslått som egen modul der ROE – COE og bokført egenkapital er sentralt. Detaljert formel og inputkrav er uspesifisert i dine interne docs, så dette må eksplisitt defineres i Deecon-spesifikasjonen før implementasjon.
Output-kontrakt: dagens pipeline skriver valuation.csv og valuation_sensitivity.csv og oppdaterer master_valued.parquet.
Sensitivitetsmotor: loggen viser at sens-fil skrives; dette bør standardiseres til få, robuste sjokk (COE/WACC, terminalvekst, margin) og used som “uncertainty flag” i screening.
### Factor/quality engine
Formål: lage forklarbare faktorer og eventuelt en black-box vekter uten å bryte A-kravet.

Eksisterende faktorgrupper: “Strategy A” opererer med fire grupper (quality, value, lowrisk, balance) med vekter som summerer til 1 og regulariseres mot prior.
Black-box komponent (tillatt av deg): kan brukes til
å lære vekter/rangering innenfor universe (forventet avkastning, sannsynlighet for drawdown etc.),
men må ikke endre intrinsic beregning.
Governance: output må alltid være forklarbar: selv om en ML-modell brukes til scoring, må screen_basic.csv beholde reason-kolonner og vise gating vs scoring.
### Technical risk filter (kun timing/gating)
Formål: sikre at teknisk analyse aldri blir “skjult verdsettelse”.

Dokumentert i WFT-regler: eligibility krever at aksje og indeks er over MA200 og MAD ≥ mad_min, og at indeksdata finnes.
Rolle i beslutning: teknisk modul skal kun gi technical_ok og reason_technical_fail, og aldri skrive til intrinsic_*. Dette matcher din data-kontrakt og artefaktkrav.
### Scenario engine
Formål: gjøre usikkerhet i intrinsic eksplisitt og kunne drive “high_uncertainty” regler.

Minstekrav: skrive “scenario cube” pr ticker: base/bull/bear eller sensitivitetstabell (dette er delvis dekket av valuation_sensitivity.csv).
Bruk i gating: hvis bearish intrinsic impliserer lav MOS → strengere MOS-krav (mos_high_uncertainty) slik decision allerede har terskler for.
### Backtest/WFT/tuning (kvartalsvis rebalansering)
Formål: robust generalisering med lav risiko for backtest-overfitting; du har allerede WFT sweep + nested optimizer-konseptet.

Dagens pipeline gjør WFT sweep/optimizer via tools-skript og skriver runs/ og eksperiment-sammendrag.
Overfitting-risiko: akademisk arbeid påpeker at vanlige holdout-metoder kan være svake i backtest-kontekst og introduserer rammeverk for å anslå sannsynlighet for backtest-overfitting (PBO). Dette støtter at du bør beholde nested/holdout + multi-seed (slik dine interne scripts allerede gjør). 
Kvartalsvis rebalansering: du har valgt dette eksplisitt. Da må calendar.py produsere rebal-datoer pr kvartal og simuleringen må holde posisjon gjennom kvartalet, med handling av utbytte (se tabell under).
### Manual override UI/recording
Formål: tillate kvalitativ justering uten at det blir “magi” eller utestbar logikk.

Hovedregel: override skal være restriktiv og auditert: hva ble endret, hvorfor, av hvem, og når.
Minimalt UI-oppsett: ikke start med full webapp; start med “override manifest” som kan redigeres og lastes inn av pipeline:
config/overrides.yaml (strukturert)
runs/<run_id>/override_audit.json (skrevet av pipeline)
Tillatte override-typer (anbefalt):
Exclude ticker (hard blokk) + tekstlig begrunnelse
Force CASH (portefølje i cash) + begrunnelse
Adjust uncertainty flag (kun strengere, aldri slakkere)
Adjust qualitative score (kun innenfor et begrenset intervall, og aldri direkte endre intrinsic)
Datakontrakt, Børsdata Pro+ “freeze”-liste og dividendebehandling
### Datakontrakt (schema + as-of + quality flags)
Denne kontrakten bygger på det du allerede har skrevet: eksplisitte skjema, ingen stille dropp, yahoo_ticker som kritisk merge-nøkkel, og forklarbare reason-kolonner.

Kjernefelter (minimum) per panelrad i master (ticker × asof):

Identitet: ticker, market, country, currency, yahoo_ticker
Tidsstempler: asof_date, period_end, available_date (for fundamentals), price_date (for pris snapshot)
Fundamentals: nok til å bygge intrinsic (inntekt/EBIT/FCF, balanseposter, aksjeantall osv.) (konkret feltssett er uspesifisert i interne docs)
Valuation: intrinsic_equity, intrinsic_ev (hvis brukt), mos
Quality/value creation: value_creation_ok_base, quality_weak_count, high_risk_flag (minst de som brukt i filterlogikken)
Teknisk: ma200, mad, above_ma200 for aksje og indeks; index_ticker, index_mad, index_above_ma200
Quality flags: missing_price, missing_fundamentals, missing_index, stale_fundamentals, suspect_scale_market_cap, data_warning_count
Forklarbarhet: reason_fundamental_fail, reason_technical_fail (string eller semikolon-separerte tags)
Datasett som bør fryses i løpet av Pro+-vinduet
Börsdata oppgir at API kan laste ned aksjekurser, selskapsdata, rapportdata og nøkkeltall; og at REST-API for egen kode ligger på Pro+ (etter 1 feb 2025). 

Börsdata beskriver også at rapportdata inkluderer kvartals- og årsrapporter, og at TTM brukes for å få 12m grunnlag mellom årsrapportene. 

Freeze-item
Hvorfor du må fryse nå
Minimum innhold
Format/versjonering
Instrument-/universliste
Stabil referanse for backtest + mapping
instrument_id, ticker, market, currency, sektor/branche
parquet + export_date
Daglige priser (aksjer)
Teknisk gating + beta + return
date, close/adj_close, volume, evt. split/dividend markers
parquet, partisjon pr år
Daglige priser (indekser)
Indeksregime (MA200/MAD) + betaberegning
index_symbol, date, close
parquet
Rapportdata kvartal/år
Intrinsic og TTM
period_end, report_period, available_date, regnskapsposter
parquet, “as-reported”
TTM-derivat eller byggbart grunnlag
Korrekte nøkkeltall pr as-of
siste 4 kvartaler pr as-of
enten fryse TTM eller fryse kvartaler slik at TTM kan bygges
Nøkkeltall/KPIer (valgfritt)
Rask feature-bygging; plausibilitetssjekk
ROIC/ROE/FCF-yield osv
parquet
Utbytte og corporate actions
Total return + korrekt aksjeantall/justering
ex-date, amount, currency, type
parquet; egen “actions” tabell
Metadata for reporting-lag
Realistisk as-of
lag/delay eller oppdateringsdato
parquet

(Merk: konkrete endepunkter/felt må matches mot Börsdata faktisk API-respons; denne rapporten beskriver kontrakter og prioritering.) 

### Dividendebehandling: reinvest vs kontant
Du er åpen for begge. Tabellen under viser konsekvenser og hva som må endres i backtest og beslutningslogikk.

S&P Dow Jones Indices beskriver at total return-indekser reflekterer både kursbevegelse og reinvestering av utbytte (dividendeffekten bygges inn i indeksnivået). De beskriver også ulike return-typer (price return vs gross/net total return) og at netto kan hensynta withholding tax. 

Valg
Backtest-beregning
Regnskapsføring i simulator
Konsekvens for decision (kvartalsvis)
Fordeler
Ulemper
Reinvester utbytte
Bruk total return-serie (enten bygd eller konstruert)
Ingen separat kontantstrøm; “equity curve” inkluderer reinvest
Decision påvirkes primært via pris/tr-forskjeller (ikke via cash-balanse)
Enkel, samsvarer med standard total return-definisjon
Skjuler kontantstrøm-mekanikk; skatt/withholding må modelleres separat hvis ønsket
Kontantutbytte
Prisreturn + eksplisitte cash flows
Hold cash_balance og legg til utbytte på ex-date/pay-date
Kan øke kjøpekraft ved neste rebal; påvirker turnover/cash share
Fullt sporbar kontantlogikk
Mer kompleks; krever antall aksjer/posisjon; skatt/FX må modelleres eksplisitt

Uspesifisert (må avklares): om du vil måle “gross” vs “net” (withholding tax), og om utbytte skal bokføres på ex-date eller pay-date. 

Tuning- og valideringsplan, samt diagnose for «always cash»
### Walk-forward schedule (kvartalsvis) og nested optimizer
Du har allerede en struktur for sweep og nested optimizer (med holdout som ikke brukes til tuning/select) og multi-seed verifikasjon.
Du har også en eksplisitt “holdout utility”-formel brukt ved rangering:
utility = excess - 0.50*abs(max_dd) - 0.10*turnover - 0.05*pct_cash.

Forslag til kvartalsvis WFT-kontrakt (uten å øke frihetsgrad ukontrollert):

Rebalance: hvert kvartal (Q1/Q2/Q3/Q4) – posisjon holdes mellom datoene. (Preferanse fra deg; ikke i interne scripts per nå.)
Outer folds: rullende 1-år testperiode (4 kvartaler) over 2006–2025, med 10–12 års treningsvindu (samme størrelsesorden som dine eksisterende sweeps).
Nested tuning: for hver outer train-periode, kjør optimizer på underfolds eller på en intern WFT på trening, men hold holdout helt separat (som du allerede gjør).
Multi-seed: minst 5 seeds og “stability rule” som i strategyA_recommendation (krav om stabil retning i flere seeds).
Parameterrammer og “low-degree knobs”
Du har eksplisitt dokumentert hvilke parametre som faktisk er tunable i WFT i dag, og hvilke terskler som ligger i decision-config. Dette bør respekteres først (for å unngå overtilpasning), og deretter kan du ekspandere gradvis.

Anbefalt start (som matcher eksisterende implementasjon):

mos_threshold: 0.30–0.45 (grid)
mad_min: -0.05–0.02 (grid)
weakness_rule_variant: baseline eller stricter (merk: “strict” er ikke implementert).
Faktorvekter (quality/value/lowrisk/balance) + regularisering (weights_reg_lambda) som “Strategy A knobs” beskriver.
Hysterese og MAD-penalty (min_hold_months/score_gap, mad_penalty_k) som i Strategy A knobs (dersom de finnes i din kodebase).
Ablation-testplan (må kjøres før du “stoler” på en black-box):

Fundamentals-only (MOS + value creation + enkel kvalitet)
Technical gating
Hysterese
Black-box vekting (kun ranking, ikke intrinsic)
Ablation er nødvendig fordi du eksplisitt sier at variablers prediktive betydning er uklar.
Diagnoseprotokoll: «always cash»
Dette er en fail-fast sjekkliste du kan kjøre før videre tuning. Den er direkte avledet av WFT-eligibility-reglene og log/artefaktkrav i prosjektet.

Symptomverifikasjon (faktabasert):

Holdout_pct_cash=1.0 i seed-oppsummering → bekrefter “always cash” i holdout.
strategyA_recommendation viser alle seeds flate (ingen improve/degrade) og beslutning “tighten regularization/penalties” – men dette er misvisende hvis modellen aldri investerer (da er alt 0).
Root cause-løype (i prioritet):

Pris-/mapping-dekning: Hvis yahoo_ticker-merge feiler vil prisdekning falle; du har et log-eksempel med “price coverage adj_close=0.041”, som er ekstremt lavt og vil gjøre MA200/MAD ubrukelig.
Indeksdata mangler: eligibility krever indeksdata + MA200/MAD for indeks, altså kan manglende indeksserie blokkere alt.
For strenge gates: mos_req kan bli maks(0.40, mos_threshold) for high_risk_flag; i små markeder kan dette alene skyve alt til CASH.
Skaleringsfeil i market cap: decision logger at market_cap ble skalert med 1e6 fordi input «så ut som millioner». Hvis dette påvirker EV/FCF-yield/andre filters indirekte kan det trigge feilaktige value/quality-fail.
“Kan investere”-tester (må implementeres som automatiserte tester):

Test 1 (data): med tekniske krav av, MOS=0 og quality gates åpne → systemet må velge minst én kandidat (ellers er data/merge brutt).
Test 2 (teknisk isolert): med fundamentals gates av → teknisk modul må produsere technical_ok=True for minst X% av tickere når prisdekning er god.
Test 3 (funnel logging): for hver rebal-dato, output antall tickere som faller på hver gate (MOS, value_creation, quality_weak, stock_ma200, stock_mad, index_present, index_ma200, index_mad). Dette speiler eligibility-listen i TUNING_PARAMS.
Test 4 (holdout sanity): bruk en kjent "good" config fra `experiments/strategyA_seed_summary.md`, og kj?r samme periode/holdout som seed summary. Hvis resultat fortsatt blir 100% cash, er det mismatch mellom WFT og holdout-simulator eller data "as-of" i holdout.
Implementeringsplan med Codex, milepæler og åpne spørsmål
### Codex-implementering: arbeidsform og kvalitetssikring
Din dokumentasjon forventer pytest som standard og Windows/PowerShell som runtime. Dette bør brukes som ryggrad i Codex-loop: specs → kontrakt-tester → implementasjon → smoke test → artefaktverifisering.

Anbefalt “API-first” kontrakt mellom moduler (for Codex):

Hver modul eksponerer én funksjon med streng input/output schema (f.eks. build_master(asof, freeze_dir, config) -> master.parquet + quality.md).
Hver funksjon har:
schema-validering i starten (fail-fast)
deterministisk output-sti under runs/<run_id>/
enhetstest + en “golden” integrasjonstest (liten sample)
Milepæler for 3-ukers Børsdata Pro+ vindu (2026-02-21 til ca. 2026-03-14)
(Datoer er konkrete gitt dagens dato; justér hvis Pro+-tilgang utløper tidligere/senere.)

Freeze-alt-uttak (Eirik, frist 2026-02-26, formål: sikre datagrunnmur)

Bygg export_freeze.py som laster ned og lagrer instrumentliste, priser (aksje+indeks), rapportdata og corporate actions/utbytte dersom tilgjengelig. 
Datakontrakt + schema validators (Eirik + Codex, frist 2026-03-01, formål: eliminere stille feil)

Implementer schemas/*.json og validate_schema.py, og krev at missing_price/reason etableres eksplisitt (matcher intern datakontrakt).
Kvartalsvis backtest-simulator (Codex, frist 2026-03-05, formål: align med preferanse)

Bygg calendar.py (quarter schedule) og simulator.py for single-stock-or-cash.
Always-cash fix gate (Eirik, frist 2026-03-08, formål: gjøre tuning meningsfull)

Implementer “kan investere”-tester + funnel-logging. Bruk price coverage-signalene fra log som startpunkt.
WFT sweep/optimizer på kvartalskalender (Codex + Eirik review, frist 2026-03-13, formål: første robuste config)

Repliker intern logikk (MOS/MAD/weakness variants) og produser sweep_summary.md/seed summary som er sammenlignbar med eksisterende.
Roadmap 6–12 uker
Stabiliser valuation engine (DCF/DDM/RIM) med standardisert valuation.csv + valuation_sensitivity.csv og eksplisitt “uncertainty flag”. 
Implementer dividend policy som konfigvalg (reinvest vs cash) og valider mot total return-definisjoner. 
Bygg black-box faktorvekting med governance (audit og begrensninger), og kjør ablations.
Bygg minimal manual override UI (f.eks. CLI/Streamlit) som skriver override_audit.json og aldri lar overrides endre intrinsic direkte.
Sett opp CI: pytest, linting, og smoke test-kommando som i README.
Åpne spørsmål og “uspesifisert”
Dette må avklares i Deecon-spesifikasjonen før endelig implementering (uten at jeg antar):

MTP fit: hva betyr “MTP” hos deg (definisjon + målvariabel + akseptkriterie)? (Uspesifisert.)
Universe: du har tidligere defaults --universe NO og “NORDIC kan brukes”, men ditt ønskede universe nå er ikke eksplisitt skrevet her.
Dividend policy: reinvest vs cash, samt gross vs net (withholding tax) og ex-date vs pay-date. 
RIM-design: nøyaktig residual income-formulering og hvilke bankspesifikke inputfelt som er obligatoriske. (Uspesifisert i interne dokumenter.)
Manual override governance: hvem kan override, hvilke typer, og om override kan gjøre regler slakkere (anbefalt: kun strengere).
Kostmodell: sweep_summary viser “placeholder” kostmodell (bps per turnover); du må definere ønsket realismegrad for kvartalsvis simulator.
