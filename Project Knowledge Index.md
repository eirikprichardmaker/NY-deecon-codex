# Project Knowledge Index — DEECON (Claude Projects)

## Formål
Dette prosjektet (DEECON) er en deterministisk investeringspipeline (kjøres med `--asof`) som velger maks én nordisk aksje eller CASH.
Kjerneprinsipp:
- Intrinsic value (verdsettelse) skal baseres utelukkende på fundamentals (ikke pris/tekniske felt).
- Teknisk analyse (MA200/MAD + indeksregime) brukes kun som timing-/risikogating.
- Beslutninger og gates skal være forklarbare (reasons/audit), og tester fungerer som kontrakt.

## Hvordan Claude skal bruke prosjektfilene
Når du svarer:
1) Bruk alltid testfiler i `tests/` som spesifikasjon (“kontrakt først”).
2) Bruk `src/run_weekly.py` for å forstå stegrekkefølge og hvilke moduler som skal finnes (run(ctx, log)).
3) For valuation: bruk `src/valuation.py` + `tests/test_valuation_input_audit.py` for expected outputs, reasons og auditkrav.
4) For DQ/decision/TA-gating: bruk `src/decision.py` + `tests/test_decision_data_quality.py`.
5) Når spørsmål gjelder schema/field mapping: bruk `docs/CANONICAL_SCHEMA.md` og senere `config/canonical_schema.yaml`.
6) Når spørsmål gjelder ESEF: bruk `config/mappings/esef_ifrs_default.yaml` + fixtures i `tests/fixtures/` (sample_esef.xhtml, mock_arelle_extracted_facts.json).

## Anbefalt “core files” (lastes først)
- Komplett systemarkitektur for Deecon-prosjektet.txt
  Rolle: overordnet arkitektur, invariants og komponentkart.
- src/run_weekly.py
  Rolle: pipeline orchestrator (default steps + modulkontrakt run(ctx, log)).
- src/valuation.py
  Rolle: DCF-verdsettelse (fundamental-only guard), quarterly R12-bridge, valuation.csv + sensitivity + audit json.
- src/decision.py
  Rolle: decision engine + gates + data quality checks + reasons.
- config/config.yaml
  Rolle: sentral config (paths + toggles + thresholds).
- tests/test_decision_data_quality.py
  Rolle: kontrakt for DQ (fail vs warn; blocked; outlier policy).
- tests/test_valuation_input_audit.py
  Rolle: kontrakt for valuation audit + quarterly prefer/require.

## “Add later” (når relevant)
- docs/CANONICAL_SCHEMA.md + config/canonical_schema.yaml + config/computed_metrics.yaml
  Når: feilsøking av KPI/field drift, mapping, schema endringer.
- config/mappings/esef_ifrs_default.yaml + fixtures (sample_esef.xhtml, mock_arelle_extracted_facts.json)
  Når: ESEF-integrasjon og XBRL/iXBRL-fact extraction.
- TUNING_PARAMS.md + experiments/*
  Når: WFT-tuning, ablation og “always cash”-diagnose.
- manifest_*.csv
  Når: behov for robust traceability og data coverage rapportering.

## “Do not upload”
- .env (sensitivt)
- exports/*/*.xlsx (output-artefakter; binært og støy for retrieval)
- store universlister (tickers*.csv) uten å først lage liten golden sample.
