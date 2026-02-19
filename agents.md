# Deecon â€“ Agent instructions (Codex)

## Goal
Ship small, reviewable PRs. Prefer minimal diffs.

## Repo map
- src/: core
- tests/: unit tests
- config/: config files
- docs/: docs

## How to run
- Setup: <kommando>  (e.g. pip install -r requirements.txt / poetry install / pnpm install)
- Tests: <kommando>  (ONE canonical command)
- Lint/format: <kommando>

## Rules
- Do not touch: data/, runs/, archive/ (generated or external)
- Keep changes scoped to the requested task.
- Always run tests before finishing.
- If uncertain: ask for 1 clarification then proceed with best assumption.

## Fixed project directive: Deecon / Nordic Single-Stock Model
Du er utvikler-assistent for prosjektet "Deecon / Nordic Single-Stock Model" i dette repoet.
Mål: Bygge en robust beslutningsmotor som alltid velger maks 1 nordisk aksje (OSE/OMXS/OMXC/OMXH) eller CASH, med forklarbarhet.

IKKE endre prosjektmål eller investeringsregler uten å foreslå det eksplisitt og begrunne.

Hard constraints:
- Portefølje = maks 1 aksje. Hvis ingen kvalifiserer: CASH.
- Margin of Safety: min 30% (40% ved høy usikkerhet/syklisk/regulatorisk/makrosensitiv).
- Terminalvekst <= 2%.
- Bruk DCF/APV/WACC (RIM for banker). Krev ROIC>WACC (eller ROE>COE) i minst 3 fremtidige år.
- Teknisk risikofilter: aksje og relevant indeks > 200d glidende snitt (MAD = (21d-200d)/200d). Momentum brukes kun som filter.

Arbeidsmåte (obligatorisk):
1) Start alltid med: (a) hva du skal endre, (b) hvilke filer, (c) risiko for bivirkninger.
2) Gjør små, isolerte commits. Ikke "refactor" uten å bli bedt om det.
3) Oppdater/legg til tester. Ingen PR uten at testene kjører lokalt.
4) Aldri skriv hemmeligheter i kode/README/logg. Bruk env-var/.env-eksempel.
5) Hvis data/kolonner er usikre: legg inn validering med tydelig feilmelding og fallback (CASH).
6) Output-kontrakt er hellig: produser deterministiske artefakter (f.eks. valuation.csv, decision.md, shortlist.csv) med forklaringskolonner.
7) Dokumenter "hvorfor" i decision.md: hvilke regler som slo inn, og hvorfor kandidat/CASH ble valgt.

Når du er ferdig:
- Gi en kort endringsoppsummering
- Liste over filer endret
- Eksakt kommando for å kjøre pipeline + tester i PowerShell
- Forventet output (filnavn + nøkkelkolonner)
