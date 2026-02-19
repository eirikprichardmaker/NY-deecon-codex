# Migration status – 2026-02-19

## Utført i denne sesjonen
1. Verifiserte tilgjengelige mapper i workspace.
2. Fant kun `../NY-deecon-codex` og ingen `../Borsdata`.
3. Strammet inn `.gitignore` for å unngå binære/tunge filer i commits.
4. Oppdaterte README med smoke test-kommando og importmanifest.

## Blokkering
- Kilderepoet (`Borsdata`) finnes ikke i dette miljøet, så det var ikke mulig å:
  - opprette branch `refactor/archive-cleanup`
  - flytte ikke-nødvendige filer til `archive/2026-02-19/...`
  - kopiere nødvendige prosjektfiler inn i target-repo
  - kjøre ønsket smoke test (mangler `src.run_weekly`)

## Neste steg når begge repoer er tilgjengelige
- Kjør arkivering i Børsdata med `git mv` for trackede filer.
- Importer bare nødvendige mapper/filer til target-repo.
- Kjør smoke test i target og fiks eventuelle sti/konfig-minimum.
