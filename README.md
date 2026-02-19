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
