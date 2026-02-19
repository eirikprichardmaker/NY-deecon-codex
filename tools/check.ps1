$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt
pytest -q

python -m src.run_weekly --asof 2026-02-16 --config .\config\config.yaml --steps valuation,decision

Write-Host "OK: tests + smoke run completed"
