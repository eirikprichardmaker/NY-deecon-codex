# AGENTS.md — NY-deecon-codex (NEW DEECON)

Goal
- Maintain a deterministic as-of pipeline that produces auditable artifacts in runs/<run_id>/.
- Never silently drop rows. Every exclusion must have a reason_* field.

Non-negotiables
- Decision output: pick at most ONE stock, else CASH.
- Intrinsic value MUST be computed from fundamentals + risk inputs only (no price as valuation input).
- Technical analysis (MA200/MAD + index regime) is gating/timing only, never affects intrinsic_*.
- Join key for merges must be yahoo_ticker. Enforce one-to-one joins; fail fast on duplicates.

How to run
- Install: pip install -r requirements.txt
- Unit tests: pytest -q
- Smoke test:
  python -m src.run_weekly --asof 2026-02-16 --config ./config/config.yaml --steps valuation,decision

Artifacts to verify (after smoke)
- runs/<run_id>/log.txt
- runs/<run_id>/manifest.json
- runs/<run_id>/valuation.csv and valuation_sensitivity.csv
- runs/<run_id>/decision.csv and decision.md

Rules
- Do not add or modify any file under data/, runs/, logs/ in git (they are ignored).
- Never commit secrets (API keys). Use environment variables only.
