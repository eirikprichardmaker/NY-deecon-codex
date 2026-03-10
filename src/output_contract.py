"""
Output contract: ensures every run produces the same set of files.
If any file is missing, the run is flagged as incomplete.
"""
from pathlib import Path

REQUIRED_OUTPUTS = [
    "manifest.json",
    "data_quality_audit.csv",
    "data_quality_audit_v2.csv",
    "screen_basic.csv",
    "valuation.csv",
    "valuation_sensitivity.csv",
    "decision.csv",
    "decision.md",
    "decision_reasons.json",
    "value_qc_summary.csv",
]

OPTIONAL_OUTPUTS = [
    "decision_agent.md",        # Only if dossier writer enabled
    "decision_agent.json",
    "skeptic_results.json",     # Only if skeptic enabled
    "quality_results.json",     # Only if quality evaluator enabled
]


def validate_run_outputs(run_dir: Path) -> tuple[bool, list[str]]:
    """
    Check all required outputs exist.
    Returns (ok, missing_files).
    """
    missing = [fname for fname in REQUIRED_OUTPUTS if not (run_dir / fname).exists()]
    return len(missing) == 0, missing


def list_optional_outputs(run_dir: Path) -> list[str]:
    """Returns which optional output files are present."""
    return [fname for fname in OPTIONAL_OUTPUTS if (run_dir / fname).exists()]
