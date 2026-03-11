"""
Output contract: ensures every run produces the same set of files.
If any required file is missing, the run is flagged as incomplete.
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
    "decision_agent.md",       # Kun hvis dossier writer er aktivert
    "decision_agent.json",
    "skeptic_results.json",    # Kun hvis skeptiker er aktivert
    "quality_results.json",    # Kun hvis quality evaluator er aktivert
]


def validate_run_outputs(run_dir: Path) -> tuple[bool, list[str]]:
    """
    Sjekk at alle påkrevde output-filer finnes.
    Returnerer (ok, manglende_filer).
    """
    missing = [fname for fname in REQUIRED_OUTPUTS if not (run_dir / fname).exists()]
    return len(missing) == 0, missing
