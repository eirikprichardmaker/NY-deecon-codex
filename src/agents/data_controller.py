"""
Data Controller Gate: deterministisk port som kjøres FØR alle LLM-agenter.

Garanterer:
  - Ingen ticker med manglende kritiske felter sendes til LLM
  - Ingen ticker med ugyldige/umulige verdier sendes til LLM
  - Maskinlesbar diagnose for hvert avslag (fikser "always cash"-diagnostikk)

Bruk i agents_step.py:
    result = check(ticker_row, valuation_row)
    if not result.ok:
        log blocked, skip agent chain
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Kritiske felter som MÅ finnes for at agentene skal få noe meningsfylt å jobbe med
# ---------------------------------------------------------------------------

# Felter som kreves for Investment Skeptic (Agent B)
_SKEPTIC_REQUIRED = [
    "adj_close",
    "market_cap",
    "intrinsic_value",
    "wacc_used",
    "mos",
]

# Felter som kreves for at Dossier Writer (Agent C) skal produsere en meningsfull rapport
_DOSSIER_REQUIRED = [
    "adj_close",
    "intrinsic_value",
    "mos",
]

# Felter der verdien i tillegg til å eksistere IKKE kan bryte harde grenser
_RANGE_CHECKS: list[tuple[str, float, float]] = [
    # (felt, min_eksklusiv, maks_eksklusiv)
    ("adj_close",       0.0,    1e7),
    ("market_cap",      0.0,    1e15),
    ("wacc_used",       0.0,    0.60),
    ("mos",            -2.0,    50.0),
    ("intrinsic_value", -1e13,  1e14),
    ("roic_current",   -5.0,    10.0),
    ("fcf_yield",      -2.0,    2.0),
    ("nd_ebitda",      -30.0,   100.0),
]


# ---------------------------------------------------------------------------
# Resultattype
# ---------------------------------------------------------------------------

@dataclass
class ControllerResult:
    ticker: str
    ok: bool
    blocked_reason: str = ""          # Kortfattet årsak (for logging)
    missing_fields: list[str] = field(default_factory=list)
    out_of_range_fields: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "ok": self.ok,
            "blocked_reason": self.blocked_reason,
            "missing_fields": self.missing_fields,
            "out_of_range_fields": self.out_of_range_fields,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Hoved-gate-funksjon
# ---------------------------------------------------------------------------

def check(
    ticker_row: pd.Series | dict,
    valuation_row: pd.Series | dict | None = None,
    required_fields: list[str] | None = None,
) -> ControllerResult:
    """
    Kjør alle data-kontroller for én ticker FØR agentene kalles.

    Args:
        ticker_row:     Rad fra decision.csv (eller master-df)
        valuation_row:  Rad fra valuation.csv (eller None)
        required_fields: Override av påkrevde felter (standard = _SKEPTIC_REQUIRED)

    Returns:
        ControllerResult med ok=True (kan sende til agenter) eller ok=False (blokkert)
    """
    if required_fields is None:
        required_fields = _SKEPTIC_REQUIRED

    # Slå sammen ticker_row + valuation_row til én oppslagsdict
    merged = _merge(ticker_row, valuation_row)
    ticker = str(merged.get("ticker", "UNKNOWN"))

    missing: list[str] = []
    out_of_range: list[str] = []
    diag: dict[str, Any] = {}

    # --- 1. Manglende påkrevde felter ---
    for field_name in required_fields:
        val = merged.get(field_name)
        if _is_missing(val):
            missing.append(field_name)
            diag[field_name] = {"status": "missing", "value": None}

    # --- 2. Harde grenser for felter som finnes ---
    for field_name, lo, hi in _RANGE_CHECKS:
        val = merged.get(field_name)
        if _is_missing(val):
            continue  # Allerede fanget opp i punkt 1 hvis påkrevd
        try:
            v = float(val)
        except (TypeError, ValueError):
            out_of_range.append(field_name)
            diag[field_name] = {"status": "not_numeric", "value": str(val)}
            continue
        if not (lo < v < hi):
            out_of_range.append(field_name)
            diag[field_name] = {"status": "out_of_range", "value": v, "allowed": f"({lo}, {hi})"}

    # --- 3. Bygg resultat ---
    if missing or out_of_range:
        parts = []
        if missing:
            parts.append(f"manglende_felt=[{','.join(missing)}]")
        if out_of_range:
            parts.append(f"utenfor_grenser=[{','.join(out_of_range)}]")
        return ControllerResult(
            ticker=ticker,
            ok=False,
            blocked_reason="; ".join(parts),
            missing_fields=missing,
            out_of_range_fields=out_of_range,
            diagnostics=diag,
        )

    return ControllerResult(ticker=ticker, ok=True, diagnostics=diag)


def check_shortlist(
    shortlist_df: pd.DataFrame,
    valuation_df: pd.DataFrame | None,
    required_fields: list[str] | None = None,
) -> tuple[pd.DataFrame, list[ControllerResult]]:
    """
    Kjør data-controller på alle rader i shortlist.

    Returns:
        (approved_df, all_results)
        approved_df:  Kun rader som passerte (ok=True)
        all_results:  ControllerResult per ticker (for diagnose og logging)
    """
    results: list[ControllerResult] = []
    approved_indices: list[int] = []

    for i, (_, row) in enumerate(shortlist_df.iterrows()):
        ticker = str(row.get("ticker", "UNKNOWN"))
        val_row = None
        if valuation_df is not None and not valuation_df.empty:
            matches = valuation_df[valuation_df["ticker"].astype(str) == ticker]
            if not matches.empty:
                val_row = matches.iloc[0]

        result = check(row, val_row, required_fields=required_fields)
        results.append(result)
        if result.ok:
            approved_indices.append(i)

    approved_df = shortlist_df.iloc[approved_indices].copy() if approved_indices else shortlist_df.iloc[0:0].copy()
    return approved_df, results


# ---------------------------------------------------------------------------
# Hjelpefunksjoner
# ---------------------------------------------------------------------------

def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return True
    if isinstance(val, str) and val.strip() in ("", "nan", "NaN", "None", "null"):
        return True
    try:
        import numpy as np
        if isinstance(val, (np.floating, np.integer)) and not np.isfinite(float(val)):
            return True
    except ImportError:
        pass
    return False


def _merge(
    ticker_row: pd.Series | dict,
    valuation_row: pd.Series | dict | None,
) -> dict:
    """Slå sammen ticker_row og valuation_row til én flat dict. ticker_row har forrang."""
    base: dict = {}
    if valuation_row is not None:
        if isinstance(valuation_row, pd.Series):
            base.update(valuation_row.to_dict())
        else:
            base.update(dict(valuation_row))
    if isinstance(ticker_row, pd.Series):
        base.update(ticker_row.to_dict())
    else:
        base.update(dict(ticker_row))
    return base
