"""
Data Quality rules with unique IDs.
Each rule has: rule_id, severity (FAIL/WARN/PASS), field(s), condition, sector_override.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd


class Severity(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class DQRule:
    rule_id: str
    description: str
    severity: Severity
    fields: list[str]
    check_fn: Callable[[dict], bool]  # True = passes
    sector_overrides: dict = field(default_factory=dict)  # sector -> override severity

    def __post_init__(self) -> None:
        # Ensure check_fn always returns a plain Python bool (not np.bool_)
        _fn = self.check_fn
        self.check_fn = lambda row: bool(_fn(row))


# --- FAIL rules (hard stop, force CASH) ---
RULES: list[DQRule] = [
    DQRule(
        rule_id="DQ-F001",
        description="Price must be positive",
        severity=Severity.FAIL,
        fields=["adj_close"],
        check_fn=lambda row: (
            np.isfinite(row.get("adj_close", np.nan))
            and row.get("adj_close", 0) > 0
        ),
    ),
    DQRule(
        rule_id="DQ-F002",
        description="Market cap must be positive",
        severity=Severity.FAIL,
        fields=["market_cap"],
        check_fn=lambda row: (
            np.isfinite(row.get("market_cap", np.nan))
            and row.get("market_cap", 0) > 0
        ),
    ),
    DQRule(
        rule_id="DQ-F003",
        description="EV must be positive (non-bank)",
        severity=Severity.FAIL,
        fields=["enterprise_value"],
        check_fn=lambda row: (
            str(row.get("sector_group", "")).lower() == "bank"
            or (
                np.isfinite(row.get("enterprise_value", np.nan))
                and row.get("enterprise_value", 0) > 0
            )
        ),
        sector_overrides={"bank": Severity.PASS},
    ),
    DQRule(
        rule_id="DQ-F004",
        description="MA200 must exist for technical gating",
        severity=Severity.FAIL,
        fields=["ma200"],
        check_fn=lambda row: np.isfinite(row.get("ma200", np.nan)),
    ),
    DQRule(
        rule_id="DQ-F005",
        description="Index MA200 must exist",
        severity=Severity.FAIL,
        fields=["index_ma200"],
        check_fn=lambda row: np.isfinite(row.get("index_ma200", np.nan)),
    ),
    DQRule(
        rule_id="DQ-F006",
        description="At least one valuation anchor required (intrinsic or multiples)",
        severity=Severity.FAIL,
        fields=["intrinsic_value", "ev_ebit"],
        check_fn=lambda row: (
            np.isfinite(row.get("intrinsic_value", np.nan))
            or np.isfinite(row.get("ev_ebit", np.nan))
        ),
    ),
    # --- WARN rules (flag, but continue with caution) ---
    DQRule(
        rule_id="DQ-W001",
        description="ND/EBITDA extreme (>10 or <-5)",
        severity=Severity.WARN,
        fields=["nd_ebitda"],
        check_fn=lambda row: (
            not np.isfinite(row.get("nd_ebitda", np.nan))
            or (-5 <= row.get("nd_ebitda", 0) <= 10)
        ),
        sector_overrides={"aquaculture": Severity.PASS},
    ),
    DQRule(
        rule_id="DQ-W002",
        description="FCF yield extreme (<-20% or >25%)",
        severity=Severity.WARN,
        fields=["fcf_yield"],
        check_fn=lambda row: (
            not np.isfinite(row.get("fcf_yield", np.nan))
            or (-0.20 <= row.get("fcf_yield", 0) <= 0.25)
        ),
    ),
    DQRule(
        rule_id="DQ-W003",
        description="ROIC extreme (<-20% or >60%)",
        severity=Severity.WARN,
        fields=["roic_current"],
        check_fn=lambda row: (
            not np.isfinite(row.get("roic_current", np.nan))
            or (-0.20 <= row.get("roic_current", 0) <= 0.60)
        ),
    ),
    DQRule(
        rule_id="DQ-W004",
        description="Shares outstanding must be positive",
        severity=Severity.WARN,
        fields=["shares_outstanding"],
        check_fn=lambda row: (
            not np.isfinite(row.get("shares_outstanding", np.nan))
            or row.get("shares_outstanding", 0) > 0
        ),
    ),
]


def run_dq_rules(
    df: pd.DataFrame,
    rules: Optional[list[DQRule]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all DQ rules on dataframe.

    Returns:
        flags_df: per-ticker columns dq_fail_v2, dq_fail_count_v2,
                  dq_warn_count_v2, dq_fail_rules, dq_warn_rules
        audit_df: per-ticker, per-rule audit log
    """
    if rules is None:
        rules = RULES

    audit_rows = []
    fail_counts = []
    warn_counts = []
    fail_rules_list = []
    warn_rules_list = []

    for _, row in df.iterrows():
        ticker = row.get("ticker", "UNKNOWN")
        sector = str(row.get("sector_group", "")).lower()
        f_count, w_count = 0, 0
        f_rules, w_rules = [], []

        for rule in rules:
            effective_severity = rule.sector_overrides.get(sector, rule.severity)
            passes = bool(rule.check_fn(dict(row)))
            audit_rows.append({
                "ticker": ticker,
                "rule_id": rule.rule_id,
                "severity": effective_severity.value,
                "fields": ",".join(rule.fields),
                "passed": passes,
                "description": rule.description,
            })
            if not passes:
                if effective_severity == Severity.FAIL:
                    f_count += 1
                    f_rules.append(rule.rule_id)
                elif effective_severity == Severity.WARN:
                    w_count += 1
                    w_rules.append(rule.rule_id)

        fail_counts.append(f_count)
        warn_counts.append(w_count)
        fail_rules_list.append(";".join(f_rules))
        warn_rules_list.append(";".join(w_rules))

    flags_df = pd.DataFrame(
        {
            "dq_fail_v2": [c > 0 for c in fail_counts],
            "dq_fail_count_v2": fail_counts,
            "dq_warn_count_v2": warn_counts,
            "dq_fail_rules": fail_rules_list,
            "dq_warn_rules": warn_rules_list,
        },
        index=df.index,
    )
    audit_df = pd.DataFrame(audit_rows)
    return flags_df, audit_df
