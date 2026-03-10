"""
Tests for src/data_quality/rules.py
Minst 1 positiv + 1 negativ test per FAIL-regel, pluss WARN-regler og run_dq_rules().
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math

import numpy as np
import pandas as pd
import pytest

from src.data_quality.rules import RULES, DQRule, Severity, run_dq_rules


# ---------------------------------------------------------------------------
# Hjelpefunksjoner
# ---------------------------------------------------------------------------

def _get_rule(rule_id: str) -> DQRule:
    rule = next((r for r in RULES if r.rule_id == rule_id), None)
    assert rule is not None, f"Regel {rule_id} ikke funnet"
    return rule


def _base_row(**overrides) -> dict:
    """Minimum gyldig rad som passerer alle FAIL-regler."""
    base = {
        "ticker": "EQNR",
        "adj_close": 280.0,
        "market_cap": 500_000.0,
        "enterprise_value": 600_000.0,
        "ma200": 260.0,
        "index_ma200": 1200.0,
        "intrinsic_value": 320.0,
        "sector_group": "energy",
    }
    base.update(overrides)
    return base


def _df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------------------
# DQ-F001: Price must be positive
# ---------------------------------------------------------------------------

class TestDQF001:
    rule = _get_rule("DQ-F001")

    def test_positive_price_passes(self):
        assert self.rule.check_fn({"adj_close": 100.0}) is True

    def test_negative_price_fails(self):
        assert self.rule.check_fn({"adj_close": -5.0}) is False

    def test_zero_price_fails(self):
        assert self.rule.check_fn({"adj_close": 0.0}) is False

    def test_nan_price_fails(self):
        assert self.rule.check_fn({"adj_close": float("nan")}) is False

    def test_missing_key_fails(self):
        assert self.rule.check_fn({}) is False

    def test_severity_is_fail(self):
        assert self.rule.severity == Severity.FAIL


# ---------------------------------------------------------------------------
# DQ-F002: Market cap must be positive
# ---------------------------------------------------------------------------

class TestDQF002:
    rule = _get_rule("DQ-F002")

    def test_positive_market_cap_passes(self):
        assert self.rule.check_fn({"market_cap": 1_000_000.0}) is True

    def test_negative_market_cap_fails(self):
        assert self.rule.check_fn({"market_cap": -100.0}) is False

    def test_zero_market_cap_fails(self):
        assert self.rule.check_fn({"market_cap": 0.0}) is False

    def test_nan_market_cap_fails(self):
        assert self.rule.check_fn({"market_cap": float("nan")}) is False

    def test_missing_key_fails(self):
        assert self.rule.check_fn({}) is False

    def test_severity_is_fail(self):
        assert self.rule.severity == Severity.FAIL


# ---------------------------------------------------------------------------
# DQ-F003: EV must be positive (non-bank)
# ---------------------------------------------------------------------------

class TestDQF003:
    rule = _get_rule("DQ-F003")

    def test_positive_ev_passes(self):
        assert self.rule.check_fn({"enterprise_value": 500_000.0, "sector_group": "energy"}) is True

    def test_negative_ev_fails_non_bank(self):
        assert self.rule.check_fn({"enterprise_value": -100.0, "sector_group": "industrial"}) is False

    def test_negative_ev_passes_for_bank(self):
        """Banker er unntatt EV-kravet."""
        assert self.rule.check_fn({"enterprise_value": -100.0, "sector_group": "bank"}) is True

    def test_nan_ev_fails_non_bank(self):
        assert self.rule.check_fn({"enterprise_value": float("nan"), "sector_group": "energy"}) is False

    def test_zero_ev_fails(self):
        assert self.rule.check_fn({"enterprise_value": 0.0, "sector_group": "energy"}) is False

    def test_severity_is_fail(self):
        assert self.rule.severity == Severity.FAIL

    def test_bank_sector_override_is_pass(self):
        assert self.rule.sector_overrides.get("bank") == Severity.PASS


# ---------------------------------------------------------------------------
# DQ-F004: MA200 must exist
# ---------------------------------------------------------------------------

class TestDQF004:
    rule = _get_rule("DQ-F004")

    def test_valid_ma200_passes(self):
        assert self.rule.check_fn({"ma200": 260.0}) is True

    def test_nan_ma200_fails(self):
        assert self.rule.check_fn({"ma200": float("nan")}) is False

    def test_missing_ma200_fails(self):
        assert self.rule.check_fn({}) is False

    def test_zero_ma200_passes(self):
        """0.0 er et gyldig tall (isfinite)."""
        assert self.rule.check_fn({"ma200": 0.0}) is True

    def test_severity_is_fail(self):
        assert self.rule.severity == Severity.FAIL


# ---------------------------------------------------------------------------
# DQ-F005: Index MA200 must exist
# ---------------------------------------------------------------------------

class TestDQF005:
    rule = _get_rule("DQ-F005")

    def test_valid_index_ma200_passes(self):
        assert self.rule.check_fn({"index_ma200": 1200.0}) is True

    def test_nan_index_ma200_fails(self):
        assert self.rule.check_fn({"index_ma200": float("nan")}) is False

    def test_missing_index_ma200_fails(self):
        assert self.rule.check_fn({}) is False

    def test_severity_is_fail(self):
        assert self.rule.severity == Severity.FAIL


# ---------------------------------------------------------------------------
# DQ-F006: At least one valuation anchor required
# ---------------------------------------------------------------------------

class TestDQF006:
    rule = _get_rule("DQ-F006")

    def test_intrinsic_value_alone_passes(self):
        assert self.rule.check_fn({"intrinsic_value": 300.0, "ev_ebit": float("nan")}) is True

    def test_ev_ebit_alone_passes(self):
        assert self.rule.check_fn({"intrinsic_value": float("nan"), "ev_ebit": 15.0}) is True

    def test_both_anchors_passes(self):
        assert self.rule.check_fn({"intrinsic_value": 300.0, "ev_ebit": 15.0}) is True

    def test_no_anchor_fails(self):
        assert self.rule.check_fn({"intrinsic_value": float("nan"), "ev_ebit": float("nan")}) is False

    def test_missing_both_keys_fails(self):
        assert self.rule.check_fn({}) is False

    def test_severity_is_fail(self):
        assert self.rule.severity == Severity.FAIL


# ---------------------------------------------------------------------------
# WARN-regler (spot-tester)
# ---------------------------------------------------------------------------

class TestDQW001:
    rule = _get_rule("DQ-W001")

    def test_normal_nd_ebitda_passes(self):
        assert self.rule.check_fn({"nd_ebitda": 3.0}) is True

    def test_extreme_high_nd_ebitda_warns(self):
        assert self.rule.check_fn({"nd_ebitda": 11.0}) is False

    def test_extreme_low_nd_ebitda_warns(self):
        assert self.rule.check_fn({"nd_ebitda": -6.0}) is False

    def test_nan_nd_ebitda_passes(self):
        """NaN betyr manglende data — ikke flagges som WARN."""
        assert self.rule.check_fn({"nd_ebitda": float("nan")}) is True

    def test_aquaculture_sector_override(self):
        assert self.rule.sector_overrides.get("aquaculture") == Severity.PASS

    def test_severity_is_warn(self):
        assert self.rule.severity == Severity.WARN


class TestDQW002:
    rule = _get_rule("DQ-W002")

    def test_normal_fcf_yield_passes(self):
        assert self.rule.check_fn({"fcf_yield": 0.05}) is True

    def test_extreme_high_fcf_yield_warns(self):
        assert self.rule.check_fn({"fcf_yield": 0.30}) is False

    def test_extreme_low_fcf_yield_warns(self):
        assert self.rule.check_fn({"fcf_yield": -0.25}) is False

    def test_nan_fcf_yield_passes(self):
        assert self.rule.check_fn({"fcf_yield": float("nan")}) is True

    def test_severity_is_warn(self):
        assert self.rule.severity == Severity.WARN


class TestDQW003:
    rule = _get_rule("DQ-W003")

    def test_normal_roic_passes(self):
        assert self.rule.check_fn({"roic_current": 0.15}) is True

    def test_extreme_high_roic_warns(self):
        assert self.rule.check_fn({"roic_current": 0.65}) is False

    def test_extreme_low_roic_warns(self):
        assert self.rule.check_fn({"roic_current": -0.25}) is False

    def test_nan_roic_passes(self):
        assert self.rule.check_fn({"roic_current": float("nan")}) is True

    def test_severity_is_warn(self):
        assert self.rule.severity == Severity.WARN


class TestDQW004:
    rule = _get_rule("DQ-W004")

    def test_positive_shares_passes(self):
        assert self.rule.check_fn({"shares_outstanding": 1_000.0}) is True

    def test_negative_shares_warns(self):
        assert self.rule.check_fn({"shares_outstanding": -100.0}) is False

    def test_zero_shares_warns(self):
        assert self.rule.check_fn({"shares_outstanding": 0.0}) is False

    def test_nan_shares_passes(self):
        assert self.rule.check_fn({"shares_outstanding": float("nan")}) is True

    def test_severity_is_warn(self):
        assert self.rule.severity == Severity.WARN


# ---------------------------------------------------------------------------
# run_dq_rules() integrasjonstester
# ---------------------------------------------------------------------------

class TestRunDQRules:

    def test_clean_row_has_no_failures(self):
        df = _df(_base_row())
        flags, audit = run_dq_rules(df)
        assert flags["dq_fail_v2"].iloc[0] is False or flags["dq_fail_v2"].iloc[0] == False
        assert flags["dq_fail_count_v2"].iloc[0] == 0

    def test_missing_price_triggers_fail(self):
        df = _df(_base_row(adj_close=float("nan")))
        flags, audit = run_dq_rules(df)
        assert flags["dq_fail_v2"].iloc[0] == True  # noqa: E712 — pandas col is np.bool_
        assert "DQ-F001" in flags["dq_fail_rules"].iloc[0]

    def test_missing_market_cap_triggers_fail(self):
        df = _df(_base_row(market_cap=float("nan")))
        flags, _ = run_dq_rules(df)
        assert "DQ-F002" in flags["dq_fail_rules"].iloc[0]

    def test_warn_does_not_set_dq_fail(self):
        """WARN-brudd skal ikke sette dq_fail_v2=True."""
        df = _df(_base_row(nd_ebitda=15.0))  # DQ-W001 vil trigge
        flags, _ = run_dq_rules(df)
        assert flags["dq_fail_v2"].iloc[0] is False or flags["dq_fail_v2"].iloc[0] == False
        assert flags["dq_warn_count_v2"].iloc[0] >= 1
        assert "DQ-W001" in flags["dq_warn_rules"].iloc[0]

    def test_multiple_fails_counted_correctly(self):
        """Rad med manglende pris + manglende market_cap gir fail_count=2."""
        df = _df(_base_row(adj_close=float("nan"), market_cap=float("nan")))
        flags, _ = run_dq_rules(df)
        assert flags["dq_fail_count_v2"].iloc[0] >= 2

    def test_audit_df_has_one_row_per_ticker_per_rule(self):
        df = _df(_base_row(ticker="EQNR"), _base_row(ticker="DNB"))
        _, audit = run_dq_rules(df)
        expected_rows = len(df) * len(RULES)
        assert len(audit) == expected_rows

    def test_audit_df_columns(self):
        df = _df(_base_row())
        _, audit = run_dq_rules(df)
        for col in ["ticker", "rule_id", "severity", "fields", "passed", "description"]:
            assert col in audit.columns

    def test_flags_df_index_matches_input(self):
        df = _df(_base_row(ticker="A"), _base_row(ticker="B"))
        df.index = [10, 20]
        flags, _ = run_dq_rules(df)
        assert list(flags.index) == [10, 20]

    def test_bank_sector_ev_rule_not_fail(self):
        """Bank med negativ EV skal ikke trigge DQ-F003."""
        df = _df(_base_row(enterprise_value=-100.0, sector_group="bank"))
        flags, audit = run_dq_rules(df)
        f003_rows = audit[audit["rule_id"] == "DQ-F003"]
        assert f003_rows["passed"].iloc[0] == True  # noqa: E712 — pandas col is np.bool_

    def test_rule_ids_are_unique(self):
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids)), "Duplikate regel-IDer funnet"

    def test_all_fail_rules_present(self):
        fail_ids = {r.rule_id for r in RULES if r.severity == Severity.FAIL}
        expected = {"DQ-F001", "DQ-F002", "DQ-F003", "DQ-F004", "DQ-F005", "DQ-F006"}
        assert expected == fail_ids

    def test_all_warn_rules_present(self):
        warn_ids = {r.rule_id for r in RULES if r.severity == Severity.WARN}
        expected = {"DQ-W001", "DQ-W002", "DQ-W003", "DQ-W004"}
        assert expected == warn_ids
