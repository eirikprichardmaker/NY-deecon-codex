"""tests/test_decision_cyclical.py

Tests for:
  - get_fcf_window: cyclical/non-cyclical FCF normalisation window
  - check_cyclical_guards: EV/EBIT + MoS gates for cyclical sectors
  - _run_data_quality_checks: DQ-EVEBIT-GLOBAL, DQ-MCAP-MIN, DQ-MOS-SUSPECT rules
"""
import numpy as np
import pandas as pd
import pytest

from src.decision import _run_data_quality_checks, check_cyclical_guards, get_fcf_window

# ── Shared helpers ─────────────────────────────────────────────────────────────

DQ_CFG = {
    "data_quality": {
        "ev_ebit_floor_global": 6.0,
        "mcap_min_hard": 50_000_000,
        "mcap_min_warn": 100_000_000,
        "mos_suspect_threshold": 0.80,
        "mos_suspect_evebit_min": 8.0,
    }
}


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal single-row DataFrame for DQ checks."""
    defaults = {
        "ticker": "TEST.ST",
        "market_cap": 500_000_000,  # 500M — above both thresholds by default
        "mos": 0.40,
        "ev_ebit": 9.0,
        "is_bank_proxy": False,
        "roic": 0.12,
        "fcf_yield": 0.08,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])

CYCLICAL_CONFIG = {
    "cyclicality": {
        "cyclical_damodaran_sectors": ["Materials", "Oil & Gas", "Shipping", "Seafood"],
        "fcf_window_years": 5,
        "default_fcf_window_years": 3,
        "ev_ebit_floor": 6.0,
        "mos_threshold": 0.40,
    }
}


# ─── get_fcf_window ────────────────────────────────────────────────────────────

class TestGetFcfWindow:
    def test_cyclical_sector_returns_5yr(self):
        for sector in ["Materials", "Oil & Gas", "Shipping", "Seafood"]:
            assert get_fcf_window(sector, CYCLICAL_CONFIG) == 5

    def test_non_cyclical_sector_returns_3yr(self):
        for sector in ["Technology", "Financials", "Healthcare", ""]:
            assert get_fcf_window(sector, CYCLICAL_CONFIG) == 3

    def test_empty_config_uses_defaults(self):
        # Empty config: no cyclical sectors defined, so all sectors return default 3yr window
        assert get_fcf_window("Materials", {}) == 3
        assert get_fcf_window("Technology", {}) == 3


# ─── check_cyclical_guards ────────────────────────────────────────────────────

class TestCheckCyclicalGuards:
    def _row(self, **kwargs):
        defaults = {"is_cyclical": True, "mos": 0.45, "ev_ebit": 8.0}
        defaults.update(kwargs)
        return defaults

    def test_non_cyclical_passes_all_guards(self):
        row = self._row(is_cyclical=False, mos=0.10, ev_ebit=2.0)
        assert check_cyclical_guards(row, CYCLICAL_CONFIG) == []

    def test_cyclical_meeting_all_thresholds_passes(self):
        row = self._row(is_cyclical=True, mos=0.45, ev_ebit=8.0)
        assert check_cyclical_guards(row, CYCLICAL_CONFIG) == []

    def test_cyclical_low_mos_fails(self):
        row = self._row(is_cyclical=True, mos=0.25, ev_ebit=8.0)
        failures = check_cyclical_guards(row, CYCLICAL_CONFIG)
        assert len(failures) == 1
        assert failures[0]["guard"] == "mos_min"
        assert failures[0]["threshold"] == 0.40

    def test_cyclical_low_ev_ebit_fails(self):
        row = self._row(is_cyclical=True, mos=0.45, ev_ebit=3.7)
        failures = check_cyclical_guards(row, CYCLICAL_CONFIG)
        assert len(failures) == 1
        assert failures[0]["guard"] == "ev_ebit_floor"
        assert failures[0]["threshold"] == 6.0

    def test_cyclical_null_ev_ebit_does_not_fail(self):
        """Missing EV/EBIT data should not block a cyclical company."""
        row = self._row(is_cyclical=True, mos=0.45, ev_ebit=None)
        failures = check_cyclical_guards(row, CYCLICAL_CONFIG)
        assert not any(f["guard"] == "ev_ebit_floor" for f in failures)

    def test_cyclical_both_fail(self):
        row = self._row(is_cyclical=True, mos=0.20, ev_ebit=2.0)
        failures = check_cyclical_guards(row, CYCLICAL_CONFIG)
        guards = {f["guard"] for f in failures}
        assert "mos_min" in guards
        assert "ev_ebit_floor" in guards

    def test_mos_exactly_at_threshold_passes(self):
        row = self._row(is_cyclical=True, mos=0.40, ev_ebit=8.0)
        assert check_cyclical_guards(row, CYCLICAL_CONFIG) == []

    def test_ev_ebit_exactly_at_floor_passes(self):
        row = self._row(is_cyclical=True, mos=0.45, ev_ebit=6.0)
        assert check_cyclical_guards(row, CYCLICAL_CONFIG) == []


# ─── DQ-EVEBIT-GLOBAL ─────────────────────────────────────────────────────────

class TestDqEvEbitGlobal:
    def _run(self, **kwargs):
        df = _make_df(**kwargs)
        flags, audit = _run_data_quality_checks(df, dec_cfg=DQ_CFG, asof="2024-01-01")
        return flags, audit

    def test_blocks_non_cyclical_below_floor(self):
        """DEDI-like: Industrials, non-cyclical, EV/EBIT=5.96 → FAIL, decision CASH."""
        flags, audit = self._run(ev_ebit=5.96, is_bank_proxy=False)
        assert bool(flags.loc[0, "dq_blocked"]), "Expected dq_blocked=True"
        fail_ids = audit[audit["severity"] == "FAIL"]["rule_id"].tolist()
        assert "DQ-EVEBIT-GLOBAL" in fail_ids

    def test_passes_above_floor(self):
        """EV/EBIT=6.5 → not blocked by DQ-EVEBIT-GLOBAL."""
        flags, audit = self._run(ev_ebit=6.5, is_bank_proxy=False)
        global_fails = audit[(audit["severity"] == "FAIL") & (audit["rule_id"] == "DQ-EVEBIT-GLOBAL")]
        assert global_fails.empty

    def test_skips_banks(self):
        """is_bank_proxy=True, ev_ebit=3.0 → DQ-EVEBIT-GLOBAL not triggered."""
        flags, audit = self._run(ev_ebit=3.0, is_bank_proxy=True)
        global_events = audit[audit["rule_id"] == "DQ-EVEBIT-GLOBAL"]
        assert global_events.empty

    def test_warns_on_negative_ebit(self):
        """ev_ebit=-2.0 → WARN (not FAIL), dq_blocked stays False."""
        flags, audit = self._run(ev_ebit=-2.0, is_bank_proxy=False, mos=0.30)
        global_events = audit[audit["rule_id"] == "DQ-EVEBIT-GLOBAL"]
        assert not global_events.empty
        assert (global_events["severity"] == "WARN").all()
        # A WARN alone should not block
        fail_count = int(flags.loc[0, "dq_fail_count"])
        global_fails = audit[(audit["severity"] == "FAIL") & (audit["rule_id"] == "DQ-EVEBIT-GLOBAL")]
        assert global_fails.empty


# ─── DQ-MCAP-MIN ──────────────────────────────────────────────────────────────

class TestDqMcapMin:
    def _run(self, **kwargs):
        df = _make_df(**kwargs)
        flags, audit = _run_data_quality_checks(df, dec_cfg=DQ_CFG, asof="2024-01-01")
        return flags, audit

    def test_hard_fail_below_50m(self):
        """market_cap=40M → FAIL, dq_blocked=True."""
        flags, audit = self._run(market_cap=40_000_000, ev_ebit=9.0, mos=0.35)
        assert bool(flags.loc[0, "dq_blocked"])
        fail_ids = audit[audit["severity"] == "FAIL"]["rule_id"].tolist()
        assert "DQ-MCAP-MIN" in fail_ids

    def test_soft_warn_between_50m_and_100m(self):
        """market_cap=75M → WARN, not FAIL → dq_blocked=False (only WARN)."""
        flags, audit = self._run(market_cap=75_000_000, ev_ebit=9.0, mos=0.35)
        warn_ids = audit[audit["severity"] == "WARN"]["rule_id"].tolist()
        assert "DQ-MCAP-MIN" in warn_ids
        mcap_fails = audit[(audit["severity"] == "FAIL") & (audit["rule_id"] == "DQ-MCAP-MIN")]
        assert mcap_fails.empty

    def test_passes_above_warn(self):
        """market_cap=150M → no DQ-MCAP-MIN event at all."""
        flags, audit = self._run(market_cap=150_000_000, ev_ebit=9.0, mos=0.35)
        mcap_events = audit[audit["rule_id"] == "DQ-MCAP-MIN"]
        assert mcap_events.empty


# ─── DQ-MOS-SUSPECT ───────────────────────────────────────────────────────────

class TestDqMosSuspect:
    def _run(self, **kwargs):
        df = _make_df(**kwargs)
        flags, audit = _run_data_quality_checks(df, dec_cfg=DQ_CFG, asof="2024-01-01")
        return flags, audit

    def test_warns_high_mos_low_evebit(self):
        """mos=0.94, ev_ebit=5.96 → WARN DQ-MOS-SUSPECT (not FAIL → not blocking alone)."""
        # Also set ev_ebit above global floor (6.5) to isolate DQ-MOS-SUSPECT
        flags, audit = self._run(mos=0.94, ev_ebit=6.5)
        suspect_events = audit[audit["rule_id"] == "DQ-MOS-SUSPECT"]
        assert not suspect_events.empty, "Expected DQ-MOS-SUSPECT warning"
        assert (suspect_events["severity"] == "WARN").all()

    def test_no_warn_high_mos_high_evebit(self):
        """mos=0.94, ev_ebit=10.0 → no DQ-MOS-SUSPECT (EV/EBIT confirms valuation)."""
        flags, audit = self._run(mos=0.94, ev_ebit=10.0)
        suspect_events = audit[audit["rule_id"] == "DQ-MOS-SUSPECT"]
        assert suspect_events.empty

    def test_no_warn_low_mos(self):
        """mos=0.40, ev_ebit=5.0 → DQ-MOS-SUSPECT not triggered (MoS below threshold)."""
        flags, audit = self._run(mos=0.40, ev_ebit=5.0)
        suspect_events = audit[audit["rule_id"] == "DQ-MOS-SUSPECT"]
        assert suspect_events.empty
