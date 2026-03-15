"""tests/test_decision_cyclical.py

Tests for cyclicality helper functions in decision.py:
  - get_fcf_window: returns correct window length per sector
  - check_cyclical_guards: returns failures for cyclical companies
"""
import pytest
from src.decision import check_cyclical_guards, get_fcf_window

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
