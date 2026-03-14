"""
Tests for fallback valuation chain.
Kontrakt: minst 1 positiv + 1 negativ test per metode og gate.
"""
import numpy as np
import pandas as pd
import pytest

from src.valuation import (
    dcf_valuation,
    epv_valuation,
    multiples_valuation,
    valuation_chain,
    compute_sector_medians,
    ValuationMethod,
    ConfidenceTier,
    MOS_FLOOR_BY_TIER,
)


# ============================================================
# Fixtures
# ============================================================

DEFAULT_CONFIG = {
    "terminal_growth": 0.02,
    "projection_years": 5,
    "multiples_min_peers": 5,
    "multiples_haircut": 0.20,
}


def make_row(**overrides):
    """Lag en pd.Series med rimelige defaults for et nordisk selskap."""
    base = {
        "ticker": "TEST.OL",
        "sector": "Industrials",
        "price": 100.0,
        "shares_outstanding": 1_000_000,
        "net_debt": 50_000_000,
        "fcf_m_median_3y": 30_000_000,
        "ebit_m_median_3y": 50_000_000,
        "ebit": 50_000_000,
        "depreciation_amortization": 10_000_000,
        "capex": 20_000_000,
        "effective_tax_rate": 0.22,
        "ev": 500_000_000,
    }
    base.update(overrides)
    return pd.Series(base)


def make_sector_medians(sector="Industrials", median_ev_ebit=12.0, peer_count=20):
    return pd.DataFrame([{
        "sector": sector,
        "median_ev_ebit": median_ev_ebit,
        "peer_count": peer_count,
    }])


# ============================================================
# DCF Tests
# ============================================================

class TestDCF:
    def test_positive_fcf_returns_valuation(self):
        """DCF med positiv FCF → gyldig intrinsic value."""
        row = make_row(fcf_m_median_3y=30_000_000)
        result = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is not None
        assert result.method == ValuationMethod.DCF
        assert result.confidence == ConfidenceTier.HIGH
        assert result.intrinsic_value_per_share > 0
        assert result.mos is not None

    def test_negative_fcf_returns_none(self):
        """DCF med negativ FCF → None (signal til chain å prøve EPV)."""
        row = make_row(fcf_m_median_3y=-5_000_000)
        result = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is None

    def test_zero_fcf_returns_none(self):
        row = make_row(fcf_m_median_3y=0)
        result = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is None

    def test_missing_fcf_returns_none(self):
        row = make_row(fcf_m_median_3y=np.nan)
        result = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is None

    def test_missing_shares_returns_unvalued(self):
        """DCF med positiv FCF men manglende shares → NONE confidence."""
        row = make_row(shares_outstanding=0)
        result = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is not None
        assert result.confidence == ConfidenceTier.NONE
        assert result.intrinsic_value_per_share is None

    def test_mos_calculation_correct(self):
        """MoS beregnes som (IV - price) / IV."""
        row = make_row(fcf_m_median_3y=30_000_000, price=50.0)
        result = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        expected_mos = (result.intrinsic_value_per_share - 50.0) / result.intrinsic_value_per_share
        assert abs(result.mos - expected_mos) < 0.001

    def test_terminal_growth_capped_at_2pct(self):
        """Terminal growth > 2% klippes til 2%."""
        config = {**DEFAULT_CONFIG, "terminal_growth": 0.05}
        row = make_row()
        result = dcf_valuation(row, wacc=0.10, config=config)
        assert result is not None
        assert "terminal_growth" in result.inputs_used
        assert result.inputs_used["terminal_growth"] <= 0.02


# ============================================================
# EPV Tests
# ============================================================

class TestEPV:
    def test_positive_ebit_returns_valuation(self):
        """EPV med positiv EBIT → gyldig IV med MEDIUM confidence."""
        row = make_row(fcf_m_median_3y=-1, ebit_m_median_3y=50_000_000)
        result = epv_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is not None
        assert result.method == ValuationMethod.EPV
        assert result.confidence == ConfidenceTier.MEDIUM
        assert result.intrinsic_value_per_share > 0

    def test_negative_ebit_returns_none(self):
        """EPV med negativ ebit_m_median_3y → None."""
        row = make_row(ebit_m_median_3y=-10_000_000, ebit=-10_000_000)
        result = epv_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is None

    def test_epv_has_higher_mos_floor_than_dcf(self):
        """EPV krever MoS ≥ 40% (høyere enn DCF's 30%)."""
        row = make_row(ebit_m_median_3y=50_000_000)
        result = epv_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is not None
        assert result.mos_floor == MOS_FLOOR_BY_TIER[ConfidenceTier.MEDIUM]
        assert result.mos_floor > MOS_FLOOR_BY_TIER[ConfidenceTier.HIGH]

    def test_epv_no_growth_assumption(self):
        """EPV skal IKKE ha terminal growth — den kapitaliserer flat inntjening."""
        row = make_row(ebit_m_median_3y=50_000_000)
        result = epv_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is not None
        assert "terminal_growth" not in result.inputs_used

    def test_maintenance_capex_deducted(self):
        """EPV trekker fra maintenance capex (D&A)."""
        row = make_row(
            ebit_m_median_3y=50_000_000,
            depreciation_amortization=10_000_000,
        )
        result = epv_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert result is not None
        nopat = result.inputs_used["nopat"]
        expected = 50_000_000 * (1 - 0.22) - 10_000_000
        assert abs(nopat - expected) < 1

    def test_epv_and_dcf_both_produce_valid_result(self):
        """Både DCF og EPV produserer gyldig IV for et selskap med positiv FCF og EBIT."""
        row = make_row(fcf_m_median_3y=30_000_000, ebit_m_median_3y=50_000_000)
        dcf = dcf_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        epv = epv_valuation(row, wacc=0.10, config=DEFAULT_CONFIG)
        assert dcf is not None and dcf.intrinsic_value_per_share is not None
        assert epv is not None and epv.intrinsic_value_per_share is not None


# ============================================================
# Multiples Tests
# ============================================================

class TestMultiples:
    def test_valid_sector_returns_valuation(self):
        """Multiples med gyldig sektor og nok peers → LOW confidence."""
        row = make_row(ebit_m_median_3y=50_000_000)
        medians = make_sector_medians()
        result = multiples_valuation(row, medians, DEFAULT_CONFIG)
        assert result.method == ValuationMethod.MULTIPLES
        assert result.confidence == ConfidenceTier.LOW
        assert result.intrinsic_value_per_share > 0

    def test_too_few_peers_returns_unvalued(self):
        """Multiples med < min_peers → UNVALUED."""
        row = make_row()
        medians = make_sector_medians(peer_count=3)
        result = multiples_valuation(row, medians, DEFAULT_CONFIG)
        assert result.method == ValuationMethod.UNVALUED
        assert result.confidence == ConfidenceTier.NONE

    def test_unknown_sector_returns_unvalued(self):
        row = make_row(sector="NonexistentSector")
        medians = make_sector_medians(sector="Industrials")
        result = multiples_valuation(row, medians, DEFAULT_CONFIG)
        assert result.confidence == ConfidenceTier.NONE

    def test_haircut_applied(self):
        """20% haircut på sektormedian EV/EBIT."""
        row = make_row(ebit_m_median_3y=50_000_000)
        medians = make_sector_medians(median_ev_ebit=12.0)
        result = multiples_valuation(row, medians, DEFAULT_CONFIG)
        assert abs(result.inputs_used["conservative_multiple"] - 12.0 * 0.80) < 0.001

    def test_multiples_highest_mos_floor(self):
        """Multiples krever MoS ≥ 50%."""
        row = make_row()
        medians = make_sector_medians()
        result = multiples_valuation(row, medians, DEFAULT_CONFIG)
        assert result.mos_floor == 0.50

    def test_negative_ebit_returns_unvalued(self):
        """Ingen positiv EBIT → UNVALUED (siste stopp i chain)."""
        row = make_row(ebit_m_median_3y=-10_000_000, ebit=-10_000_000)
        medians = make_sector_medians()
        result = multiples_valuation(row, medians, DEFAULT_CONFIG)
        assert result.method == ValuationMethod.UNVALUED


# ============================================================
# Chain Tests (integrasjon)
# ============================================================

class TestValuationChain:
    def test_dcf_preferred_when_available(self):
        """Chain velger DCF når FCF er positiv."""
        row = make_row(fcf_m_median_3y=30_000_000)
        medians = make_sector_medians()
        result = valuation_chain(row, 0.10, medians, DEFAULT_CONFIG)
        assert result.method == ValuationMethod.DCF

    def test_falls_to_epv_on_negative_fcf(self):
        """Chain faller til EPV når FCF er negativ men EBIT er positiv."""
        row = make_row(fcf_m_median_3y=-5_000_000, ebit_m_median_3y=50_000_000)
        medians = make_sector_medians()
        result = valuation_chain(row, 0.10, medians, DEFAULT_CONFIG)
        assert result.method == ValuationMethod.EPV
        assert any("DCF feilet" in r for r in result.reasons)

    def test_falls_to_multiples_on_negative_ebit_after_epv(self):
        """Chain prøver alle metoder; UNVALUED er akseptabelt utfall."""
        row = make_row(
            fcf_m_median_3y=-5_000_000,
            ebit_m_median_3y=-2_000_000,
            ebit=-2_000_000,
        )
        medians = make_sector_medians()
        result = valuation_chain(row, 0.10, medians, DEFAULT_CONFIG)
        assert result.method in (ValuationMethod.MULTIPLES, ValuationMethod.UNVALUED)

    def test_unvalued_when_everything_fails(self):
        """Alt feiler → UNVALUED med full audit trail."""
        row = make_row(fcf_m_median_3y=-1, ebit_m_median_3y=-1, ebit=-1)
        medians = make_sector_medians()
        result = valuation_chain(row, 0.10, medians, DEFAULT_CONFIG)
        assert result.method == ValuationMethod.UNVALUED
        assert result.confidence == ConfidenceTier.NONE
        assert len(result.reasons) >= 3

    def test_chain_preserves_attempt_log(self):
        """Reasons skal inneholde info om alle forsøkte metoder."""
        row = make_row(fcf_m_median_3y=-5_000_000, ebit_m_median_3y=50_000_000)
        medians = make_sector_medians()
        result = valuation_chain(row, 0.10, medians, DEFAULT_CONFIG)
        combined = " ".join(result.reasons)
        assert "DCF feilet" in combined
        assert "EPV" in combined


# ============================================================
# Sector Medians Tests
# ============================================================

class TestSectorMedians:
    def test_computes_median_correctly(self):
        df = pd.DataFrame([
            {"sector": "Tech", "ev": 100, "ebit": 10},
            {"sector": "Tech", "ev": 200, "ebit": 10},
            {"sector": "Tech", "ev": 150, "ebit": 10},
        ])
        result = compute_sector_medians(df)
        assert len(result) == 1
        assert result.iloc[0]["median_ev_ebit"] == 15.0
        assert result.iloc[0]["peer_count"] == 3

    def test_excludes_negative_ebit(self):
        df = pd.DataFrame([
            {"sector": "Tech", "ev": 100, "ebit": 10},
            {"sector": "Tech", "ev": 100, "ebit": -5},
        ])
        result = compute_sector_medians(df)
        assert result.iloc[0]["peer_count"] == 1

    def test_excludes_extreme_multiples(self):
        df = pd.DataFrame([
            {"sector": "Tech", "ev": 100,  "ebit": 10},  # 10x OK
            {"sector": "Tech", "ev": 1000, "ebit": 10},  # 100x → filtrert
            {"sector": "Tech", "ev": 5,    "ebit": 10},  # 0.5x → filtrert
        ])
        result = compute_sector_medians(df)
        assert result.iloc[0]["peer_count"] == 1
        assert result.iloc[0]["median_ev_ebit"] == 10.0


# ============================================================
# Confidence Tier / MoS Floor Tests
# ============================================================

class TestConfidenceTiers:
    def test_mos_floors_are_monotonically_increasing(self):
        """Lavere confidence → høyere MoS-krav."""
        assert MOS_FLOOR_BY_TIER[ConfidenceTier.HIGH] < MOS_FLOOR_BY_TIER[ConfidenceTier.MEDIUM]
        assert MOS_FLOOR_BY_TIER[ConfidenceTier.MEDIUM] < MOS_FLOOR_BY_TIER[ConfidenceTier.LOW]
        assert MOS_FLOOR_BY_TIER[ConfidenceTier.NONE] == float("inf")

    def test_none_tier_blocks_candidate(self):
        """NONE confidence → MoS-gulv er uendelig → kan aldri bli kandidat."""
        assert MOS_FLOOR_BY_TIER[ConfidenceTier.NONE] == float("inf")
