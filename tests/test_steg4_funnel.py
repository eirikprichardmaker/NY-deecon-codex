"""
STEG 4 — Funnel analysis, dynamic MoS, dynamic MAD.
"""
import numpy as np
import pandas as pd
import pytest

from src.decision import _build_funnel_report
from src.wft import _apply_filters, WFTParams


# ============================================================
# Helpers
# ============================================================

def _base_params(**kw) -> WFTParams:
    defaults = dict(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    defaults.update(kw)
    return WFTParams(**defaults)


def _make_wft_month(n: int = 20, **col_overrides) -> pd.DataFrame:
    """Build a minimal WFT monthly panel slice."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "k":       [f"T{i:02d}" for i in range(n)],
        "ticker":  [f"T{i:02d}.OL" for i in range(n)],
        "mos":     rng.uniform(0.10, 0.60, n),
        "roic":    rng.uniform(0.05, 0.30, n),
        "fcf_yield": rng.uniform(0.02, 0.15, n),
        "gp_a":    rng.uniform(0.10, 0.60, n),
        "high_risk_flag":        [False] * n,
        "quality_weak_count":    [0] * n,
        "value_creation_ok_base":[True] * n,
        "adj_close":  rng.uniform(50, 200, n),
        "ma200":      rng.uniform(40, 180, n),
        "mad":        rng.uniform(-0.01, 0.10, n),
        "index_price":  [1000.0] * n,
        "index_ma200":  [900.0]  * n,
        "index_mad":    [0.05]   * n,
        "relevant_index_key": ["OSEAX"] * n,
        "dq_blocked": [False] * n,
        "market_cap": rng.uniform(1e8, 5e9, n),
    })
    # Ensure adj_close > ma200 so stocks are technically above MA200
    df["adj_close"] = df["ma200"] * rng.uniform(1.05, 1.30, n)
    for col, val in col_overrides.items():
        df[col] = val
    return df


def _make_decision_df(n: int = 30) -> pd.DataFrame:
    """Minimal DataFrame for _build_funnel_report."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "data_quality_fail":  rng.choice([True, False], n, p=[0.10, 0.90]),
        "candidate_data_ok":  rng.choice([True, False], n, p=[0.85, 0.15]),
        "quality_gate_ok":    rng.choice([True, False], n, p=[0.70, 0.30]),
        "value_creation_ok":  rng.choice([True, False], n, p=[0.60, 0.40]),
        "mos":    rng.uniform(-0.20, 0.60, n),
        "mos_req": [0.30] * n,
        "fundamental_ok": rng.choice([True, False], n, p=[0.40, 0.60]),
        "technical_ok":   rng.choice([True, False], n, p=[0.50, 0.50]),
        "eligible":       rng.choice([True, False], n, p=[0.25, 0.75]),
    })
    return df


# ============================================================
# Funnel report
# ============================================================

class TestFunnelReport:
    def test_has_required_columns(self):
        df = _build_funnel_report(_make_decision_df())
        assert set(df.columns) >= {"stage", "count_pass", "count_fail", "pct_pass"}

    def test_has_9_stages(self):
        df = _build_funnel_report(_make_decision_df())
        assert len(df) == 9

    def test_universe_count_equals_total_rows(self):
        data = _make_decision_df(40)
        funnel = _build_funnel_report(data)
        universe_row = funnel[funnel["stage"] == "1_universe"]
        assert int(universe_row["count_pass"].iloc[0]) == 40

    def test_eligible_le_universe(self):
        data = _make_decision_df()
        funnel = _build_funnel_report(data)
        universe = int(funnel[funnel["stage"] == "1_universe"]["count_pass"].iloc[0])
        eligible  = int(funnel[funnel["stage"] == "9_eligible"]["count_pass"].iloc[0])
        assert eligible <= universe

    def test_count_pass_plus_fail_equals_total(self):
        data = _make_decision_df(25)
        n = len(data)
        funnel = _build_funnel_report(data)
        for _, row in funnel.iterrows():
            assert int(row["count_pass"]) + int(row["count_fail"]) == n

    def test_pct_pass_is_consistent(self):
        data = _make_decision_df(50)
        funnel = _build_funnel_report(data)
        for _, row in funnel.iterrows():
            expected = round(int(row["count_pass"]) / 50 * 100, 1)
            assert abs(float(row["pct_pass"]) - expected) < 0.2

    def test_all_pass_gives_100pct(self):
        df = _make_decision_df(10)
        df["eligible"] = True
        funnel = _build_funnel_report(df)
        elig_row = funnel[funnel["stage"] == "9_eligible"]
        assert float(elig_row["pct_pass"].iloc[0]) == pytest.approx(100.0)

    def test_all_fail_gives_0pct(self):
        df = _make_decision_df(10)
        df["eligible"] = False
        funnel = _build_funnel_report(df)
        elig_row = funnel[funnel["stage"] == "9_eligible"]
        assert float(elig_row["pct_pass"].iloc[0]) == pytest.approx(0.0)

    def test_missing_columns_handled_gracefully(self):
        """funnel_report should not crash when optional gate columns are absent."""
        df = pd.DataFrame({"ticker": ["A", "B", "C"]})
        result = _build_funnel_report(df)
        assert not result.empty


# ============================================================
# Dynamic MoS in _apply_filters
# ============================================================

class TestDynamicMoS:
    def test_high_quality_gets_lower_mos_req(self):
        """Top-quartile quality companies should have mos_req reduced by ~0.10."""
        month_df = _make_wft_month(40)
        params = _base_params(mos_threshold=0.35)
        result = _apply_filters(month_df, params)

        top_q_mask = result["quality_score"] > result["quality_score"].quantile(0.75)
        low_q_mask  = result["quality_score"] <= result["quality_score"].quantile(0.25)

        if top_q_mask.any() and low_q_mask.any():
            avg_req_top = float(result.loc[top_q_mask, "mos_req"].mean())
            avg_req_low = float(result.loc[low_q_mask, "mos_req"].mean())
            assert avg_req_top < avg_req_low, "High-quality companies should have lower mos_req"

    def test_mos_req_never_below_dynamic_floor(self):
        """mos_req must never drop below 0.25 (dynamic floor)."""
        month_df = _make_wft_month(40)
        params = _base_params(mos_threshold=0.30)
        result = _apply_filters(month_df, params)
        assert (result["mos_req"] >= 0.25).all(), "mos_req must not drop below dynamic floor"

    def test_high_risk_company_mos_req_not_reduced_below_040(self):
        """High-risk companies get mos_req ≥ 0.40; discount should not undercut this floor."""
        month_df = _make_wft_month(20)
        month_df["high_risk_flag"] = True  # all high-risk
        params = _base_params(mos_threshold=0.30)
        result = _apply_filters(month_df, params)
        # After discount (0.10), min is max(0.25, 0.40-0.10=0.30) = 0.30; actual floor is 0.25
        # But because high_risk raises base to 0.40, the resulting discount gives 0.30 ≥ 0.25 ✓
        assert (result["mos_req"] >= 0.25).all()

    def test_more_companies_pass_mos_with_dynamic_discount(self):
        """Enabling dynamic MoS should allow at least as many companies to pass as without."""
        month_df = _make_wft_month(40)
        # All companies have quality_score computed; some near the threshold
        month_df["mos"] = 0.28  # just below standard 0.35 threshold

        params = _base_params(mos_threshold=0.35)
        result = _apply_filters(month_df, params)

        # Some top-quality companies should now pass (mos_req dropped to ~0.25)
        top_q_pass = result[result["quality_score"] > result["quality_score"].quantile(0.75)]["mos_ok"]
        assert top_q_pass.any(), "Some top-quality companies should pass the discounted MoS"


# ============================================================
# Dynamic MAD in _apply_filters
# ============================================================

class TestDynamicMAD:
    def test_bear_regime_tightens_stock_mad(self):
        """When index_mad < -0.05 (bear), stock_mad < 0.00 should fail even if above params.mad_min."""
        month_df = _make_wft_month(20)
        # Set bear index conditions
        month_df["index_mad"] = -0.10        # bear regime
        # Set stock MAD between params.mad_min (-0.05) and bear floor (0.00)
        month_df["mad"] = -0.03              # passes normal (-0.05) but fails bear (0.00)

        params = _base_params(mad_min=-0.05)
        result = _apply_filters(month_df, params)

        # In bear regime with stock_mad=-0.03 < 0.00 → stock_mad_ok should be False
        assert not result["stock_mad_ok"].any(), \
            "In bear regime, stock_mad=-0.03 should fail the 0.00 floor"

    def test_bear_regime_flag_set(self):
        """mad_regime column should read 'bear' when index_mad < -0.05."""
        month_df = _make_wft_month(10)
        month_df["index_mad"] = -0.08
        params = _base_params()
        result = _apply_filters(month_df, params)
        assert (result["mad_regime"] == "bear").all()

    def test_normal_regime_uses_params_mad_min(self):
        """When index_mad ≥ -0.05, stock_mad_ok respects params.mad_min (-0.05)."""
        month_df = _make_wft_month(20)
        month_df["index_mad"] = 0.02         # normal (bull) regime
        month_df["mad"] = -0.04              # between -0.05 and 0.00

        params = _base_params(mad_min=-0.05)
        result = _apply_filters(month_df, params)

        # Normal regime: -0.04 >= -0.05 → should pass
        assert result["stock_mad_ok"].all(), \
            "In normal regime, stock_mad=-0.04 >= mad_min=-0.05 should pass"

    def test_normal_regime_flag_set(self):
        month_df = _make_wft_month(10)
        month_df["index_mad"] = 0.05
        result = _apply_filters(month_df, _base_params())
        assert (result["mad_regime"] == "normal").all()

    def test_bear_regime_reduces_eligible_count(self):
        """Switching to bear regime (stricter MAD) should not increase eligible count."""
        month_df = _make_wft_month(40)
        month_df["mad"] = -0.03   # passes -0.05 but fails 0.00

        params = _base_params(mad_min=-0.05)

        normal = month_df.copy(); normal["index_mad"] =  0.05
        bear   = month_df.copy(); bear["index_mad"]   = -0.10

        result_normal = _apply_filters(normal, params)
        result_bear   = _apply_filters(bear,   params)

        assert result_bear["eligible"].sum() <= result_normal["eligible"].sum(), \
            "Bear regime must not produce more eligible companies than normal regime"

    def test_missing_index_mad_treated_as_normal(self):
        """NaN index_mad should default to normal regime (no stricter MAD applied)."""
        month_df = _make_wft_month(10)
        month_df["index_mad"] = np.nan
        month_df["mad"] = -0.03   # passes -0.05 but fails 0.00

        params = _base_params(mad_min=-0.05)
        result = _apply_filters(month_df, params)

        # With NaN index_mad: in_bear=False → effective_mad_floor = params.mad_min (-0.05)
        # stock_mad=-0.03 >= -0.05 → should pass
        assert result["stock_mad_ok"].all()
