"""
STEG 3 — Factor scoring: GP/A sub-factor, rebalanced weights, correlation logging.
"""
import json
import logging

import numpy as np
import pandas as pd
import pytest

from src.decision import (
    _baseline_quality_score,
    _strategy_a_composite_score,
    _compute_factor_correlations,
    _log_factor_correlations,
    _STRATEGY_A_WEIGHTS_DEFAULT,
    _QUALITY_SUBWEIGHTS,
    _FACTOR_CORR_WARN_THRESHOLD,
    _GPA_CAP,
)


# ============================================================
# Fixtures
# ============================================================

def _make_df(**overrides) -> pd.DataFrame:
    base = {
        "ticker": ["A.OL", "B.OL", "C.OL", "D.OL", "E.OL",
                   "F.OL", "G.OL", "H.OL", "I.OL", "J.OL"],
        "roic":      [0.15, 0.12, 0.08, 0.20, 0.05, 0.18, 0.10, 0.22, 0.07, 0.14],
        "fcf_yield": [0.08, 0.06, 0.03, 0.10, 0.02, 0.09, 0.04, 0.11, 0.01, 0.07],
        "gp_a":      [0.40, 0.35, 0.20, 0.50, 0.15, 0.45, 0.25, 0.55, 0.10, 0.38],
        "mos":       [0.30, 0.20, 0.10, 0.40, 0.05, 0.35, 0.15, 0.45, 0.02, 0.28],
        "nd_ebitda": [1.0,  2.0,  3.0,  0.5,  4.0,  1.5,  2.5,  0.0,  3.5,  1.8],
        "t3_liabilities_to_equity": [1.0, 2.0, 3.0, 0.5, 5.0, 1.5, 2.5, 0.3, 4.0, 1.8],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ============================================================
# GP/A integration in _baseline_quality_score
# ============================================================

class TestGPASubFactor:
    def test_gpa_improves_score_for_high_gpa_company(self):
        """Company with the highest GP/A in the cohort gets the highest quality sub-score."""
        df = _make_df()
        # Replace first company with very high GP/A, last with very low GP/A; rest the same
        df = df.copy()
        df["gp_a"] = [0.70, 0.40, 0.35, 0.38, 0.30, 0.42, 0.33, 0.28, 0.31, 0.10]
        score = _baseline_quality_score(df)
        # Company at index 0 (highest GP/A=0.70) should score higher than index 9 (lowest GP/A=0.10)
        # all else equal in this row range
        assert float(score.iloc[0]) > float(score.iloc[9])

    def test_missing_gpa_falls_back_to_roic_fcf(self):
        """When gp_a is all NaN, score should still be computed from ROIC + FCF."""
        df = _make_df()
        df["gp_a"] = np.nan
        score = _baseline_quality_score(df)
        assert score.notna().any()

    def test_gpa_weight_is_20pct(self):
        """GP/A weight in quality sub-group must be 0.20."""
        assert _QUALITY_SUBWEIGHTS["gp_a"] == pytest.approx(0.20)

    def test_roic_weight_reduced_from_60_to_50pct(self):
        """ROIC weight in quality sub-group should be 0.50 (was 0.60 before GP/A)."""
        assert _QUALITY_SUBWEIGHTS["roic"] == pytest.approx(0.50)

    def test_gpa_capped_at_100pct(self):
        """GP/A values > _GPA_CAP should not blow up the score."""
        df = _make_df()
        # One extreme outlier among varied values — should not produce inf/NaN for others
        df["gp_a"] = [5.0, 0.30, 0.25, 0.40, 0.20, 0.35, 0.28, 0.45, 0.18, 0.32]
        score = _baseline_quality_score(df)
        assert score.notna().all()
        assert np.all(np.isfinite(score.values))

    def test_gpa_present_vs_absent_gives_different_score(self):
        """Adding real GP/A data changes the quality score."""
        with_gpa    = _make_df()
        without_gpa = _make_df()
        without_gpa["gp_a"] = np.nan

        score_with    = _baseline_quality_score(with_gpa)
        score_without = _baseline_quality_score(without_gpa)

        # Scores won't be identical when GP/A adds signal
        assert not np.allclose(score_with.fillna(0), score_without.fillna(0), atol=1e-6)


# ============================================================
# Rebalanced weights
# ============================================================

class TestRebalancedWeights:
    def test_quality_weight_is_35pct(self):
        assert _STRATEGY_A_WEIGHTS_DEFAULT["quality"] == pytest.approx(0.35)

    def test_value_weight_is_30pct(self):
        assert _STRATEGY_A_WEIGHTS_DEFAULT["value"] == pytest.approx(0.30)

    def test_lowrisk_weight_is_20pct(self):
        assert _STRATEGY_A_WEIGHTS_DEFAULT["lowrisk"] == pytest.approx(0.20)

    def test_balance_weight_is_15pct(self):
        assert _STRATEGY_A_WEIGHTS_DEFAULT["balance"] == pytest.approx(0.15)

    def test_weights_sum_to_one(self):
        total = sum(_STRATEGY_A_WEIGHTS_DEFAULT.values())
        assert total == pytest.approx(1.0)

    def test_value_no_longer_dominates(self):
        """value weight must be ≤ quality weight (was 0.46 vs 0.27 before)."""
        assert _STRATEGY_A_WEIGHTS_DEFAULT["value"] <= _STRATEGY_A_WEIGHTS_DEFAULT["quality"]

    def test_composite_uses_custom_weights(self):
        """Passing custom weights overrides defaults."""
        df = _make_df()
        score_default = _strategy_a_composite_score(df)
        score_custom  = _strategy_a_composite_score(df, weights={"quality": 1.0, "value": 0.0, "lowrisk": 0.0, "balance": 0.0})
        # When quality weight is 1.0, composite == quality score
        quality_only = _baseline_quality_score(df)
        assert not np.allclose(
            score_default.fillna(0).values,
            score_custom.fillna(0).values,
            atol=1e-6,
        )


# ============================================================
# Factor correlation computation
# ============================================================

class TestFactorCorrelations:
    def _factor_scores(self) -> dict:
        n = 50
        rng = np.random.default_rng(42)
        return {
            "quality":  pd.Series(rng.normal(0, 1, n)),
            "value":    pd.Series(rng.normal(0, 1, n)),
            "lowrisk":  pd.Series(rng.normal(0, 1, n)),
            "balance":  pd.Series(rng.normal(0, 1, n)),
        }

    def test_returns_dataframe_with_correct_columns(self):
        corr = _compute_factor_correlations(self._factor_scores())
        assert set(corr.columns) == {"factor_a", "factor_b", "pearson_r"}

    def test_number_of_pairs(self):
        """4 factors → C(4,2) = 6 pairs."""
        corr = _compute_factor_correlations(self._factor_scores())
        assert len(corr) == 6

    def test_correlation_range(self):
        corr = _compute_factor_correlations(self._factor_scores())
        assert (corr["pearson_r"].abs() <= 1.0).all()

    def test_perfect_correlation_detected(self):
        """Perfectly correlated factors should show r ≈ 1.0."""
        x = pd.Series(np.linspace(0, 1, 50))
        scores = {"a": x, "b": x * 2.0 + 1.0}
        corr = _compute_factor_correlations(scores)
        assert len(corr) == 1
        assert abs(float(corr.iloc[0]["pearson_r"])) > 0.999

    def test_too_few_observations_skipped(self):
        """Pairs with < 10 valid overlapping observations are excluded."""
        scores = {
            "a": pd.Series([1.0, 2.0, np.nan] * 3),  # only 6 non-NaN
            "b": pd.Series([1.0, 2.0, 3.0] * 3),
        }
        corr = _compute_factor_correlations(scores)
        assert corr.empty

    def test_log_returns_json_string(self):
        result = _log_factor_correlations(self._factor_scores())
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert len(parsed) == 6  # 6 pairs

    def test_high_correlation_triggers_warning(self, caplog):
        """Pearson |r| ≥ threshold should log a WARNING."""
        x = pd.Series(np.linspace(0, 1, 50))
        scores = {"quality": x, "value": x + 0.01 * np.random.default_rng(1).normal(0, 1, 50)}
        with caplog.at_level(logging.WARNING):
            _log_factor_correlations(scores)
        assert any("High factor correlation" in r.message for r in caplog.records)

    def test_low_correlation_no_warning(self, caplog):
        """Independent factors should not trigger a WARNING."""
        rng = np.random.default_rng(99)
        scores = {
            "quality": pd.Series(rng.normal(0, 1, 100)),
            "value":   pd.Series(rng.normal(0, 1, 100)),
        }
        with caplog.at_level(logging.WARNING):
            _log_factor_correlations(scores)
        assert not any("High factor correlation" in r.message for r in caplog.records)


# ============================================================
# End-to-end: composite score changes with GP/A
# ============================================================

class TestCompositeScoreE2E:
    def test_composite_score_non_trivial(self):
        """Strategy A composite must not be constant (i.e., factors have dispersion)."""
        df = _make_df()
        score = _strategy_a_composite_score(df)
        assert score.std() > 0.01

    def test_composite_monotone_with_quality(self):
        """Adding one high-ROIC company to a pool → that company scores above median."""
        df = _make_df()
        # Replace last row with a clearly superior ROIC company
        df_target = df.copy()
        df_target.loc[9, "roic"] = 0.80   # well above the rest (max was 0.22)
        score = _strategy_a_composite_score(df_target)
        # The high-ROIC company (row 9) should score above the median
        assert float(score.iloc[9]) > float(score.median())

    def test_composite_monotone_with_mos(self):
        """Higher MoS → higher value factor → higher composite."""
        df = _make_df()
        df_rich = df.copy(); df_rich["mos"] = 0.60
        df_poor = df.copy(); df_poor["mos"] = 0.00
        s_rich = _strategy_a_composite_score(df_rich)
        s_poor = _strategy_a_composite_score(df_poor)
        assert float(s_rich.mean()) > float(s_poor.mean())
