"""
STEG 1 — Point-in-Time (PIT) fundamentals for WFT.
Kontrakt: ingen fremtidige facts lekker inn i historiske fold-måneder.
"""
import numpy as np
import pandas as pd
import pytest

from src.wft import (
    _pit_fundamentals_for_fold,
    _apply_pit_override,
    _find_fundamentals_history,
    _PIT_OVERRIDE_COLS,
)


# ============================================================
# Helpers
# ============================================================

def _make_hist(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal fundamentals_history DataFrame."""
    base = {
        "yahoo_ticker": "TEST.OL",
        "ticker": "TEST.OL",
        "ins_id": 1,
        "company": "Test Co",
        "country": "Norway",
        "sector": "Industrials",
        "report_type": "annual",
        "price_type": "last",
        "kpi_id": 0,
    }
    records = []
    for r in rows:
        rec = dict(base)
        rec.update(r)
        records.append(rec)
    return pd.DataFrame(records)


def _fold_ts(year: int, month: int = 12) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=31)


# ============================================================
# _find_fundamentals_history
# ============================================================

class TestFindFundamentalsHistory:
    def test_returns_none_for_empty_dir(self, tmp_path):
        assert _find_fundamentals_history(tmp_path) is None

    def test_finds_file_in_subdir(self, tmp_path):
        sub = tmp_path / "2026-03-14"
        sub.mkdir()
        p = sub / "fundamentals_history.parquet"
        pd.DataFrame({"x": [1]}).to_parquet(p)
        result = _find_fundamentals_history(tmp_path)
        assert result == p

    def test_returns_most_recent_subdir(self, tmp_path):
        for d in ["2024-01-01", "2026-03-14", "2025-06-30"]:
            sub = tmp_path / d
            sub.mkdir()
            (sub / "fundamentals_history.parquet").write_bytes(
                pd.DataFrame({"d": [d]}).to_parquet(None)  # type: ignore
            )
        # Sorted reverse → 2026-03-14 first
        result = _find_fundamentals_history(tmp_path)
        assert result is not None
        assert "2026" in str(result)


# ============================================================
# _pit_fundamentals_for_fold — core PIT contract
# ============================================================

class TestPITFundamentalsForFold:
    def test_excludes_observations_after_cutoff(self):
        """Facts with date > fold_ts - 90d must NOT appear in result."""
        fold_ts = _fold_ts(2020, 12)
        cutoff = fold_ts - pd.Timedelta(days=90)

        hist = _make_hist([
            # Safe: well before cutoff
            {"yahoo_ticker": "SAFE.OL", "metric": "roic",  "value": 0.15, "date": "2020-06-30"},
            {"yahoo_ticker": "SAFE.OL", "metric": "fcf_m", "value": 50.0, "date": "2020-06-30"},
            {"yahoo_ticker": "SAFE.OL", "metric": "mcap_m","value": 300.0,"date": "2020-06-30"},
            # Leak: dated AFTER cutoff (future fact)
            {"yahoo_ticker": "LEAK.OL", "metric": "roic",  "value": 0.30, "date": str((fold_ts + pd.Timedelta(days=1)).date())},
            {"yahoo_ticker": "LEAK.OL", "metric": "fcf_m", "value": 99.0, "date": str((fold_ts + pd.Timedelta(days=1)).date())},
        ])

        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})

        k_values = result["k"].tolist()
        assert "SAFE" in k_values, "SAFE.OL should appear"
        assert "LEAK" not in k_values, "Future fact LEAK.OL must be excluded"

    def test_exactly_on_cutoff_is_excluded(self):
        """date == fold_ts - 90d is the boundary; it IS included (<=)."""
        fold_ts = _fold_ts(2020, 12)
        cutoff = fold_ts - pd.Timedelta(days=90)

        hist = _make_hist([
            {"yahoo_ticker": "BOUNDARY.OL", "metric": "roic",  "value": 0.10, "date": str(cutoff.date())},
            {"yahoo_ticker": "BOUNDARY.OL", "metric": "fcf_m", "value": 20.0, "date": str(cutoff.date())},
            {"yahoo_ticker": "BOUNDARY.OL", "metric": "mcap_m","value": 100.0,"date": str(cutoff.date())},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})
        assert "BOUNDARY" in result["k"].tolist()

    def test_one_day_after_cutoff_is_excluded(self):
        fold_ts = _fold_ts(2020, 12)
        cutoff = fold_ts - pd.Timedelta(days=90)
        future_date = cutoff + pd.Timedelta(days=1)

        hist = _make_hist([
            {"yahoo_ticker": "FUTURE.OL", "metric": "roic", "value": 0.10, "date": str(future_date.date())},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})
        assert result.empty or "FUTURE" not in result["k"].tolist()

    def test_returns_empty_when_no_data_before_cutoff(self):
        fold_ts = _fold_ts(2015, 1)  # Very early — no data yet
        hist = _make_hist([
            {"yahoo_ticker": "TEST.OL", "metric": "roic", "value": 0.15, "date": "2020-01-01"},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})
        assert result.empty

    def test_uses_most_recent_observation_before_cutoff(self):
        """When multiple observations exist, the latest one before cutoff wins."""
        fold_ts = _fold_ts(2021, 12)
        hist = _make_hist([
            {"yahoo_ticker": "MULTI.OL", "metric": "roic", "value": 0.10, "date": "2019-12-31"},
            {"yahoo_ticker": "MULTI.OL", "metric": "roic", "value": 0.25, "date": "2021-06-30"},  # newer, before cutoff
            {"yahoo_ticker": "MULTI.OL", "metric": "fcf_m", "value": 40.0, "date": "2021-06-30"},
            {"yahoo_ticker": "MULTI.OL", "metric": "mcap_m","value": 200.0, "date": "2021-06-30"},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})
        row = result[result["k"] == "MULTI"].iloc[0]
        assert abs(float(row["roic"]) - 0.25) < 0.01, "Should use latest pre-cutoff value"

    def test_positive_fcf_produces_mos(self):
        """Positive FCF + positive mcap → non-NaN MoS."""
        fold_ts = _fold_ts(2020, 12)
        hist = _make_hist([
            {"yahoo_ticker": "GOOD.OL", "metric": "fcf_m",  "value": 50.0,  "date": "2020-06-30"},
            {"yahoo_ticker": "GOOD.OL", "metric": "mcap_m", "value": 300.0, "date": "2020-06-30"},
            {"yahoo_ticker": "GOOD.OL", "metric": "netdebt_m", "value": 20.0, "date": "2020-06-30"},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={"GOOD": 0.09})
        assert len(result) == 1
        mos = float(result.iloc[0]["mos"])
        assert np.isfinite(mos), "MoS should be finite for valid FCF/mcap"

    def test_negative_fcf_produces_nan_mos(self):
        fold_ts = _fold_ts(2020, 12)
        hist = _make_hist([
            {"yahoo_ticker": "BAD.OL", "metric": "fcf_m",  "value": -10.0, "date": "2020-06-30"},
            {"yahoo_ticker": "BAD.OL", "metric": "mcap_m", "value": 200.0, "date": "2020-06-30"},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})
        assert len(result) == 1
        assert not np.isfinite(result.iloc[0]["mos"]), "Negative FCF → NaN MoS"

    def test_output_has_required_columns(self):
        fold_ts = _fold_ts(2020, 12)
        hist = _make_hist([
            {"yahoo_ticker": "A.OL", "metric": "roic", "value": 0.12, "date": "2020-06-30"},
        ])
        result = _pit_fundamentals_for_fold(hist, fold_ts, wacc_by_k={})
        required = {"k", "mos", "roic", "fcf_yield", "high_risk_flag",
                    "quality_weak_count", "value_creation_ok_base"}
        assert required.issubset(set(result.columns))

    def test_different_folds_see_different_data(self):
        """Earlier fold must not see data that only became available after its cutoff."""
        hist = _make_hist([
            # Only visible to fold 2020+ (pub_lag 90d → cutoff 2019-10-02 for fold 2020-01-01)
            {"yahoo_ticker": "LATER.OL", "metric": "roic",  "value": 0.20, "date": "2019-12-31"},
            {"yahoo_ticker": "LATER.OL", "metric": "fcf_m", "value": 30.0, "date": "2019-12-31"},
            # Visible to both
            {"yahoo_ticker": "LATER.OL", "metric": "roic",  "value": 0.10, "date": "2018-12-31"},
            {"yahoo_ticker": "LATER.OL", "metric": "fcf_m", "value": 20.0, "date": "2018-12-31"},
        ])
        early_fold = pd.Timestamp("2019-06-30")  # cutoff = 2019-04-01 → can't see 2019-12-31
        late_fold  = pd.Timestamp("2020-06-30")  # cutoff = 2020-04-01 → can see 2019-12-31

        early_result = _pit_fundamentals_for_fold(hist, early_fold, wacc_by_k={})
        late_result  = _pit_fundamentals_for_fold(hist, late_fold,  wacc_by_k={})

        assert not early_result.empty, "Early fold should see 2018 data"
        assert not late_result.empty,  "Late fold should see 2019 data"

        early_roic = float(early_result[early_result["k"] == "LATER"].iloc[0]["roic"])
        late_roic  = float(late_result[late_result["k"] == "LATER"].iloc[0]["roic"])
        assert abs(early_roic - 0.10) < 0.01, "Early fold should use 2018 ROIC"
        assert abs(late_roic  - 0.20) < 0.01, "Late fold should use 2019 ROIC"


# ============================================================
# _apply_pit_override
# ============================================================

class TestApplyPITOverride:
    def _make_month_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "k": "ABC",
            "ticker": "ABC.OL",
            "adj_close": 100.0,
            "month": pd.Timestamp("2020-12-31"),
            "mos": 0.05,          # stale static value
            "roic": 0.08,
            "fcf_yield": 0.03,
            "high_risk_flag": False,
            "quality_weak_count": 1,
            "value_creation_ok_base": False,
            "market_cap": 1_000_000.0,
        }])

    def test_override_replaces_fundamental_columns(self):
        month_df = self._make_month_df()
        pit = pd.DataFrame([{
            "k": "ABC",
            "mos": 0.50,
            "roic": 0.20,
            "fcf_yield": 0.10,
            "high_risk_flag": False,
            "quality_weak_count": 0,
            "value_creation_ok_base": True,
            "market_cap": 900_000.0,
        }])
        result = _apply_pit_override(month_df, pit)
        assert float(result.iloc[0]["mos"]) == pytest.approx(0.50)
        assert float(result.iloc[0]["roic"]) == pytest.approx(0.20)
        assert bool(result.iloc[0]["value_creation_ok_base"]) is True

    def test_non_fundamental_columns_preserved(self):
        month_df = self._make_month_df()
        pit = pd.DataFrame([{"k": "ABC", "mos": 0.40}])
        result = _apply_pit_override(month_df, pit)
        assert "adj_close" in result.columns
        assert float(result.iloc[0]["adj_close"]) == pytest.approx(100.0)

    def test_unknown_ticker_gets_nan_fundamentals(self):
        month_df = self._make_month_df()
        pit = pd.DataFrame([{"k": "OTHER", "mos": 0.40}])
        result = _apply_pit_override(month_df, pit)
        # ABC not in pit → mos should be NaN after override
        assert pd.isna(result.iloc[0]["mos"])

    def test_empty_pit_returns_unchanged(self):
        month_df = self._make_month_df()
        result = _apply_pit_override(month_df, pd.DataFrame())
        pd.testing.assert_frame_equal(result, month_df)

    def test_none_pit_returns_unchanged(self):
        month_df = self._make_month_df()
        result = _apply_pit_override(month_df, None)  # type: ignore
        pd.testing.assert_frame_equal(result, month_df)
