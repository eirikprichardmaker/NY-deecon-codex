from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_master import (
    _build_dividend_feature_snapshot,
    _derive_dividend_features,
    _latest_raw_snapshot_dir,
)


def test_latest_raw_snapshot_dir_respects_asof(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "2026-02-16").mkdir(parents=True)
    (raw / "2026-02-23").mkdir(parents=True)
    (raw / "2026-02-25").mkdir(parents=True)

    got = _latest_raw_snapshot_dir(raw, "2026-02-24")
    assert got is not None
    assert got.name == "2026-02-23"

    none_case = _latest_raw_snapshot_dir(raw, "2026-02-10")
    assert none_case is None


def test_derive_dividend_features_from_reports():
    reports_y = pd.DataFrame(
        {
            "year": [2019, 2020, 2021, 2022, 2023, 2024],
            "period": [5, 5, 5, 5, 5, 5],
            "dividend": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
            "profit_To_Equity_Holders": [50.0, 55.0, 60.0, 70.0, 80.0, 95.0],
            "earnings_Per_Share": [2.0, 2.4, 2.8, 3.2, 3.6, 4.0],
            "number_Of_Shares": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "stock_Price_Average": [50.0, 52.0, 53.0, 54.0, 55.0, 50.0],
            "report_Date": [
                "2020-02-01",
                "2021-02-01",
                "2022-02-01",
                "2023-02-01",
                "2024-02-01",
                "2025-02-01",
            ],
        }
    )

    reports_r12 = pd.DataFrame(
        {
            "year": [2025],
            "period": [4],
            "revenues": [1000.0],
            "profit_To_Equity_Holders": [100.0],
            "cash_Flow_From_Operating_Activities": [120.0],
            "free_Cash_Flow": [80.0],
            "total_Assets": [2000.0],
            "report_Date": ["2026-02-01"],
        }
    )

    out = _derive_dividend_features(reports_y=reports_y, reports_r12=reports_r12, asof_dt=pd.Timestamp("2026-02-24"))

    assert int(out["dividend_history_years"]) == 6
    assert int(out["dividend_paid_years"]) == 6
    assert abs(float(out["payout_ratio"]) - 0.5) < 1e-12
    assert abs(float(out["dividend_yield"]) - 0.04) < 1e-12
    assert abs(float(out["fcf_margin"]) - 0.08) < 1e-12
    assert abs(float(out["ocf_margin"]) - 0.12) < 1e-12
    assert abs(float(out["profit_margin"]) - 0.10) < 1e-12
    assert abs(float(out["roa"]) - 0.05) < 1e-12
    assert np.isfinite(float(out["dividend_growth_5y"]))
    assert np.isfinite(float(out["profit_growth_5y"]))
    assert np.isfinite(float(out["share_count_growth_5y"]))


def test_build_dividend_feature_snapshot_reads_raw_reports(tmp_path: Path):
    raw = tmp_path / "raw" / "2026-02-23"
    y_dir = raw / "reports_y" / "market=NO"
    r12_dir = raw / "reports_r12" / "market=NO"
    y_dir.mkdir(parents=True)
    r12_dir.mkdir(parents=True)

    reports_y = pd.DataFrame(
        {
            "year": [2023, 2024],
            "period": [5, 5],
            "dividend": [1.5, 2.0],
            "earnings_Per_Share": [3.0, 4.0],
            "number_Of_Shares": [100.0, 101.0],
            "stock_Price_Average": [45.0, 50.0],
            "report_Date": ["2024-02-01", "2025-02-01"],
        }
    )
    reports_r12 = pd.DataFrame(
        {
            "year": [2025],
            "period": [4],
            "revenues": [1000.0],
            "profit_To_Equity_Holders": [80.0],
            "cash_Flow_From_Operating_Activities": [90.0],
            "free_Cash_Flow": [70.0],
            "total_Assets": [1600.0],
            "report_Date": ["2026-01-31"],
        }
    )

    reports_y.to_parquet(y_dir / "ins_id=1.parquet", index=False)
    reports_r12.to_parquet(r12_dir / "ins_id=1.parquet", index=False)

    master = pd.DataFrame({"ticker": ["AAA"], "yahoo_ticker": ["AAA.OL"]})
    ins_map = pd.DataFrame(
        {
            "yahoo_ticker": ["AAA.OL"],
            "yahoo_key": ["AAA.OL"],
            "ticker_norm": ["AAA"],
            "ins_id": [1],
            "market": ["NO"],
        }
    )

    snap = _build_dividend_feature_snapshot(
        master=master,
        raw_snapshot_dir=raw,
        asof_dt=pd.Timestamp("2026-02-24"),
        ins_map=ins_map,
    )

    assert len(snap) == 1
    row = snap.iloc[0]
    assert str(row["yahoo_ticker"]) == "AAA.OL"
    assert np.isfinite(float(row["payout_ratio"]))
    assert np.isfinite(float(row["dividend_yield"]))
    assert np.isfinite(float(row["profit_margin"]))
