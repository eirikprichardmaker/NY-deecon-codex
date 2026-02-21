from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _write_ma200_sanity_report


def test_ma200_sanity_report_writes_expected_ratios(tmp_path):
    df = pd.DataFrame(
        {
            "adj_close": [10.0, 10.0, None],
            "ma200": [9.0, None, 8.0],
            "relevant_index_key": ["OSEAX", "OSEAX", ""],
            "index_price": [100.0, 90.0, None],
            "index_ma200": [95.0, None, None],
        }
    )

    _write_ma200_sanity_report(tmp_path, "2026-02-21", df)

    csv_path = tmp_path / "ma200_sanity_report.csv"
    md_path = tmp_path / "ma200_sanity_report.md"
    assert csv_path.exists()
    assert md_path.exists()

    out = pd.read_csv(csv_path)
    stock_nan = out[(out["scope"] == "stock") & (out["metric"] == "ma200_nan_rows")].iloc[0]
    assert int(stock_nan["numerator"]) == 1
    assert int(stock_nan["denominator"]) == 3

    stock_gt = out[(out["scope"] == "stock") & (out["metric"] == "price_gt_ma200_rows")].iloc[0]
    assert int(stock_gt["numerator"]) == 1
    assert int(stock_gt["denominator"]) == 1

    bench_nan = out[(out["scope"] == "benchmark") & (out["metric"] == "ma200_nan_rows")].iloc[0]
    assert int(bench_nan["numerator"]) == 1
    assert int(bench_nan["denominator"]) == 2

    bench_gt = out[(out["scope"] == "benchmark") & (out["metric"] == "price_gt_ma200_rows")].iloc[0]
    assert int(bench_gt["numerator"]) == 1
    assert int(bench_gt["denominator"]) == 1

