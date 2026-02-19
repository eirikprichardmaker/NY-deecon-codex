from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transform_prices import INDEX_TICKERS, _append_missing_indices


class _Log:
    def info(self, *_args, **_kwargs):
        return None


def _base_prices() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=220, freq="D")
    return pd.DataFrame(
        {
            "date": list(dates),
            "ticker": ["EQNR.OL"] * len(dates),
            "adj_close": [100.0 + i for i in range(len(dates))],
            "volume": [1000] * len(dates),
        }
    )


def test_append_missing_indices_skips_fetch_when_all_indices_present():
    base = _base_prices()
    dates = pd.date_range("2025-01-01", periods=220, freq="D")
    rows = []
    for idx in INDEX_TICKERS:
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": [idx] * len(dates),
                    "adj_close": [200.0 + i for i in range(len(dates))],
                    "volume": [1000] * len(dates),
                }
            )
        )
    df = pd.concat([base] + rows, ignore_index=True)

    def _fetcher(_missing, _start, _end, _log):
        raise AssertionError("fetcher should not be called when indices already exist")

    out = _append_missing_indices(df, asof="2026-02-16", log=_Log(), fetcher=_fetcher)
    assert set(INDEX_TICKERS).issubset(set(out["ticker"].unique().tolist()))
    assert {"ma21", "ma200", "mad", "above_ma200"}.issubset(set(out.columns))


def test_append_missing_indices_appends_fetched_rows_when_missing():
    base = _base_prices()

    def _fetcher(missing, _start, _end, _log):
        assert set(missing) == set(INDEX_TICKERS)
        dates = pd.date_range("2025-01-01", periods=220, freq="D")
        rows = []
        for idx in missing:
            rows.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "ticker": [idx] * len(dates),
                        "adj_close": [300.0 + i for i in range(len(dates))],
                        "volume": [1000] * len(dates),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    out = _append_missing_indices(base, asof="2026-02-16", log=_Log(), fetcher=_fetcher)

    for idx in INDEX_TICKERS:
        assert idx in set(out["ticker"].unique().tolist())

    idx_rows = out[out["ticker"] == "^OSEAX"].sort_values("date")
    assert len(idx_rows) >= 200
    assert pd.notna(idx_rows["ma200"].iloc[-1])
