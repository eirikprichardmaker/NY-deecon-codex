from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _apply_index_technical_filter, _attach_relevant_index, _build_index_snapshot


def _mk_base_row(yahoo_ticker: str, country: str = "") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [yahoo_ticker.split(".", 1)[0] if "." in yahoo_ticker else yahoo_ticker],
            "yahoo_ticker": [yahoo_ticker],
            "info_country": [country],
        }
    )


def test_attach_relevant_index_maps_suffixes_and_country_fallback():
    df = pd.DataFrame(
        {
            "ticker": ["EQNR", "ABB", "MAERSK-A", "NDA-FI", "NOMAP"],
            "yahoo_ticker": ["EQNR.OL", "ABB.ST", "MAERSK-A.CO", "NDA-FI.HE", ""],
            "info_country": ["Norway", "Sweden", "Denmark", "Finland", "NO"],
        }
    )
    out = _attach_relevant_index(df, {})

    assert out["relevant_index_key"].tolist() == ["OSEAX", "OMXS", "OMXC25", "HEX", "OSEAX"]


def test_apply_index_filter_forces_fail_when_index_data_missing():
    base = _mk_base_row("EQNR.OL", "Norway")
    out = _apply_index_technical_filter(
        base,
        prices_df=pd.DataFrame(),
        asof="2026-02-16",
        dec_cfg={},
        mad_min=-0.05,
    )

    assert bool(out.loc[0, "index_data_ok"]) is False
    assert bool(out.loc[0, "index_ma200_ok"]) is False
    assert bool(out.loc[0, "index_tech_ok"]) is False


def test_apply_index_filter_passes_when_index_above_ma200_and_mad_ok():
    base = _mk_base_row("EQNR.OL", "Norway")
    dates = pd.date_range("2025-01-01", periods=260, freq="D")
    prices = pd.DataFrame(
        {
            "ticker": ["^OSEAX"] * len(dates),
            "date": dates,
            "adj_close": [100.0 + i for i in range(len(dates))],
        }
    )

    out = _apply_index_technical_filter(
        base,
        prices_df=prices,
        asof="2026-02-16",
        dec_cfg={"require_index_ma200": True, "require_index_mad": True},
        mad_min=-0.05,
    )

    assert bool(out.loc[0, "index_data_ok"]) is True
    assert bool(out.loc[0, "index_ma200_ok"]) is True
    assert bool(out.loc[0, "index_mad_ok"]) is True
    assert bool(out.loc[0, "index_tech_ok"]) is True


def test_apply_index_filter_blocks_when_index_trend_is_weak():
    base = _mk_base_row("EQNR.OL", "Norway")
    dates = pd.date_range("2025-01-01", periods=260, freq="D")
    prices = pd.DataFrame(
        {
            "ticker": ["^OSEAX"] * len(dates),
            "date": dates,
            "adj_close": [500.0 - i for i in range(len(dates))],
        }
    )

    out = _apply_index_technical_filter(
        base,
        prices_df=prices,
        asof="2026-02-16",
        dec_cfg={"require_index_ma200": True, "require_index_mad": True},
        mad_min=-0.05,
    )

    assert bool(out.loc[0, "index_data_ok"]) is True
    assert bool(out.loc[0, "index_tech_ok"]) is False
    assert (bool(out.loc[0, "index_ma200_ok"]) is False) or (bool(out.loc[0, "index_mad_ok"]) is False)


def test_build_index_snapshot_uses_price_over_ma200_not_input_flag():
    dates = pd.date_range("2025-01-01", periods=260, freq="D")
    prices = pd.DataFrame(
        {
            "ticker": ["^OSEAX"] * len(dates),
            "date": dates,
            "adj_close": [100.0 + i for i in range(len(dates))],
            "ma200": [9999.0] * len(dates),
            "mad": [-1.0] * len(dates),
            "above_ma200": [False] * len(dates),
        }
    )

    snap = _build_index_snapshot(prices_df=prices, asof="2026-02-16", index_keys=["OSEAX"])
    assert len(snap) == 1
    last = snap.iloc[0]
    assert bool(last["index_above_ma200"]) is True
    assert float(last["index_ma200"]) < float(last["index_price"])


def test_apply_index_filter_treats_warmup_as_missing_data():
    base = _mk_base_row("EQNR.OL", "Norway")
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    prices = pd.DataFrame(
        {
            "ticker": ["^OSEAX"] * len(dates),
            "date": dates,
            "adj_close": [100.0 + i for i in range(len(dates))],
        }
    )

    out = _apply_index_technical_filter(
        base,
        prices_df=prices,
        asof="2026-02-16",
        dec_cfg={"require_index_ma200": True, "require_index_mad": True},
        mad_min=-0.05,
    )

    assert bool(out.loc[0, "index_data_ok"]) is False
    assert bool(out.loc[0, "index_ma200_ok"]) is False
    assert bool(out.loc[0, "index_tech_ok"]) is False
