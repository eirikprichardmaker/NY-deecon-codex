from __future__ import annotations

import numpy as np
import pandas as pd

from src.stress_test_markets import _build_static_proxy, _fetch_true_market_instrument_ids, _latest_kpi_value


def test_latest_kpi_value_handles_numpy_array_and_prefers_annual() -> None:
    values = np.array(
        [
            {"y": 2025, "p": 3, "v": 1.0},
            {"y": 2024, "p": 5, "v": 2.5},
            {"y": 2023, "p": 5, "v": 2.0},
        ],
        dtype=object,
    )
    got = _latest_kpi_value(values)
    assert got == 2.5


def test_build_static_proxy_produces_core_fields() -> None:
    kpi_snap = pd.DataFrame(
        {
            "ins_id": [101],
            "10": [8.0],    # ev/ebit
            "42": [1.2],    # nd/ebitda
            "33": [18.0],   # roe (% proxy for roic)
            "54": [500.0],  # ebitda (MCURR)
            "68": [12.0],   # ocf/share
            "71": [10.0],   # ebitda/share
            "73": [20.0],   # net debt/share
        }
    )
    latest_price = pd.DataFrame({"ins_id": [101], "latest_price": [100.0]})

    out = _build_static_proxy(market="DE", kpi_snap=kpi_snap, latest_price=latest_price, wacc=0.09)
    assert len(out) == 1

    row = out.iloc[0]
    assert row["ticker"] == "I101.DE"
    assert float(row["market_cap"]) > 0.0
    assert np.isfinite(float(row["mos"]))
    assert np.isfinite(float(row["fcf_yield"]))
    assert np.isfinite(float(row["roic"]))
    assert bool(row["value_creation_ok_base"]) is True


def test_fetch_true_market_instrument_ids_maps_marketid(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "instruments": [
                    {"insId": 100, "countryId": 8, "instrument": 0},   # DE equity
                    {"insId": 101, "countryId": 9, "instrument": 0},   # FR equity
                    {"insId": 102, "countryId": 8, "instrument": 2},   # DE index -> filtered out
                    {"insId": 103, "countryId": 4, "instrument": 0},   # DK equity
                    {"insId": "bad", "countryId": 8, "instrument": 0},
                ]
            }

    def _fake_get(*args, **kwargs):
        return _Resp()

    monkeypatch.setenv("BORSDATA_AUTHKEY", "dummy")
    monkeypatch.setattr("src.stress_test_markets.requests.get", _fake_get)

    out = _fetch_true_market_instrument_ids(["DE", "FR"])
    assert out["DE"] == {100}
    assert out["FR"] == {101}
