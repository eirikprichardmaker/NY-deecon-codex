from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src import compute_factors


class _Log:
    def info(self, _msg: str) -> None:
        return


def test_compute_factors_fcf_yield_aligns_units_between_fcf_and_ev(tmp_path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    master = pd.DataFrame(
        {
            "ticker": ["SANION"],
            "yahoo_ticker": ["SANION.ST"],
            "market_cap_-_current": [2968.0],   # millions
            "net_debt_-_current": [-88.0],      # millions
            "fcf_-_millions": [574.3],          # millions
            "roic_-_current": [10.0],
            "ev/ebit_-_current": [3.8],
            "n.debt/ebitda_-_current": [-1.1],
        }
    )
    master.to_parquet(processed / "master.parquet", index=False)

    ctx = SimpleNamespace(
        cfg={
            "paths": {
                "processed_dir": "data/processed",
            }
        },
        project_root=tmp_path,
    )

    rc = compute_factors.run(ctx, _Log())
    assert rc == 0

    out = pd.read_parquet(processed / "master_factors.parquet")
    ev_m = 2968.0 - 88.0
    expected = 574.3 / ev_m
    got = float(out.loc[0, "fcf_yield"])
    assert abs(got - expected) < 1e-12
