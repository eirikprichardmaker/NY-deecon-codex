from __future__ import annotations

import numpy as np
import pandas as pd

from src.stress_test_yahoo_markets import _build_static, _extract_wiki_tickers


def test_extract_wiki_tickers_adds_suffix_and_dedups() -> None:
    html = """
    <html><body>
    <table class="wikitable">
      <tr><th>Company</th><th>Ticker symbol</th></tr>
      <tr><td>A</td><td>SAP</td></tr>
      <tr><td>B</td><td>BMW.DE</td></tr>
      <tr><td>C</td><td>SAP</td></tr>
    </table>
    </body></html>
    """
    out = _extract_wiki_tickers(html, default_suffix="DE")
    assert out == ["SAP.DE", "BMW.DE"]


def test_build_static_keeps_required_columns() -> None:
    fdf = pd.DataFrame(
        {
            "ticker": ["SAP.DE"],
            "market_cap": [100.0],
            "mos": [0.2],
            "high_risk_flag": [False],
            "quality_weak_count": [1],
            "value_creation_ok_base": [True],
            "roic": [0.15],
            "fcf_yield": [0.05],
            "ev_ebit": [12.0],
            "nd_ebitda": [1.0],
            "intrinsic_equity": [120.0],
        }
    )
    out = _build_static(fdf, market="DE")
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "SAP.DE"
    assert out.iloc[0]["relevant_index_symbol"] == "^GDAXI"
    assert np.isfinite(float(out.iloc[0]["mos"]))
