# tests/test_contracts.py
from pathlib import Path
import pandas as pd

REQUIRED = [
    "yahoo_ticker", "Company", "Info - Country",
    "Market Cap - Current", "OCF - Millions", "Capex - Millions", "FCF - Millions",
    "Net Debt - Current", "N.Debt/Ebitda - Current", "ROIC - Current", "EV/EBIT - Current",
    "P/E - Current", "P/S - Current",
    "price_date", "price", "ma200", "mad", "above_ma200",
    "fundamental_ok", "technical_ok", "decision",
]

def test_master_contract():
    repo = Path(__file__).resolve().parents[1]
    p = repo / "data/latest/master.parquet"
    assert p.exists(), "Run: python src/pipeline.py run"
    df = pd.read_parquet(p)

    assert df["yahoo_ticker"].notna().all()
    assert not df["yahoo_ticker"].duplicated().any()

    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    forbidden = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    assert not forbidden, f"Forbidden merge leftovers: {forbidden}"

    assert df["decision"].isin(["CANDIDATE","WAIT","HOLD"]).all()
