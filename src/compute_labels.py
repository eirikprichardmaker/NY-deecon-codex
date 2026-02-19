from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fwd_return(prices: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    """
    prices: columns [yahoo_ticker, date, price]
    Returner: [yahoo_ticker, date, fwd_Xm_return]
    """
    prices = prices.sort_values(["yahoo_ticker", "date"]).copy()
    prices["target_date"] = prices["date"] + pd.DateOffset(months=horizon_months)

    # finn pris på/etter target_date pr ticker (direction='forward')
    future = pd.merge_asof(
        prices[["yahoo_ticker", "date", "target_date", "price"]].sort_values(["yahoo_ticker", "target_date"]),
        prices[["yahoo_ticker", "date", "price"]].rename(columns={"date": "future_date", "price": "future_price"}).sort_values(["yahoo_ticker", "future_date"]),
        by="yahoo_ticker",
        left_on="target_date",
        right_on="future_date",
        direction="forward",
        allow_exact_matches=True,
    )

    label = f"fwd_{horizon_months}m_return"
    future[label] = (future["future_price"] / future["price"]) - 1.0

    # DoD: leakage-guard (datoer)
    ok = future["future_date"].isna() | (future["future_date"] >= future["target_date"])
    if not bool(ok.all()):
        bad = future.loc[~ok, ["yahoo_ticker", "date", "target_date", "future_date"]].head(20)
        raise AssertionError(f"Leakage-assert feilet (future_date < target_date). Eksempel:\n{bad}")

    out = future[["yahoo_ticker", "date", label]]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--panel", default="data/processed/panel.parquet")
    p.add_argument("--prices", default="data/golden/prices_panel.parquet")
    p.add_argument("--horizon-months", type=int, default=12)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    panel = pd.read_parquet(Path(args.panel))
    prices = pd.read_parquet(Path(args.prices))

    prices["date"] = pd.to_datetime(prices["date"])
    if "price" not in prices.columns:
        if "adj_close" in prices.columns:
            prices = prices.rename(columns={"adj_close": "price"})
        elif "close" in prices.columns:
            prices = prices.rename(columns={"close": "price"})
        else:
            raise ValueError("prices må ha price/adj_close/close")

    labels = _fwd_return(prices[["yahoo_ticker", "date", "price"]], horizon_months=args.horizon_months)

    # join labels på panel (panel må ha yahoo_ticker + date/period_end)
    if "date" not in panel.columns:
        if "period_end" in panel.columns:
            panel = panel.rename(columns={"period_end": "date"})
        elif "date" in panel.columns:
            pass
        else:
            raise ValueError("panel må ha 'date' eller 'period_end'")

    panel["date"] = pd.to_datetime(panel["date"])
    out = panel.merge(labels, on=["yahoo_ticker", "date"], how="left")

    out_path = data_dir / "processed" / "panel_labeled.parquet"
    out.to_parquet(out_path, index=False)
    print(f"OK: skrev {out_path}")


if __name__ == "__main__":
    main()
