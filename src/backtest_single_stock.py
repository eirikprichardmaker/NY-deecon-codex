from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Weights:
    feature_cols: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    label: str


def _load_weights(path: Path) -> Weights:
    d = json.loads(path.read_text(encoding="utf-8"))
    return Weights(
        feature_cols=d["feature_cols"],
        scaler_mean=np.array(d["scaler_mean"], dtype=float),
        scaler_scale=np.array(d["scaler_scale"], dtype=float),
        coef=np.array(d["coef"], dtype=float),
        intercept=float(d["intercept"]),
        label=d["label"],
    )


def _predict_score(df: pd.DataFrame, w: Weights) -> np.ndarray:
    X = df[w.feature_cols].astype(float).values
    Xs = (X - w.scaler_mean) / np.where(w.scaler_scale == 0, 1.0, w.scaler_scale)
    return Xs @ w.coef + w.intercept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/panel_labeled.parquet")
    p.add_argument("--weights", required=True, help="runs/<run_id>/regression_weights.json")
    p.add_argument("--date-col", default="date")
    p.add_argument("--rebalance", choices=["M", "Q"], default="M")
    p.add_argument("--tc-bps", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()

    np.random.seed(args.seed)

    df = pd.read_parquet(Path(args.input))
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    w = _load_weights(Path(args.weights))

    # Forvent at df har price + evt mcap etc. Vi scorer på feature_cols.
    missing = [c for c in w.feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Mangler feature-kolonner i panel: {missing}")

    df = df.dropna(subset=w.feature_cols + ["price"]).copy()
    df["score"] = _predict_score(df, w)

    # Rebalance-datoer: månedsslutt/kvartalsslutt basert på siste observasjon pr ticker pr periode
    df["period"] = df[args.date_col].dt.to_period("M" if args.rebalance == "M" else "Q")
    snap = (
        df.sort_values([args.date_col])
          .groupby(["period", "yahoo_ticker"], as_index=False)
          .tail(1)
          .copy()
    )

    # Enkelt “hold cash hvis ingen”: topp-score må være > 0
    # (Bytt senere til din faktiske decision.py-regel + filters.)
    trades = []
    equity = []
    nav = 1.0
    last_ticker = None

    # bygg prisserie for avkastning mellom rebal-datoer
    snap = snap.sort_values([args.date_col])

    periods = sorted(snap["period"].unique().tolist())
    for i, per in enumerate(periods[:-1]):
        today = snap[snap["period"] == per]
        nxt = snap[snap["period"] == periods[i + 1]]

        cand = today.sort_values("score", ascending=False).head(1)
        if cand.empty or float(cand["score"].iloc[0]) <= 0:
            # hold cash
            equity.append({"period": str(per), "date": str(today[args.date_col].max().date()), "position": "CASH", "nav": nav})
            trades.append({"period": str(per), "action": "HOLD_CASH", "ticker": "", "tc": 0.0})
            last_ticker = None
            continue

        ticker = str(cand["yahoo_ticker"].iloc[0])
        entry_price = float(cand["price"].iloc[0])

        # finn exit price på neste rebalance (samme ticker)
        nxt_row = nxt[nxt["yahoo_ticker"] == ticker]
        if nxt_row.empty:
            # hvis ikke finnes, hold cash denne perioden
            equity.append({"period": str(per), "date": str(today[args.date_col].max().date()), "position": "CASH", "nav": nav})
            trades.append({"period": str(per), "action": "HOLD_CASH_NO_NEXT", "ticker": ticker, "tc": 0.0})
            last_ticker = None
            continue

        exit_price = float(nxt_row["price"].iloc[0])
        gross_ret = (exit_price / entry_price) - 1.0

        # transaksjonskost: betales ved bytte inn/ut (enkelt: 2 * tc)
        tc = (args.tc_bps / 10000.0)
        roundtrip_tc = 2 * tc if last_ticker != ticker else 0.0  # samme posisjon = ingen ny kost
        net_ret = (1 + gross_ret) * (1 - roundtrip_tc) - 1

        nav *= (1 + net_ret)

        action = "BUY" if last_ticker is None else ("HOLD" if last_ticker == ticker else "SWITCH")
        trades.append({"period": str(per), "action": action, "ticker": ticker, "entry_price": entry_price, "exit_price": exit_price, "gross_ret": gross_ret, "tc": roundtrip_tc, "net_ret": net_ret})
        equity.append({"period": str(per), "date": str(today[args.date_col].max().date()), "position": ticker, "nav": nav})

        last_ticker = ticker

    run_dir = Path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trades_path = run_dir / "backtest_trades.csv"
    equity_path = run_dir / "backtest_equity.csv"
    report_path = run_dir / "backtest_report.md"

    pd.DataFrame(trades).to_csv(trades_path, index=False)
    pd.DataFrame(equity).to_csv(equity_path, index=False)

    rep = f"""# Backtest report

run_id: {args.run_id}
rebalance: {args.rebalance}
tc_bps: {args.tc_bps}
rule: top score > 0 else cash (placeholder til decision.py)

Outputs:
- backtest_trades.csv
- backtest_equity.csv
"""
    report_path.write_text(rep, encoding="utf-8")

    print(f"OK: {trades_path}")
    print(f"OK: {equity_path}")
    print(f"OK: {report_path}")


if __name__ == "__main__":
    main()
