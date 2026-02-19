from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

TICKER_CANDIDATES = ["yahoo_ticker", "symbol", "ticker", "yahoo", "yahoosymbol", "instrument", "secid"]
DATE_CANDIDATES   = ["date", "datetime", "timestamp", "price_date"]
PRICE_CANDIDATES  = ["adj_close", "adjclose", "adjusted_close", "adjustedclose", "close", "price", "last"]
VOLUME_CANDIDATES = ["volume", "vol"]

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def _looks_like_wide_price_matrix(df: pd.DataFrame) -> bool:
    # Heuristikk: har en date-kolonne eller datetime-index + mange "ticker-ish" kolonner
    # og ingen eksplisitt ticker-kolonne.
    ticker_col = _pick_col(df, TICKER_CANDIDATES)
    if ticker_col is not None:
        return False

    date_col = _pick_col(df, DATE_CANDIDATES)
    if date_col is not None:
        non_date_cols = [c for c in df.columns if c != date_col]
        return len(non_date_cols) >= 10
    # hvis index er datetime
    if isinstance(df.index, pd.DatetimeIndex):
        return df.shape[1] >= 10
    return False

def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"data\golden\prices_panel.parquet")
    ap.add_argument("--output", default=r"data\golden\prices_panel.parquet")
    ap.add_argument("--force-price-col", default="", help="Overstyr valg av price-kolonne (valgfritt)")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        print(f"ERROR: Fant ikke input: {inp}")
        return 1

    df = pd.read_parquet(inp)

    # Hvis index kan være dato, ta den med
    if isinstance(df.index, pd.DatetimeIndex) and ("date" not in [c.lower() for c in df.columns]):
        df = df.reset_index().rename(columns={"index": "date"})

    # Hvis MultiIndex columns (sjeldent), flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if str(x) != ""]) for tup in df.columns.values]

    # Wide -> long (melt)
    if _looks_like_wide_price_matrix(df):
        date_col = _pick_col(df, DATE_CANDIDATES)
        if date_col is None and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
            date_col = "date"
        if date_col is None:
            print("ERROR: Klarte ikke å finne date-kolonne for wide-format.")
            print("Cols:", df.columns.tolist()[:80])
            return 1

        value_cols = [c for c in df.columns if c != date_col]
        long_df = df.melt(id_vars=[date_col], value_vars=value_cols, var_name="yahoo_ticker", value_name="adj_close")
        long_df = long_df.rename(columns={date_col: "date"})
        long_df["date"] = _ensure_datetime(long_df["date"])
        long_df["adj_close"] = pd.to_numeric(long_df["adj_close"], errors="coerce")
        long_df = long_df.dropna(subset=["yahoo_ticker", "date", "adj_close"])
        long_df = long_df.sort_values(["yahoo_ticker", "date"], kind="mergesort")
        out = long_df
    else:
        # Long-format: finn/standardiser kolonner
        tcol = _pick_col(df, TICKER_CANDIDATES)
        dcol = _pick_col(df, DATE_CANDIDATES)

        if tcol is None:
            print("ERROR: Fant ingen ticker-kolonne. Kolonner:")
            print(df.columns.tolist())
            return 1
        if dcol is None:
            # kan ligge i index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
                dcol = "date"
            else:
                print("ERROR: Fant ingen date-kolonne. Kolonner:")
                print(df.columns.tolist())
                return 1

        if args.force_price_col:
            pcol = args.force_price_col
            if pcol not in df.columns:
                print(f"ERROR: --force-price-col={pcol} finnes ikke. Kolonner:")
                print(df.columns.tolist())
                return 1
        else:
            pcol = _pick_col(df, PRICE_CANDIDATES)

        if pcol is None:
            print("ERROR: Fant ingen price-kolonne (adj_close/close/price). Kolonner:")
            print(df.columns.tolist())
            return 1

        vcol = _pick_col(df, VOLUME_CANDIDATES)

        cols = [tcol, dcol, pcol] + ([vcol] if vcol else [])
        out = df[cols].copy()
        out = out.rename(columns={tcol: "yahoo_ticker", dcol: "date", pcol: "adj_close"})
        if vcol:
            out = out.rename(columns={vcol: "volume"})

        out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
        out["date"] = _ensure_datetime(out["date"])
        out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
        if "volume" in out.columns:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

        out = out.dropna(subset=["yahoo_ticker", "date", "adj_close"])
        out = out.sort_values(["yahoo_ticker", "date"], kind="mergesort")

    # Backup hvis output finnes
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = outp.with_suffix(outp.suffix + f".bak_{ts}")
        outp.replace(bak)
        print(f"OK: backup -> {bak}")

    out.to_parquet(outp, index=False)

    # Rapport
    print(f"OK: wrote {outp}")
    print("rows:", len(out))
    print("tickers:", out["yahoo_ticker"].nunique())
    print("date_min:", out["date"].min(), "date_max:", out["date"].max())
    print("cols:", out.columns.tolist())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
