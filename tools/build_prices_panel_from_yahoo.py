"""
Build prices_panel.parquet for Nordic stocks using Yahoo Finance (yfinance).

Tickers hentes fra instrument_master_nordic.json.gz i Borsdata capture.
Krever: pip install yfinance
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def _load_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _get_nordic_tickers(nordic_master: Path) -> list[str]:
    data = _load_gz(nordic_master)
    payload = data.get("payload", data)
    ins_list = next((v for v in payload.values() if isinstance(v, list)), [])
    tickers = []
    for ins in ins_list:
        yahoo = ins.get("yahoo") or ins.get("yahooTicker")
        if yahoo and str(yahoo).strip():
            tickers.append(str(yahoo).strip())
    return sorted(set(tickers))


def _download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            return pd.DataFrame()

        # yfinance returns MultiIndex columns when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = [tickers[0]]

        rows = []
        for ticker in close.columns:
            s = close[ticker].dropna()
            for dt, price in s.items():
                rows.append({"ticker": ticker, "date": dt, "adj_close": float(price)})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  Batch feil: {e}")
        return pd.DataFrame()


def build_prices_panel(
    nordic_master: Path,
    output: Path,
    start: str = "2024-01-01",
    end: str | None = None,
    batch_size: int = 100,
) -> int:
    if not nordic_master.exists():
        print(f"ERROR: {nordic_master} not found")
        return 1

    tickers = _get_nordic_tickers(nordic_master)
    print(f"Nordic tickers: {len(tickers)}")

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    all_frames: list[pd.DataFrame] = []
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    print(f"Laster ned {len(tickers)} tickers i {len(batches)} batch (start={start} end={end})...")

    for i, batch in enumerate(batches, 1):
        df = _download_batch(batch, start, end)
        if not df.empty:
            all_frames.append(df)
        if i % 5 == 0 or i == len(batches):
            total = sum(len(f) for f in all_frames)
            print(f"  Batch {i}/{len(batches)}, rader så langt: {total}")
        time.sleep(0.3)

    if not all_frames:
        print("ERROR: Ingen prisdata hentet")
        return 1

    result = pd.concat(all_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_localize(None)
    result = result.dropna(subset=["adj_close"])
    result = result.drop_duplicates(subset=["ticker", "date"])
    result = result.sort_values(["ticker", "date"])
    result["volume"] = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)

    print(f"\nFerdig:")
    print(f"  Rader:    {len(result)}")
    print(f"  Tickers:  {result['ticker'].nunique()}")
    print(f"  Fra dato: {result['date'].min().date()}")
    print(f"  Til dato: {result['date'].max().date()}")
    print(f"  Skrevet:  {output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bygg prices_panel.parquet fra Yahoo Finance")
    ap.add_argument("--capture-dir",
                    default="data/freeze/borsdata_proplus/2026-03-12/meta_global_capture",
                    help="Mappe med instrument_master_nordic.json.gz")
    ap.add_argument("--output",
                    default="data/raw/prices/prices_panel.parquet")
    ap.add_argument("--start", default="2024-01-01",
                    help="Startdato for priser (YYYY-MM-DD)")
    ap.add_argument("--end", default=None,
                    help="Sluttdato (default: i dag)")
    ap.add_argument("--batch-size", type=int, default=100)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    nordic_master = root / args.capture_dir / "instrument_master_nordic.json.gz"
    output = root / args.output
    return build_prices_panel(
        nordic_master=nordic_master,
        output=output,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
