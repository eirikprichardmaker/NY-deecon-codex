"""
Bygger data/golden/prices_wft_combined.parquet fra Børsdata-freeze prishistorikk.

Kilder:
  data/freeze/borsdata/<freeze_date>/raw/stockprices_history/*.json.gz
    → stockPricesList per instrument: {d, h, l, c, o, v}
  data/processed/instrument_metadata.parquet
    → ins_id → yahoo_ticker mapping
  data/golden/prices_wft_combined.parquet (eksisterende)
    → behold indekstickers (^OSEAX, ^OMXS, ^OMXC25, ^HEX) som ikke er i freeze

Output: data/golden/prices_wft_combined.parquet
  Kolonner: ticker (yahoo-format), date, adj_close, volume

Detaljer:
  - Børsdata close-priser (c) er split-justerte
  - 700 vinduer à 1 år, 2006-02-23 → 2026-02-18
  - Dedupliserer på (ticker, date) ved evt. overlapp mellom vinduer
  - Skriver atomisk (tmp-fil + rename)

Bruk:
  python -m tools.build_prices_from_freeze
  python -m tools.build_prices_from_freeze --freeze-date 2026-02-18
  python -m tools.build_prices_from_freeze --no-keep-index   # ikke behold ^-tickers
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _latest_freeze_date(repo: Path) -> str:
    freeze_root = repo / "data" / "freeze" / "borsdata"
    dates = sorted(
        d for d in os.listdir(freeze_root)
        if (freeze_root / d / "raw" / "stockprices_history").exists()
    )
    if not dates:
        raise FileNotFoundError(
            f"Ingen freeze med stockprices_history under {freeze_root}"
        )
    return dates[-1]


def _build_ins_to_yahoo(meta_path: Path) -> dict[int, str]:
    """ins_id (int) → yahoo_ticker (str)"""
    df = pd.read_parquet(meta_path, columns=["ins_id", "yahoo_ticker"])
    df = df.dropna(subset=["ins_id", "yahoo_ticker"])
    df = df.drop_duplicates(subset="ins_id")
    return {int(row.ins_id): str(row.yahoo_ticker) for row in df.itertuples()}


def _parse_window_file(path: Path, ins_to_yahoo: dict[int, str]) -> pd.DataFrame:
    """
    Parse én price-window-fil.
    Returnerer DataFrame: ticker, date, adj_close, volume
    """
    with gzip.open(path) as f:
        payload = json.load(f)["payload"]

    rows: list[tuple[str, str, float, int]] = []
    for item in payload.get("stockPricesArrayList", []):
        ins_id = item.get("instrument")
        yahoo = ins_to_yahoo.get(ins_id)
        if not yahoo:
            continue
        for p in item.get("stockPricesList", []):
            d = p.get("d")
            c = p.get("c")
            v = p.get("v", 0)
            if d and c is not None:
                rows.append((yahoo, d, float(c), int(v or 0)))

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "adj_close", "volume"])

    return pd.DataFrame(rows, columns=["ticker", "date", "adj_close", "volume"])


def build(
    freeze_date: str | None = None,
    keep_index: bool = True,
    repo: Path | None = None,
    verbose: bool = True,
) -> Path:
    if repo is None:
        repo = Path(__file__).parent.parent

    if freeze_date is None:
        freeze_date = _latest_freeze_date(repo)
    if verbose:
        print(f"Bruker freeze: {freeze_date}")

    price_dir = repo / "data" / "freeze" / "borsdata" / freeze_date / "raw" / "stockprices_history"
    meta_path = repo / "data" / "processed" / "instrument_metadata.parquet"
    golden_out = repo / "data" / "golden" / "prices_wft_combined.parquet"

    if not price_dir.exists():
        raise FileNotFoundError(f"stockprices_history ikke funnet: {price_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"instrument_metadata.parquet mangler — kjør tools/fetch_borsdata_metadata.py"
        )

    ins_to_yahoo = _build_ins_to_yahoo(meta_path)
    if verbose:
        print(f"ins_id-mapping: {len(ins_to_yahoo)} instrumenter")

    price_files = sorted(price_dir.glob("*.json.gz"))
    if verbose:
        print(f"Prosesserer {len(price_files)} prisfiler...")

    chunks: list[pd.DataFrame] = []
    n_rows = 0

    for i, f in enumerate(price_files):
        chunk = _parse_window_file(f, ins_to_yahoo)
        if not chunk.empty:
            chunks.append(chunk)
            n_rows += len(chunk)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(price_files)} filer — {n_rows:,} rader så langt")

    if not chunks:
        raise RuntimeError("Ingen prisdata funnet i freeze-filene")

    if verbose:
        print(f"Kombinerer {len(chunks)} chunks ({n_rows:,} rader totalt)...")

    prices = pd.concat(chunks, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date"])

    # Dedupliser: ved overlapp mellom vinduer beholder vi siste verdi
    prices = (
        prices
        .sort_values(["ticker", "date"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
    )

    # Behold indekstickers fra eksisterende golden prices
    if keep_index and golden_out.exists():
        existing = pd.read_parquet(golden_out)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        idx_tickers = existing[existing["ticker"].str.startswith("^")]
        if not idx_tickers.empty:
            # Kun kolonner vi trenger
            idx_tickers = idx_tickers[["ticker", "date", "adj_close", "volume"]].copy()
            prices = pd.concat([prices, idx_tickers], ignore_index=True)
            prices = (
                prices
                .sort_values(["ticker", "date"])
                .drop_duplicates(subset=["ticker", "date"], keep="last")
            )
            if verbose:
                kept = idx_tickers["ticker"].nunique()
                print(f"Beholdt {kept} indekstickers fra eksisterende golden prices")

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Stats
    if verbose:
        first_obs = prices.groupby("ticker")["date"].min()
        print(f"\nResultat:")
        print(f"  Rader totalt:     {len(prices):,}")
        print(f"  Unike tickers:    {prices['ticker'].nunique():,}")
        print(f"  Datoperiode:      {prices['date'].min().date()} til {prices['date'].max().date()}")
        print(f"  Tickers fra 2006: {(first_obs.dt.year <= 2006).sum()}")
        print(f"  Tickers fra 2007: {(first_obs.dt.year <= 2007).sum()}")
        print(f"  Tickers fra 2010: {(first_obs.dt.year <= 2010).sum()}")
        print(f"  Tickers fra 2020+:{(first_obs.dt.year >= 2020).sum()}")
        # Beta-dekning
        months = ((prices.groupby("ticker")["date"].max() - prices.groupby("ticker")["date"].min()).dt.days / 30.4)
        print(f"  Tickers >= 36m:   {(months >= 36).sum()}")
        print(f"  Tickers >= 24m:   {(months >= 24).sum()}")
        print(f"  Tickers < 24m:    {(months < 24).sum()}")

    # Atomisk skriv (tmp + rename)
    golden_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = golden_out.with_suffix(".tmp.parquet")
    prices.to_parquet(tmp, index=False)
    tmp.replace(golden_out)

    if verbose:
        print(f"\nSkrevet: {golden_out}")

    return golden_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Bygg prices_wft_combined.parquet fra Børsdata freeze")
    ap.add_argument("--freeze-date", default=None, help="Freeze-dato (default: siste tilgjengelige)")
    ap.add_argument("--no-keep-index", action="store_true", help="Ikke behold ^-indekstickers fra eksisterende fil")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    build(
        freeze_date=args.freeze_date,
        keep_index=not args.no_keep_index,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
