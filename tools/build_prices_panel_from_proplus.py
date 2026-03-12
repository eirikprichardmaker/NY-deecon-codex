"""
Build prices_panel.parquet from borsdata_proplus_capture output.

Reads:
  - instrument_master_global.json.gz  -> ins_id to yahoo ticker mapping
  - global_prices/by_date/*.json.gz   -> daily price snapshots

Writes:
  - data/raw/prices/prices_panel.parquet  (ticker=yahoo, date, adj_close, volume)
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def _load_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _find_list(obj: object) -> list | None:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("instruments", "instrumentList", "instrumentsList",
                    "stockPricesList", "stockprices", "prices", "list"):
            if isinstance(obj.get(key), list):
                return obj[key]
        for v in obj.values():
            if isinstance(v, list):
                return v
    return None


def _build_ins_map(master_path: Path) -> dict[int, str]:
    data = _load_gz(master_path)
    payload = data.get("payload", data)
    ins_list = _find_list(payload)
    if not ins_list:
        raise ValueError(f"No instruments list in {master_path}")
    mapping: dict[int, str] = {}
    for ins in ins_list:
        if not isinstance(ins, dict):
            continue
        ins_id = ins.get("insId") or ins.get("ins_id") or ins.get("id")
        yahoo = ins.get("yahoo") or ins.get("yahooTicker") or ins.get("ticker")
        if ins_id and yahoo:
            mapping[int(ins_id)] = str(yahoo).strip()
    return mapping


def _parse_date_file(path: Path, fallback_date: str) -> list[dict]:
    data = _load_gz(path)
    date_str = data.get("date") or fallback_date
    payload = data.get("payload", data)
    price_list = _find_list(payload)
    if not price_list:
        return []
    rows = []
    for item in price_list:
        if not isinstance(item, dict):
            continue
        ins_id = item.get("i") or item.get("insId") or item.get("ins_id")
        close = item.get("c") or item.get("close")
        d = item.get("d") or item.get("date") or date_str
        volume = item.get("v") or item.get("volume") or 0
        if ins_id and close is not None:
            rows.append({
                "ins_id": int(ins_id),
                "date": str(d),
                "adj_close": float(close),
                "volume": int(volume) if volume else 0,
            })
    return rows


def build_prices_panel(capture_dir: Path, output: Path) -> int:
    # Nordic master has correct Yahoo tickers (.OL, .ST, .CO, .HE)
    # Global master lacks these suffixes
    nordic_path = capture_dir / "instrument_master_nordic.json.gz"
    global_path = capture_dir / "instrument_master_global.json.gz"

    if not nordic_path.exists() and not global_path.exists():
        print(f"ERROR: instrument master not found in {capture_dir}")
        return 1

    print(f"Loading instrument master...")
    ins_map: dict[int, str] = {}
    if nordic_path.exists():
        ins_map.update(_build_ins_map(nordic_path))
        print(f"  Nordic: {len(ins_map)} instruments")
    if global_path.exists():
        before = len(ins_map)
        global_map = _build_ins_map(global_path)
        # Only add global entries not already covered by Nordic
        for k, v in global_map.items():
            if k not in ins_map:
                ins_map[k] = v
        print(f"  Global: +{len(ins_map) - before} additional instruments")
    print(f"  Totalt: {len(ins_map)} instruments with yahoo ticker")

    by_date_dir = capture_dir / "global_prices" / "by_date"
    if not by_date_dir.exists():
        print(f"ERROR: {by_date_dir} not found")
        return 1

    date_files = sorted(by_date_dir.glob("date_*.json.gz"))
    print(f"Processing {len(date_files)} date files...")

    all_rows: list[dict] = []
    for i, fpath in enumerate(date_files, 1):
        rows = _parse_date_file(fpath, fpath.stem.replace("date_", ""))
        all_rows.extend(rows)
        if i % 50 == 0 or i == len(date_files):
            print(f"  {i}/{len(date_files)} filer, {len(all_rows)} rader")

    if not all_rows:
        print("ERROR: Ingen prisrader funnet")
        return 1

    df = pd.DataFrame(all_rows)
    df["ticker"] = df["ins_id"].map(ins_map)
    df = df.dropna(subset=["ticker"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_close"])
    df = df[["ticker", "date", "adj_close", "volume"]].sort_values(["ticker", "date"])
    df = df.drop_duplicates(subset=["ticker", "date"])

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    print(f"\nFerdig:")
    print(f"  Rader:    {len(df)}")
    print(f"  Tickers:  {df['ticker'].nunique()}")
    print(f"  Fra dato: {df['date'].min().date()}")
    print(f"  Til dato: {df['date'].max().date()}")
    print(f"  Skrevet:  {output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bygg prices_panel.parquet fra proplus capture")
    ap.add_argument("--capture-dir",
                    default="data/freeze/borsdata_proplus/2026-03-12/meta_global_capture",
                    help="Sti til capture-mappen (relativ til prosjektrot)")
    ap.add_argument("--output",
                    default="data/raw/prices/prices_panel.parquet",
                    help="Output parquet-fil")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    capture_dir = root / args.capture_dir
    output = root / args.output
    return build_prices_panel(capture_dir, output)


if __name__ == "__main__":
    raise SystemExit(main())
