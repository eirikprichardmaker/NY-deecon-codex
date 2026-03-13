"""
Hent selskapsinfo fra Yahoo Finance og skriv til data/raw/ir/{ticker}/{date}/company_info.txt.

Gir Agent A kvalitativ kontekst: forretningsbeskrivelse, sektor, industri, ansatte m.m.

Bruk:
  python -m tools.fetch_company_info --asof 2026-03-12 --tickers TALK,VEI,CAMBI
  python -m tools.fetch_company_info --asof 2026-03-12  (leser fra siste shortlist)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _find_nordic_master(repo: Path) -> Path:
    candidates = [
        repo / "data/freeze/borsdata_proplus/2026-03-12/meta_global_capture/instrument_master_nordic.json.gz",
    ]
    import glob
    pattern = str(repo / "data/freeze/borsdata_proplus/*/meta_global_capture/instrument_master_nordic.json.gz")
    found = sorted(glob.glob(pattern))
    if found:
        return Path(found[-1])
    raise FileNotFoundError("Fant ikke instrument_master_nordic.json.gz")


def _load_master(repo: Path) -> dict[str, str]:
    """Returner {ticker -> yahoo_symbol}."""
    import gzip
    path = _find_nordic_master(repo)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    instruments = data if isinstance(data, list) else data.get("instruments", [])
    mapping = {}
    for inst in instruments:
        ticker = inst.get("ticker") or inst.get("symbol") or ""
        yahoo = inst.get("yahoo") or inst.get("yahoo_symbol") or ""
        if ticker and yahoo:
            mapping[ticker.upper()] = yahoo
    return mapping


def _find_latest_shortlist(repo: Path) -> list[str]:
    runs = sorted((repo / "runs").glob("*/shortlist.csv"))
    if not runs:
        return []
    import csv
    tickers = []
    with open(runs[-1], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("ticker", "").strip()
            if t:
                tickers.append(t)
    return tickers


def _fetch_info(yahoo_symbol: str) -> dict:
    try:
        import yfinance as yf
        tick = yf.Ticker(yahoo_symbol)
        info = tick.info or {}
        return info
    except Exception:
        return {}


def _format_info(ticker: str, yahoo_symbol: str, info: dict) -> str:
    lines = [f"=== Selskapsinfo: {ticker} ({yahoo_symbol}) ===\n"]

    for label, key in [
        ("Navn", "longName"),
        ("Sektor", "sector"),
        ("Industri", "industry"),
        ("Land", "country"),
        ("Ansatte", "fullTimeEmployees"),
        ("Børs", "exchange"),
        ("Valuta", "currency"),
    ]:
        val = info.get(key)
        if val:
            lines.append(f"{label}: {val}")

    desc = info.get("longBusinessSummary", "")
    if desc:
        lines.append(f"\nForretningsbeskrivelse:\n{desc}")

    for label, key in [
        ("Markedsverdi", "marketCap"),
        ("P/E", "trailingPE"),
        ("P/B", "priceToBook"),
        ("EV/EBITDA", "enterpriseToEbitda"),
        ("Omsetning (TTM)", "totalRevenue"),
        ("EBITDA (TTM)", "ebitda"),
        ("Netto gjeld", "totalDebt"),
        ("Kontanter", "totalCash"),
        ("52-ukers høy", "fiftyTwoWeekHigh"),
        ("52-ukers lav", "fiftyTwoWeekLow"),
        ("Analytiker anbefaling", "recommendationKey"),
        ("Antall analytikere", "numberOfAnalystOpinions"),
        ("Mål-pris (gj.snitt)", "targetMeanPrice"),
    ]:
        val = info.get(key)
        if val is not None:
            if isinstance(val, float) and val > 1_000_000:
                lines.append(f"{label}: {val/1e6:.1f}M")
            else:
                lines.append(f"{label}: {val}")

    return "\n".join(lines) + "\n"


def fetch_company_info(
    asof: str,
    tickers: list[str],
    repo: Path,
    delay: float = 1.0,
) -> None:
    master = _load_master(repo)
    out_base = repo / "data/raw/ir"

    for ticker in tickers:
        yahoo = master.get(ticker.upper())
        if not yahoo:
            print(f"  {ticker}: ingen Yahoo-ticker i master, hopper over")
            continue

        print(f"  {ticker} ({yahoo}): henter info...", end=" ", flush=True)
        info = _fetch_info(yahoo)

        if not info or not info.get("longName"):
            print("ingen data")
            continue

        text = _format_info(ticker, yahoo, info)
        out_dir = out_base / ticker / asof
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "company_info.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"skrevet ({len(text)} tegn)")
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hent selskapsinfo fra Yahoo Finance")
    parser.add_argument("--asof", required=True, help="Dato (YYYY-MM-DD)")
    parser.add_argument("--tickers", help="Kommaseparert liste, f.eks. TALK,VEI")
    parser.add_argument("--repo", default=".", help="Rot-mappe for prosjektet")
    parser.add_argument("--delay", type=float, default=1.0, help="Sekunder mellom kall")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = _find_latest_shortlist(repo)
        if not tickers:
            print("Ingen tickers funnet. Bruk --tickers eller kjør decision-steget først.")
            return
        print(f"Fra siste shortlist: {tickers}")

    print(f"Henter info for {len(tickers)} selskaper...")
    fetch_company_info(asof=args.asof, tickers=tickers, repo=repo, delay=args.delay)


if __name__ == "__main__":
    main()
