"""
Henter instrument-metadata fra Børsdata API og lagrer til
data/processed/instrument_metadata.parquet

Inneholder per ticker:
  ins_id, ticker, yahoo_ticker, company, country, country_id,
  market, market_id, sector_id, sector_se, sector_en,
  branch_id, branch_se, damodaran_sector

Damodaran-mapping brukes som T2-fallback i cost_of_capital.py
når beregnet beta mangler.

Bruk:
  python -m tools.fetch_borsdata_metadata
  python -m tools.fetch_borsdata_metadata --out data/processed/instrument_metadata.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://apiservice.borsdata.se/v1"

# ---------------------------------------------------------------------------
# Børsdata sector_id → English name
# ---------------------------------------------------------------------------
SECTOR_EN: dict[int, str] = {
    1:  "Finance & Real Estate",
    2:  "Consumer Staples",
    3:  "Energy",
    4:  "Healthcare",
    5:  "Industrials",
    6:  "Information Technology",
    7:  "Materials",
    8:  "Consumer Discretionary",
    9:  "Telecom",
    10: "Utilities",
}

# ---------------------------------------------------------------------------
# Børsdata branch_id → Damodaran sector label
# Used by cost_of_capital T2 beta fallback.
# Damodaran labels must match keys in _DAMODARAN_UNLEVERED in cost_of_capital.py
# ---------------------------------------------------------------------------
BRANCH_DAMODARAN: dict[int, str] = {
    # Energi (sectorId=3)
    1:  "Oil & Gas",   # Olja & Gas - Borrning
    2:  "Oil & Gas",   # Exploatering
    3:  "Oil & Gas",   # Transport
    4:  "Oil & Gas",   # Försäljning
    5:  "Oil & Gas",   # Service
    6:  "Energy",      # Bränsle - Kol
    7:  "Energy",      # Bränsle - Uran
    # Kraftförsörjning (sectorId=10)
    8:  "Utilities",
    9:  "Utilities",
    10: "Utilities",
    11: "Utilities",
    12: "Utilities",
    13: "Utilities",
    14: "Utilities",
    # Material (sectorId=7)
    15: "Materials",   # Kemikalier
    16: "Materials",   # Gruv - Prospekt
    17: "Materials",   # Gruv - Industrimetaller
    18: "Materials",   # Gruv - Guld & Silver
    19: "Materials",   # Gruv - Ädelstenar
    20: "Materials",   # Gruv - Service
    21: "Materials",   # Skogsbolag
    22: "Materials",   # Förpackning
    # Industri (sectorId=5)
    23: "Industrials",
    24: "Industrials",
    25: "Industrials",
    26: "Industrials",  # Militär & Försvar
    27: "Industrials",
    28: "Industrials",  # Byggnation
    29: "Industrials",  # Bostadsbyggnation
    30: "Industrials",
    31: "Industrials",
    32: "Industrials",
    33: "Industrials",  # Bemanning
    34: "Industrials",  # Affärskonsulter
    35: "Industrials",  # Säkerhet
    36: "Industrials",  # Utbildning
    37: "Industrials",
    38: "Industrials",
    39: "Industrials",
    40: "Industrials",  # Flygtransport
    41: "Shipping",     # Sjöfart & Rederi
    42: "Industrials",  # Tåg- & Lastbilstransport
    # Sällankapsvaror / Consumer Discretionary (sectorId=8)
    43: "Consumer Discretionary",
    44: "Consumer Discretionary",
    45: "Consumer Discretionary",
    46: "Consumer Discretionary",
    47: "Consumer Discretionary",
    48: "Consumer Discretionary",
    49: "Consumer Discretionary",
    50: "Consumer Discretionary",
    51: "Consumer Discretionary",
    52: "Consumer Discretionary",
    53: "Consumer Discretionary",
    54: "Consumer Discretionary",
    55: "Consumer Discretionary",
    56: "Consumer Discretionary",
    57: "Consumer Discretionary",
    # Dagligvaror / Consumer Staples (sectorId=2)
    58: "Consumer Staples",   # Bryggeri
    59: "Consumer Staples",   # Drycker
    60: "Consumer Staples",   # Jordbruk
    61: "Seafood",            # Fiskodling
    62: "Consumer Staples",   # Tobak
    63: "Consumer Staples",   # Livsmedel
    64: "Consumer Staples",
    65: "Consumer Staples",
    66: "Consumer Staples",
    67: "Consumer Staples",
    # Finans & Fastighet (sectorId=1)
    68: "Banking",            # Banker
    69: "Banking",            # Nischbanker
    70: "Financial Services", # Kredit & Finansiering
    71: "Financial Services", # Kapitalförvaltning
    72: "Financial Services", # Fondförvaltning
    73: "Financial Services", # Investmentbolag
    74: "Insurance",          # Försäkring
    75: "Real Estate",        # Fastighetsbolag
    76: "Real Estate",        # Fastighet - REIT
    # Hälsovård (sectorId=4)
    77: "Pharmaceuticals",    # Läkemedel
    78: "Healthcare",         # Biotech
    79: "Healthcare",         # Medicinsk Utrustning
    80: "Healthcare",
    81: "Healthcare",
    # Informationsteknik (sectorId=6)
    82: "Technology",
    83: "Technology",
    84: "Technology",
    85: "Technology",
    86: "Technology",
    87: "Technology",
    88: "Technology",
    89: "Software",           # IT-Konsulter
    90: "Software",           # Affärs- & IT-System
    91: "Technology",         # Internettjänster
    92: "Technology",         # Betalning & E-handel
    # Telekommunikation (sectorId=9)
    93: "Telecom",
    94: "Telecom",
}

# Sector-level fallback when branch is unknown
SECTOR_DAMODARAN: dict[int, str] = {
    1:  "Financial Services",
    2:  "Consumer Staples",
    3:  "Energy",
    4:  "Healthcare",
    5:  "Industrials",
    6:  "Technology",
    7:  "Materials",
    8:  "Consumer Discretionary",
    9:  "Telecom",
    10: "Utilities",
}


def _get(url: str, params: dict, timeout: int = 15, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  [RETRY {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return {}


def fetch(out_path: Path, verbose: bool = True) -> pd.DataFrame:
    key = (
        os.environ.get("BORSDATA_AUTHKEY")
        or os.environ.get("BORSDATA_API_KEY")
        or os.environ.get("BORSDATA_KEY")
    )
    if not key:
        raise RuntimeError(
            "Mangler Børsdata API-nøkkel. "
            "Sett BORSDATA_AUTHKEY i .env-fil."
        )

    params = {"authKey": key}

    if verbose:
        print("Henter /instruments ...")
    instruments = _get(f"{BASE_URL}/instruments", params)["instruments"]

    if verbose:
        print("Henter /branches ...")
    branches_raw = _get(f"{BASE_URL}/branches", params)["branches"]

    if verbose:
        print("Henter /sectors ...")
    sectors_raw = _get(f"{BASE_URL}/sectors", params)["sectors"]

    if verbose:
        print("Henter /markets ...")
    markets_raw = _get(f"{BASE_URL}/markets", params)["markets"]

    if verbose:
        print("Henter /countries ...")
    countries_raw = _get(f"{BASE_URL}/countries", params)["countries"]

    # Build lookup dicts
    branch_name: dict[int, str] = {b["id"]: b["name"] for b in branches_raw}
    sector_se: dict[int, str] = {s["id"]: s["name"] for s in sectors_raw}
    market_name: dict[int, str] = {m["id"]: m["name"] for m in markets_raw}
    country_name: dict[int, str] = {c["id"]: c["name"] for c in countries_raw}

    rows = []
    for inst in instruments:
        ins_id   = inst.get("insId")
        ticker   = inst.get("ticker", "")
        yahoo    = inst.get("yahoo", "")
        company  = inst.get("name", "")
        cid      = inst.get("countryId")
        mid      = inst.get("marketId")
        sid      = inst.get("sectorId")
        bid      = inst.get("branchId")

        # Damodaran: branch first, then sector fallback
        dam = (
            BRANCH_DAMODARAN.get(bid)
            or SECTOR_DAMODARAN.get(sid)
            or "Default"
        )

        rows.append({
            "ins_id":           ins_id,
            "ticker":           ticker,
            "yahoo_ticker":     yahoo,
            "company":          company,
            "country_id":       cid,
            "country":          country_name.get(cid, ""),
            "market_id":        mid,
            "market":           market_name.get(mid, ""),
            "sector_id":        sid,
            "sector_se":        sector_se.get(sid, ""),
            "sector_en":        SECTOR_EN.get(sid, ""),
            "branch_id":        bid,
            "branch_se":        branch_name.get(bid, ""),
            "damodaran_sector": dam,
        })

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\n{len(df)} instrumenter hentet.")
        print("\nSektor-fordeling:")
        print(df["sector_en"].value_counts().to_string())
        print("\nDamodaran-fordeling:")
        print(df["damodaran_sector"].value_counts().to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nLagret: {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Hent Børsdata instrument-metadata")
    ap.add_argument(
        "--out",
        default="data/processed/instrument_metadata.parquet",
        help="Utdatafil (default: data/processed/instrument_metadata.parquet)",
    )
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    repo = Path(__file__).parent.parent
    out = repo / args.out

    fetch(out, verbose=args.verbose)


if __name__ == "__main__":
    main()
