"""
Genererer docs/data_schema.json — et kompakt skjema-snapshot av alle
pipeline-nøkkelfiler i data/processed/ og data/golden/.

Inneholder per fil:
  path, size_mb, rows, columns (navn + dtype + null_pct),
  numeric_stats (p10/median/p90 for viktige kolonner),
  date_range (for datokolonner),
  sample_tickers

Bruk:
  python -m tools.generate_data_schema
  python -m tools.generate_data_schema --out docs/data_schema.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Filer som inngår i skjema-snapshot
PIPELINE_FILES = [
    "data/processed/master.parquet",
    "data/processed/master_factors.parquet",
    "data/processed/master_cost.parquet",
    "data/processed/master_valued.parquet",
    "data/processed/instrument_metadata.parquet",
    "data/processed/prices.parquet",
    "data/golden/prices_wft_combined.parquet",
]

# Kolonner som får full percentil-statistikk
STAT_COLS = {
    "roic", "wacc", "wacc_used", "coe", "coe_used", "beta",
    "market_cap", "intrinsic_equity", "net_debt_used", "net_debt_current",
    "fcf_used_millions", "fcf_m_median_3y", "fcf_m_years_available",
    "n_debt_ebitda_current", "adj_close", "ma200", "mad",
    "quality_score", "roic_wacc_spread",
}


def _dtype_label(dtype) -> str:
    s = str(dtype)
    if "int" in s:   return "int"
    if "float" in s: return "float"
    if "bool" in s:  return "bool"
    if "datetime" in s: return "datetime"
    return "str"


def _col_schema(series: pd.Series, name: str) -> dict:
    n = len(series)
    null_n = int(series.isna().sum())
    entry: dict = {
        "dtype":    _dtype_label(series.dtype),
        "null_pct": round(null_n / n * 100, 1) if n else 100.0,
    }

    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

    if numeric and name in STAT_COLS:
        valid = series.dropna()
        if len(valid) > 0:
            entry["p10"]    = _fmt(float(valid.quantile(0.10)))
            entry["median"] = _fmt(float(valid.median()))
            entry["p90"]    = _fmt(float(valid.quantile(0.90)))
            entry["min"]    = _fmt(float(valid.min()))
            entry["max"]    = _fmt(float(valid.max()))

    if pd.api.types.is_datetime64_any_dtype(series):
        valid = series.dropna()
        if len(valid):
            entry["min_date"] = str(valid.min().date())
            entry["max_date"] = str(valid.max().date())

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_bool_dtype(series):
        vc = series.value_counts()
        entry["top_values"] = {str(k): int(v) for k, v in vc.head(8).items()}

    return entry


def _fmt(v: float) -> float:
    """Round to 4 significant figures for compactness."""
    if v == 0 or not np.isfinite(v):
        return v
    from math import log10, floor
    mag = floor(log10(abs(v)))
    return round(v, -mag + 3)


def _file_schema(path: Path) -> dict:
    size_mb = round(path.stat().st_size / 1e6, 2)
    df = pd.read_parquet(path)

    # Parse date-like columns
    for c in df.columns:
        if df[c].dtype == object and c in ("date", "dato", "asof"):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    schema: dict = {
        "path":     str(path),
        "size_mb":  size_mb,
        "rows":     len(df),
        "columns":  {},
    }

    # Sample tickers
    if "ticker" in df.columns:
        sample = df["ticker"].dropna().unique()
        schema["sample_tickers"] = list(sample[:12])
        schema["ticker_count"]   = int(df["ticker"].nunique())

    # Date range (for price files)
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(dates):
            schema["date_min"] = str(dates.min().date())
            schema["date_max"] = str(dates.max().date())

    for col in df.columns:
        schema["columns"][col] = _col_schema(df[col], col)

    return schema


def generate(out_path: Path, repo: Path, verbose: bool = True) -> dict:
    result: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

    for rel in PIPELINE_FILES:
        p = repo / rel
        if not p.exists():
            if verbose:
                print(f"  SKIP (mangler): {rel}")
            result["files"][rel] = {"missing": True}
            continue
        if verbose:
            print(f"  Leser: {rel} ...")
        result["files"][rel] = _file_schema(p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nSkrevet: {out_path}  ({out_path.stat().st_size // 1024} KB)")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/data_schema.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).parent.parent
    generate(repo / args.out, repo=repo, verbose=not args.quiet)


if __name__ == "__main__":
    main()
