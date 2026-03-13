"""
tools/ingest_fundamentals_from_freeze.py

Lager fundamentals_history.parquet fra lokale Borsdata freeze-filer
i stedet for å kalle Borsdata API.

Eksempel:
    python -m tools.ingest_fundamentals_from_freeze --asof 2026-03-12
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# KPI-specs identisk med ingest_fundamentals_history.py
KPI_SPECS_FULL = [
    {"name": "roic",          "kpi_id": 37,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "ev_ebit",       "kpi_id": 10,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "netdebt_ebitda","kpi_id": 42,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "gross_profit_m","kpi_id": 135, "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
    {"name": "total_assets_m","kpi_id": 57,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "fcf_m",         "kpi_id": 63,  "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
    {"name": "mcap_m",        "kpi_id": 50,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "ev_m",          "kpi_id": 49,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "netdebt_m",     "kpi_id": 60,  "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "ebit_m",        "kpi_id": 55,  "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
    {"name": "ebitda_m",      "kpi_id": 54,  "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
]
KPI_SPECS_CORE = KPI_SPECS_FULL[:3]
KPI_BY_ID: Dict[int, dict] = {s["kpi_id"]: s for s in KPI_SPECS_FULL}


def _find_freeze_root(base: Path, asof: str, require_subdir: str = "") -> Path:
    """Finn nyeste freeze-mappe <= asof under base.

    Hvis require_subdir er satt, hopper over mapper som ikke har den undermappen.
    """
    candidates = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    for d in candidates:
        if d.name > asof:
            continue
        if require_subdir and not (d / require_subdir).exists():
            continue
        return d
    # Fallback: nyeste som har undermappen (uavhengig av dato)
    if require_subdir:
        for d in candidates:
            if (d / require_subdir).exists():
                return d
    elif candidates:
        return candidates[0]
    raise FileNotFoundError(f"Ingen freeze-mappe funnet under {base}" + (f" med {require_subdir}" if require_subdir else ""))


def _load_instrument_master(freeze_proplus: Path, asof: str) -> pd.DataFrame:
    """Les instrument_master_nordic.json.gz og returner DataFrame med ins_id, ticker, yahoo, etc."""
    candidates = sorted([d for d in freeze_proplus.iterdir() if d.is_dir()], reverse=True)
    master_path: Optional[Path] = None
    for d in candidates:
        p = d / "meta_global_capture" / "instrument_master_nordic.json.gz"
        if p.exists():
            master_path = p
            break
    if not master_path:
        raise FileNotFoundError(f"instrument_master_nordic.json.gz ikke funnet under {freeze_proplus}")

    with gzip.open(master_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    # Freeze-filer har {"meta": ..., "params": ..., "payload": ...} wrapper
    if isinstance(data, dict) and "payload" in data:
        data = data["payload"]

    if isinstance(data, list):
        instruments = data
    else:
        instruments = data.get("instruments", data.get("instrumentList", data.get("Instruments", [])))
    df = pd.json_normalize(instruments)

    # Normaliser kolonnenavn
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("insid", "ins_id", "id"):
            col_map[c] = "ins_id"
        elif cl == "ticker":
            col_map[c] = "ticker"
        elif cl in ("yahoo", "yahooticker", "yahoo_ticker"):
            col_map[c] = "yahoo_ticker"
        elif cl in ("name", "company", "companyname"):
            col_map[c] = "company"
        elif cl in ("country", "countrycode"):
            col_map[c] = "country"
        elif cl in ("sector", "sectorname"):
            col_map[c] = "sector"
    df = df.rename(columns=col_map)

    for col in ["ins_id", "ticker", "yahoo_ticker", "company", "country", "sector"]:
        if col not in df.columns:
            df[col] = ""

    df["ins_id"] = pd.to_numeric(df["ins_id"], errors="coerce")
    df = df.dropna(subset=["ins_id"]).copy()
    df["ins_id"] = df["ins_id"].astype(int)
    return df[["ins_id", "ticker", "yahoo_ticker", "company", "country", "sector"]].drop_duplicates(subset=["ins_id"])


def _parse_payload(payload: Any) -> List[Dict[str, Any]]:
    """Parser Borsdata KPI-history payload til liste med {ins_id, date, value}."""
    rows: List[Dict[str, Any]] = []

    def _points(obj: Dict[str, Any]) -> list:
        for k in ["values", "Values", "history", "data", "items", "kpiHistory"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        return []

    def _date_from_year_period(y_raw: Any, p_raw: Any) -> Optional[str]:
        try:
            y, p = int(float(y_raw)), int(float(p_raw))
        except Exception:
            return None
        quarter_map = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31", 5: "12-31"}
        mm = quarter_map.get(p)
        return f"{y:04d}-{mm}" if mm else None

    def _date_val(pt: Dict[str, Any]):
        d = pt.get("d") or pt.get("date") or pt.get("reportEndDate")
        if d in (None, ""):
            d = _date_from_year_period(pt.get("y"), pt.get("p"))
        v = pt.get("v") if "v" in pt else pt.get("value", pt.get("val"))
        return d, v

    if isinstance(payload, dict):
        for k in ["kpisList", "kpiHistoryList", "data", "Data", "instruments"]:
            if k in payload and isinstance(payload[k], list):
                payload = payload[k]
                break

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            has_id = any(k in first for k in ("insId", "ins_id", "InsId", "id", "instrument"))
            if has_id:
                for inst in payload:
                    if not isinstance(inst, dict):
                        continue
                    ins_id = (inst.get("insId") or inst.get("ins_id") or inst.get("InsId")
                              or inst.get("id") or inst.get("instrument"))
                    for pt in _points(inst):
                        if isinstance(pt, dict):
                            d, v = _date_val(pt)
                            rows.append({"ins_id": ins_id, "date": d, "value": v})
            else:
                for pt in payload:
                    if isinstance(pt, dict):
                        d, v = _date_val(pt)
                        rows.append({"ins_id": None, "date": d, "value": v})
    return rows


def _read_batch_files(kpi_history_dir: Path, kpi_id: int, reporttype: str, pricetype: str) -> pd.DataFrame:
    """Les alle batch-filer for en bestemt KPI/reporttype/pricetype."""
    pattern = f"kpi_{kpi_id}__{reporttype}_{pricetype}__batch_*.json.gz"
    files = sorted(kpi_history_dir.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["ins_id", "date", "value"])

    all_rows: List[Dict] = []
    for f in files:
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                wrapper = json.load(fh)
            payload = wrapper.get("payload", wrapper)
            all_rows.extend(_parse_payload(payload))
        except Exception as e:
            print(f"  [WARN] kunne ikke lese {f.name}: {e}")

    if not all_rows:
        return pd.DataFrame(columns=["ins_id", "date", "value"])

    df = pd.DataFrame(all_rows)
    df["ins_id"] = pd.to_numeric(df["ins_id"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["ins_id", "date"]).copy()
    df["ins_id"] = df["ins_id"].astype(int)
    return df


def build_fundamentals_from_freeze(
    asof: str,
    freeze_root: Path,
    raw_dir: Path,
    kpi_set: str = "full",
    mode: str = "both",
    include_r12: bool = True,
) -> Path:
    freeze_proplus = freeze_root / "borsdata_proplus"
    # year/quarter KPI-data: data/freeze/borsdata/{date}/raw/kpi_history/
    freeze_borsdata = freeze_root / "borsdata"
    # r12 KPI-data: data/freeze/borsdata_proplus_freeze/{date}/raw/kpi_history/
    freeze_proplus_freeze = freeze_root / "borsdata_proplus_freeze"

    print(f"Instrument master: ", end="")
    master_df = _load_instrument_master(freeze_proplus, asof)
    print(f"{len(master_df)} instrumenter")

    # Finn nyeste borsdata-freeze for year/quarter som faktisk har kpi_history
    borsdata_freeze_dir = _find_freeze_root(freeze_borsdata, asof, require_subdir="raw/kpi_history")
    kpi_history_dir_year = borsdata_freeze_dir / "raw" / "kpi_history"
    n_year = len(list(kpi_history_dir_year.glob("*.json.gz"))) if kpi_history_dir_year.exists() else 0
    print(f"KPI year/quarter freeze: {borsdata_freeze_dir.name}  ({n_year} batch-filer)")

    # Finn nyeste proplus_freeze for r12
    kpi_history_dir_r12: Optional[Path] = None
    if freeze_proplus_freeze.exists():
        try:
            r12_freeze_dir = _find_freeze_root(freeze_proplus_freeze, asof)
            kpi_history_dir_r12 = r12_freeze_dir / "raw" / "kpi_history"
            n_r12 = len(list(kpi_history_dir_r12.glob("*.json.gz"))) if kpi_history_dir_r12.exists() else 0
            print(f"KPI r12 freeze: {r12_freeze_dir.name}  ({n_r12} batch-filer)")
        except FileNotFoundError:
            print("KPI r12 freeze: ikke funnet")

    want_year = mode in ("year", "both")
    want_quarter = mode in ("quarter", "both")
    target_reporttypes = []
    if want_year:
        target_reporttypes.append("year")
    if want_quarter:
        target_reporttypes.append("quarter")
    if include_r12:
        target_reporttypes.append("r12")

    kpis = KPI_SPECS_CORE if kpi_set == "core" else KPI_SPECS_FULL
    all_rows: List[pd.DataFrame] = []

    for spec in kpis:
        name = spec["name"]
        kpi_id = int(spec["kpi_id"])
        pricetype = spec.get("pricetype", "mean")
        available = set(spec["reporttypes"])

        for reporttype in target_reporttypes:
            if reporttype not in available:
                continue
            # r12 ligger i proplus_freeze, year/quarter i borsdata
            if reporttype == "r12":
                if kpi_history_dir_r12 is None or not kpi_history_dir_r12.exists():
                    continue
                kpi_dir = kpi_history_dir_r12
            else:
                kpi_dir = kpi_history_dir_year
            df = _read_batch_files(kpi_dir, kpi_id, reporttype, pricetype)
            if df.empty:
                print(f"  {name} ({kpi_id}) / {reporttype}: ingen data")
                continue
            df["kpi_id"] = kpi_id
            df["metric"] = name
            df["report_type"] = reporttype
            df["price_type"] = pricetype
            all_rows.append(df)
            print(f"  {name} ({kpi_id}) / {reporttype}: {len(df):,} rader, {df['ins_id'].nunique()} selskaper")

    if not all_rows:
        raise RuntimeError("Ingen KPI-data funnet i freeze. Sjekk kpi_history-mappen.")

    hist = pd.concat(all_rows, ignore_index=True)
    meta_cols = [c for c in ["ins_id", "ticker", "yahoo_ticker", "company", "country", "sector"]
                 if c in master_df.columns]
    hist = hist.merge(master_df[meta_cols], on="ins_id", how="left")

    cols = [c for c in ["yahoo_ticker", "ticker", "ins_id", "company", "country", "sector",
                        "metric", "kpi_id", "report_type", "price_type", "date", "value"]
            if c in hist.columns]
    hist = hist[cols].sort_values(["yahoo_ticker", "metric", "report_type", "date"], kind="mergesort")

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "fundamentals_history.parquet"
    hist.to_parquet(out_path, index=False)

    meta = {
        "asof": asof,
        "source": "freeze",
        "freeze_date": kpi_freeze_dir.name,
        "kpi_set": kpi_set,
        "mode": mode,
        "include_r12": include_r12,
        "rows": int(len(hist)),
        "tickers": int(hist["ticker"].nunique()) if "ticker" in hist.columns else 0,
        "kpis": [s["name"] for s in kpis],
    }
    (raw_dir / "ingest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[OK] {out_path}  ({len(hist):,} rader, {meta['tickers']} tickers)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Bygg fundamentals_history.parquet fra Borsdata freeze-filer")
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD")
    ap.add_argument("--freeze-root", default="data/freeze", help="Rot-mappe for freeze-data")
    ap.add_argument("--kpi-set", choices=["core", "full"], default="full")
    ap.add_argument("--mode", choices=["year", "quarter", "both"], default="both")
    ap.add_argument("--no-r12", action="store_true", help="Ikke inkluder r12")
    args = ap.parse_args()

    raw_dir = Path("data") / "raw" / args.asof
    freeze_root = Path(args.freeze_root)

    build_fundamentals_from_freeze(
        asof=args.asof,
        freeze_root=freeze_root,
        raw_dir=raw_dir,
        kpi_set=args.kpi_set,
        mode=args.mode,
        include_r12=not args.no_r12,
    )


if __name__ == "__main__":
    main()
