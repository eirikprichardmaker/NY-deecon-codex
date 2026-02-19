# src/ingest_fundamentals_history.py
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from src.common.utils import get_first_env

BASE_URL = "https://apiservice.borsdata.se/v1"

load_dotenv(override=True)

KPI_SPECS_FULL = [
    {"name": "roic", "kpi_id": 37, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "ev_ebit", "kpi_id": 10, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "netdebt_ebitda", "kpi_id": 42, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "gross_profit_m", "kpi_id": 135, "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
    {"name": "total_assets_m", "kpi_id": 57, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "fcf_m", "kpi_id": 63, "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
    {"name": "mcap_m", "kpi_id": 50, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "ev_m", "kpi_id": 49, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "netdebt_m", "kpi_id": 60, "reporttypes": ["year", "r12"], "pricetype": "mean"},
    {"name": "ebit_m", "kpi_id": 55, "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
    {"name": "ebitda_m", "kpi_id": 54, "reporttypes": ["year", "r12", "quarter"], "pricetype": "mean"},
]
KPI_SPECS_CORE = [KPI_SPECS_FULL[0], KPI_SPECS_FULL[1], KPI_SPECS_FULL[2]]


def _auth_key() -> str:
    key = get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY", "BORSDATA_KEY"])
    if not key:
        raise RuntimeError("Mangler Borsdata API-key. Sett env var BORSDATA_AUTHKEY (evt BORSDATA_API_KEY).")
    return key
def _verify_tls() -> bool:
    return os.environ.get("BORSDATA_TLS_INSECURE", "").strip() != "1"



def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        status=10,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_SESSION = _make_session()


def _chunked(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _get_json(url: str, params: Dict[str, Any], timeout: int, max_retries: int) -> Any:
    """
    Robust GET:
    - Retry adapter hÃ¥ndterer 429/5xx
    - Denne wrapperen hÃ¥ndterer ConnectionReset/SSL/ProtocolError med backoff + jitter
    """
    backoff = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout, verify=_verify_tls())
            # Hvis adapteren ga oss en ikke-200 likevel:
            if r.status_code >= 400:
                # 401/403 skal ikke retries
                if r.status_code in (401, 403):
                    r.raise_for_status()
                # 429/5xx hÃ¥ndteres normalt av Retry, men hvis det likevel bobler opp:
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff + random.random())
                    backoff = min(backoff * 2, 30)
                    continue
                r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError, requests.exceptions.Timeout) as e:
            last_exc = e
            time.sleep(backoff + random.random())
            backoff = min(backoff * 2, 30)
            continue

    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP error uten exception (uventet)")


def _read_tickers_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = set(df.columns)

    if "yahoo_ticker" not in cols:
        raise ValueError(f"{path}: mÃ¥ ha kolonne yahoo_ticker")

    if "ins_id" not in cols and "ticker" not in cols:
        raise ValueError(f"{path}: mÃ¥ ha minst Ã©n av kolonnene ins_id eller ticker (i tillegg til yahoo_ticker).")

    for c in ["ticker", "yahoo_ticker", "country", "sector", "company"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "ins_id" in df.columns:
        df["ins_id"] = pd.to_numeric(df["ins_id"], errors="coerce")

    df = df[df["yahoo_ticker"].astype(str).str.strip().ne("")].copy()
    if "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.strip().ne("")].copy()

    for col in ["country", "sector", "company"]:
        if col not in df.columns:
            df[col] = ""

    keep = ["yahoo_ticker"]
    if "ticker" in df.columns:
        keep.append("ticker")
    if "ins_id" in df.columns:
        keep.append("ins_id")
    df = df.drop_duplicates(subset=keep)
    return df


def _fetch_instruments_map(authkey: str, timeout: int, max_retries: int) -> pd.DataFrame:
    url = f"{BASE_URL}/instruments"
    data = _get_json(url, params={"authKey": authkey}, timeout=timeout, max_retries=max_retries)

    if isinstance(data, dict):
        for k in ["instruments", "Instruments", "data", "Data"]:
            if k in data and isinstance(data[k], list):
                data = data[k]
                break

    if not isinstance(data, list):
        raise RuntimeError(f"Uventet instruments-respons: {type(data)}")

    df = pd.json_normalize(data)
    low = {c.lower(): c for c in df.columns}
    tcol = low.get("ticker")
    idcol = low.get("insid") or low.get("ins_id") or low.get("id")
    if not tcol or not idcol:
        raise RuntimeError(f"Fant ikke ticker/insId i instruments. Kolonner: {df.columns.tolist()[:40]}")
    out = df[[tcol, idcol]].rename(columns={tcol: "ticker", idcol: "ins_id"})
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["ins_id"] = pd.to_numeric(out["ins_id"], errors="coerce")
    out = out.dropna(subset=["ins_id"]).copy()
    out["ins_id"] = out["ins_id"].astype(int)
    return out.drop_duplicates(subset=["ticker"])


def _parse_kpi_history_payload(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _points(obj: Dict[str, Any]) -> List[Any]:
        for k in ["values", "history", "data", "items", "kpiHistory"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        return []

    def _date_val(pt: Dict[str, Any]):
        d = pt.get("d") or pt.get("date") or pt.get("reportEndDate") or pt.get("reportDate")
        v = pt.get("v") if "v" in pt else pt.get("value", pt.get("val"))
        return d, v

    if isinstance(payload, dict):
        for k in ["data", "Data", "instruments", "Instruments"]:
            if k in payload and isinstance(payload[k], list):
                payload = payload[k]
                break

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and any(k.lower() in ("insid", "ins_id", "id") for k in payload[0].keys()):
            for inst_obj in payload:
                if not isinstance(inst_obj, dict):
                    continue
                ins_id = inst_obj.get("insId") or inst_obj.get("ins_id") or inst_obj.get("InsId") or inst_obj.get("id")
                pts = _points(inst_obj)
                for pt in pts:
                    if isinstance(pt, dict):
                        d, v = _date_val(pt)
                        rows.append({"ins_id": ins_id, "date": d, "value": v})
        else:
            for pt in payload:
                if isinstance(pt, dict):
                    d, v = _date_val(pt)
                    rows.append({"ins_id": None, "date": d, "value": v})
    return rows


def _fetch_kpi_history_batch(authkey: str, ins_ids: List[int], kpi_id: int, reporttype: str, pricetype: str,
                            timeout: int, max_retries: int) -> pd.DataFrame:
    url = f"{BASE_URL}/Instruments/kpis/{kpi_id}/{reporttype}/{pricetype}/history"
    payload = _get_json(
        url,
        params={"authKey": authkey, "instList": ",".join(map(str, ins_ids))},
        timeout=timeout,
        max_retries=max_retries,
    )
    rows = _parse_kpi_history_payload(payload)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ins_id"] = pd.to_numeric(df["ins_id"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["ins_id", "date"])
    df["ins_id"] = df["ins_id"].astype(int)
    return df


def _save_partial(raw_dir: Path, asof: str, hist: pd.DataFrame, meta: dict, suffix: str) -> None:
    if hist is None or hist.empty:
        return
    outp = raw_dir / f"fundamentals_history_{suffix}.parquet"
    hist.to_parquet(outp, index=False)
    (raw_dir / f"ingest_meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[WARN] Saved partial -> {outp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tickers-csv", default=r"config\\tickers.csv")
    ap.add_argument("--min-periods", type=int, default=20)
    ap.add_argument("--mode", choices=["year", "quarter", "both"], default="both")
    ap.add_argument("--include-r12", action="store_true")
    ap.add_argument("--kpi-set", choices=["core", "full"], default="full")

    # NYE: stabilitet
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.25, help="pause mellom API-kall (sek)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max-retries", type=int, default=12)

    args = ap.parse_args()

    asof = args.asof
    raw_dir = Path("data") / "raw" / asof
    raw_dir.mkdir(parents=True, exist_ok=True)

    authkey = _auth_key()
    tickers_df = _read_tickers_csv(Path(args.tickers_csv))

    # Preferer ins_id hvis tilgjengelig
    if "ins_id" in tickers_df.columns and tickers_df["ins_id"].notna().any():
        merged = tickers_df.copy()
        merged["ins_id"] = pd.to_numeric(merged["ins_id"], errors="coerce")
        missing = merged[merged["ins_id"].isna()][[c for c in ["ticker", "yahoo_ticker"] if c in merged.columns]].copy()
        if not missing.empty:
            missing.to_csv(raw_dir / "missing_insid.csv", index=False)
            print(f"[WARN] {len(missing)} rader mangler ins_id. Se data/raw/{asof}/missing_insid.csv")
        merged = merged.dropna(subset=["ins_id"]).copy()
        merged["ins_id"] = merged["ins_id"].astype(int)
    else:
        instruments_map = _fetch_instruments_map(authkey, timeout=args.timeout, max_retries=args.max_retries)
        merged = tickers_df.merge(instruments_map, on="ticker", how="left")
        missing = merged[merged["ins_id"].isna()][["ticker", "yahoo_ticker"]]
        if not missing.empty:
            missing.to_csv(raw_dir / "missing_instruments.csv", index=False)
            print(f"[WARN] {len(missing)} tickers i tickers.csv matchet ikke /v1/instruments. Se data/raw/{asof}/missing_instruments.csv")
        merged = merged.dropna(subset=["ins_id"]).copy()
        merged["ins_id"] = merged["ins_id"].astype(int)

    merged.to_csv(raw_dir / "ticker_map.csv", index=False)

    want_year = args.mode in ("year", "both")
    want_quarter = args.mode in ("quarter", "both")
    want_r12 = bool(args.include_r12)

    target_reporttypes: List[str] = []
    if want_year:
        target_reporttypes.append("year")
    if want_quarter:
        target_reporttypes.append("quarter")
    if want_r12:
        target_reporttypes.append("r12")

    kpis = KPI_SPECS_CORE if args.kpi_set == "core" else KPI_SPECS_FULL

    ins_ids_all = merged["ins_id"].tolist()
    all_rows: List[pd.DataFrame] = []

    # progress counters
    total_calls = 0
    hist = None

    try:
        for spec in kpis:
            name = spec["name"]
            kpi_id = int(spec["kpi_id"])
            pricetype = spec.get("pricetype", "mean")
            available = set(spec["reporttypes"])

            for reporttype in target_reporttypes:
                if reporttype not in available:
                    continue

                for chunk in _chunked(ins_ids_all, args.chunk_size):
                    df = _fetch_kpi_history_batch(
                        authkey, chunk, kpi_id, reporttype, pricetype,
                        timeout=args.timeout, max_retries=args.max_retries
                    )
                    total_calls += 1
                    if not df.empty:
                        df["kpi_id"] = kpi_id
                        df["metric"] = name
                        df["report_type"] = reporttype
                        df["price_type"] = pricetype
                        all_rows.append(df)

                    if args.sleep > 0:
                        time.sleep(args.sleep)

        hist = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
            columns=["ins_id", "date", "value", "kpi_id", "metric", "report_type", "price_type"]
        )

        meta_cols = [c for c in ["ins_id", "ticker", "yahoo_ticker", "country", "sector", "company"] if c in merged.columns]
        hist = hist.merge(merged[meta_cols], on="ins_id", how="left")

        cols = [c for c in ["yahoo_ticker", "ticker", "ins_id", "company", "country", "sector",
                            "metric", "kpi_id", "report_type", "price_type", "date", "value"] if c in hist.columns]
        hist = hist[cols].sort_values(["yahoo_ticker", "metric", "report_type", "date"], kind="mergesort")

        out_path = raw_dir / "fundamentals_history.parquet"
        hist.to_parquet(out_path, index=False)

        coverage = (
            hist.dropna(subset=["value"])
            .groupby(["yahoo_ticker", "metric", "report_type"])["date"]
            .nunique()
            .reset_index(name="n_periods")
            .sort_values(["yahoo_ticker", "metric", "report_type"])
        )
        coverage.to_csv(raw_dir / "coverage.csv", index=False)

        core_metrics = ["roic", "ev_ebit", "netdebt_ebitda"]
        core = coverage[coverage["metric"].isin(core_metrics)]
        core_best = core[core["report_type"].isin(["r12", "year"])].groupby(["yahoo_ticker", "metric"])["n_periods"].max().reset_index()
        failing = core_best[core_best["n_periods"] < int(args.min_periods)].copy()
        failing.to_csv(raw_dir / "coverage_fail.csv", index=False)

        ingest_meta = {
            "asof": asof,
            "mode": args.mode,
            "include_r12": bool(args.include_r12),
            "kpi_set": args.kpi_set,
            "min_periods": int(args.min_periods),
            "rows": int(len(hist)),
            "tickers_in": int(len(tickers_df)),
            "tickers_used": int(len(merged)),
            "chunk_size": int(args.chunk_size),
            "sleep": float(args.sleep),
            "timeout": int(args.timeout),
            "max_retries": int(args.max_retries),
            "api_calls": int(total_calls),
            "base_url": BASE_URL,
        }
        (raw_dir / "ingest_meta.json").write_text(json.dumps(ingest_meta, indent=2), encoding="utf-8")

        print(f"[OK] wrote: {out_path}")
        print(f"[OK] wrote: {raw_dir / 'ingest_meta.json'}")
        if len(failing):
            print(f"[WARN] coverage_fail rows={len(failing)} -> {raw_dir / 'coverage_fail.csv'}")

    except KeyboardInterrupt:
        # lagre partial og avslutt pent
        if all_rows:
            hist = pd.concat(all_rows, ignore_index=True)
            ingest_meta = {"asof": asof, "partial": True, "api_calls": int(total_calls)}
            _save_partial(raw_dir, asof, hist, ingest_meta, "partial")
        raise
    except Exception as e:
        if all_rows:
            hist = pd.concat(all_rows, ignore_index=True)
            ingest_meta = {"asof": asof, "partial": True, "api_calls": int(total_calls), "error": str(e)}
            _save_partial(raw_dir, asof, hist, ingest_meta, "crash")
        raise


if __name__ == "__main__":
    main()




