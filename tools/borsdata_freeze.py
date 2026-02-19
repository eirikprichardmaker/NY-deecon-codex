from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

API_BASE = "https://apiservice.borsdata.se/v1"
MAX_INSTLIST = 50  # safe default for array endpoints


# -----------------------------
# Utils
# -----------------------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_ins_ids(csv_path: Path, col: str) -> List[int]:
    ins_ids: List[int] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            try:
                ins_ids.append(int(float(raw)))
            except ValueError:
                continue
    # unique, stable
    seen = set()
    out = []
    for x in ins_ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def chunks(lst: List[int], n: int) -> Iterable[Tuple[int, List[int]]]:
    i = 0
    batch = 0
    while i < len(lst):
        batch += 1
        yield batch, lst[i : i + n]
        i += n


def gz_json_write(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def gz_jsonl_append(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    n = 0
    with gzip.open(path, "at", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def safe_filename(s: str) -> str:
    # conservative: keep alnum, dash, underscore, dot
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


# -----------------------------
# State / resume
# -----------------------------
@dataclass
class FreezeState:
    done: Dict[str, Dict[str, Any]]

    @staticmethod
    def load(path: Path) -> "FreezeState":
        if path.exists():
            return FreezeState(done=json.loads(path.read_text(encoding="utf-8")).get("done", {}))
        return FreezeState(done={})

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        payload = {"done": self.done, "saved_at_utc": utc_now().isoformat(timespec="seconds")}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str, meta: Dict[str, Any]) -> None:
        self.done[key] = meta


# -----------------------------
# HTTP with retry/backoff + 429 handling
# -----------------------------
def borsdata_get(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    tries: int = 8,
    timeout: int = 60,
    min_sleep: float = 0.2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (json_payload, response_meta)
    response_meta includes status_code, elapsed_s, headers_subset
    """
    delay = 1.0
    last_err: Optional[Exception] = None

    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            elapsed = time.time() - t0

            # Rate limit
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(float(retry_after), min_sleep)
                    except ValueError:
                        sleep_s = max(delay, min_sleep)
                else:
                    sleep_s = max(delay, min_sleep)
                time.sleep(min(sleep_s, 60.0))
                delay = min(delay * 1.8, 30.0)
                continue

            resp.raise_for_status()
            js = resp.json()

            meta = {
                "status_code": resp.status_code,
                "elapsed_s": round(elapsed, 3),
                "headers": {
                    "Date": resp.headers.get("Date"),
                    "Retry-After": resp.headers.get("Retry-After"),
                    "Content-Type": resp.headers.get("Content-Type"),
                },
            }
            # Gentle pacing
            time.sleep(min_sleep)
            return js, meta

        except Exception as e:
            last_err = e
            time.sleep(max(delay, min_sleep))
            delay = min(delay * 1.8, 30.0)

    raise RuntimeError(f"GET failed after {tries} tries: {url} err={last_err}")


# -----------------------------
# Normalization (best-effort)
# -----------------------------
def find_first_list(payload: Any) -> Optional[List[Any]]:
    """Heuristic: find the first list of dict-ish items in a nested JSON payload."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return v
        # search deeper (limited)
        for v in payload.values():
            if isinstance(v, dict):
                lst = find_first_list(v)
                if lst is not None:
                    return lst
    return None


def normalize_reports(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Make JSONL rows with:
    - ins_id (if present)
    - report_type (if present)
    - item (raw item flattened minimally)
    """
    lst = find_first_list(payload)
    if not lst:
        return []

    rows = []
    for item in lst:
        if not isinstance(item, dict):
            continue
        # common keys
        ins_id = item.get("insId") or item.get("instrumentId") or item.get("ins_id")
        report_type = item.get("reportType") or item.get("type") or item.get("report_type")
        rows.append(
            {
                "dataset": "reports",
                "ins_id": ins_id,
                "report_type": report_type,
                "fetched_at_utc": utc_now().isoformat(timespec="seconds"),
                "item": item,  # keep full item to avoid schema loss
            }
        )
    return rows


def normalize_prices(payload: Dict[str, Any], from_date: str, to_date: str) -> Iterable[Dict[str, Any]]:
    lst = find_first_list(payload)
    if not lst:
        return []
    rows = []
    for item in lst:
        if not isinstance(item, dict):
            continue
        ins_id = item.get("insId") or item.get("instrumentId") or item.get("ins_id")
        # many schemas: points in nested list
        points = None
        for kk in ("values", "history", "data", "items", "points", "stockPrices"):
            if isinstance(item.get(kk), list):
                points = item.get(kk)
                break
        rows.append(
            {
                "dataset": "stockprices",
                "ins_id": ins_id,
                "from": from_date,
                "to": to_date,
                "fetched_at_utc": utc_now().isoformat(timespec="seconds"),
                "item": item if points is None else {"meta": {k: v for k, v in item.items() if k != kk}, "points": points},
            }
        )
    return rows


def normalize_kpi_history(payload: Dict[str, Any], kpi_id: int, period: str, value_type: str) -> Iterable[Dict[str, Any]]:
    # KPI History commonly has kpisList
    kpis_list = payload.get("kpisList")
    if not isinstance(kpis_list, list):
        lst = find_first_list(payload)
        kpis_list = lst if isinstance(lst, list) else []
    rows = []
    for item in kpis_list:
        if not isinstance(item, dict):
            continue
        ins_id = item.get("insId") or item.get("instrumentId") or item.get("ins_id")
        rows.append(
            {
                "dataset": "kpi_history",
                "kpi_id": kpi_id,
                "period": period,
                "value_type": value_type,
                "ins_id": ins_id,
                "fetched_at_utc": utc_now().isoformat(timespec="seconds"),
                "item": item,
            }
        )
    return rows


# -----------------------------
# Freeze Runner
# -----------------------------
@dataclass
class FreezeConfig:
    auth_env: str
    asof: str
    ids_csv: Path
    ids_col: str
    out_root: Path
    batch_size: int
    pace_s: float

    # Reports
    do_reports: bool
    reports_original: int
    reports_maxcount: int

    do_reports_pi: bool
    reports_pi_maxcount: int
    # Prices
    do_prices: bool
    prices_years_back: int
    prices_window_days: int
    do_prices_last: bool

    # KPI History
    do_kpi_history: bool
    kpi_ids: List[int]
    kpi_period: str
    kpi_value_type: str

    # Holdings
    do_holdings: bool


def build_config(args: argparse.Namespace) -> FreezeConfig:
    asof = args.asof or datetime.now().date().isoformat()
    out_root = Path(args.out_root)

    kpi_ids = []
    if args.kpi_ids:
        kpi_ids = [int(x.strip()) for x in args.kpi_ids.split(",") if x.strip()]

    do_reports_pi = bool(args.reports_per_instrument) and (not args.only or "reports_pi" in args.only)

    return FreezeConfig(
        auth_env=args.auth_env,
        asof=asof,
        ids_csv=Path(args.ids_csv),
        ids_col=args.ids_col,
        out_root=out_root,
        batch_size=min(max(1, args.batch_size), MAX_INSTLIST),
        pace_s=max(0.0, args.pace_s),
        do_reports=not args.only or "reports" in args.only,
        reports_original=1 if args.reports_original else 0,
        reports_maxcount=args.reports_maxcount,
        do_reports_pi=do_reports_pi,
        reports_pi_maxcount=args.reports_pi_maxcount,
        do_prices=not args.only or "prices" in args.only,
        prices_years_back=args.prices_years_back,
        prices_window_days=args.prices_window_days,
        do_prices_last=not args.only or "prices_last" in args.only,
        do_kpi_history=(not args.only or "kpi" in args.only) and bool(kpi_ids),
        kpi_ids=kpi_ids,
        kpi_period=args.kpi_period,
        kpi_value_type=args.kpi_value_type,
        do_holdings=(not args.only or "holdings" in args.only) and args.holdings,
    )


def write_manifest(base_dir: Path, cfg: FreezeConfig, ins_count: int) -> None:
    manifest = {
        "asof": cfg.asof,
        "created_at_utc": utc_now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "auth_env": cfg.auth_env,
        "instrument_count": ins_count,
        "batch_size": cfg.batch_size,
        "pace_s": cfg.pace_s,
        "steps": {
            "reports": cfg.do_reports,
            "reports_per_instrument": cfg.do_reports_pi,
            "prices": cfg.do_prices,
            "prices_last": cfg.do_prices_last,
            "kpi_history": cfg.do_kpi_history,
            "holdings": cfg.do_holdings,
        },
        "reports": {"original": cfg.reports_original, "maxcount": cfg.reports_maxcount},
        "reports_per_instrument": {"original": cfg.reports_original, "maxcount": cfg.reports_pi_maxcount},
        "prices": {"years_back": cfg.prices_years_back, "window_days": cfg.prices_window_days},
        "kpi_history": {"kpi_ids": cfg.kpi_ids, "period": cfg.kpi_period, "value_type": cfg.kpi_value_type},
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_freeze(cfg: FreezeConfig) -> None:
    auth = os.getenv(cfg.auth_env)
    if not auth:
        raise SystemExit(f"Mangler API key i env var: {cfg.auth_env}")

    # Layout
    base_dir = cfg.out_root / cfg.asof
    raw_dir = base_dir / "raw"
    norm_dir = base_dir / "normalized"
    ensure_dir(raw_dir)
    ensure_dir(norm_dir)

    state_path = base_dir / "state.json"
    state = FreezeState.load(state_path)

    ins_ids = read_ins_ids(cfg.ids_csv, cfg.ids_col)
    write_manifest(base_dir, cfg, len(ins_ids))

    session = requests.Session()

    # ------------------ Reports (intrinsic core) ------------------
    # Best coverage per call: /v1/instruments/reports?instList=...&original=1&maxcount=...
    if cfg.do_reports:
        endpoint = f"{API_BASE}/instruments/reports"
        out_raw = raw_dir / "reports"
        out_norm = norm_dir / "reports.jsonl.gz"
        ensure_dir(out_raw)

        for batch_no, batch_ids in chunks(ins_ids, cfg.batch_size):
            key = f"reports|batch={batch_no}|n={len(batch_ids)}|original={cfg.reports_original}|maxcount={cfg.reports_maxcount}"
            if state.is_done(key):
                continue

            inst_list = ",".join(str(x) for x in batch_ids)
            params = {
                "authKey": auth,
                "instList": inst_list,
                "original": cfg.reports_original,
                "maxcount": cfg.reports_maxcount,
            }
            payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)

            raw_path = out_raw / f"batch_{batch_no:04d}.json.gz"
            gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

            # normalize best-effort
            nrows = gz_jsonl_append(out_norm, normalize_reports(payload))

            state.mark_done(
                key,
                {"raw_file": str(raw_path), "normalized_rows": nrows, "done_at_utc": utc_now().isoformat(timespec="seconds")},
            )
            state.save(state_path)

        # ------------------ Reports per instrument (20y via &maxcount) ------------------

    # ------------------ Reports per instrument (20y via reporttype endpoints) ------------------
    if cfg.do_reports_pi:
        out_raw = raw_dir / "reports_per_instrument"
        out_norm = norm_dir / "reports_per_instrument.jsonl.gz"
        ensure_dir(out_raw)

        reporttypes = ["year", "r12", "quarter"]
        for ins_id in ins_ids:
            for rt in reporttypes:
                key = f"reports_pi|rt={rt}|ins_id={ins_id}|original={cfg.reports_original}|maxcount={cfg.reports_pi_maxcount}"
                if state.is_done(key):
                    continue

                endpoint = f"{API_BASE}/instruments/{ins_id}/reports/{rt}"
                params = {
                    "authKey": auth,
                    "original": cfg.reports_original,
                    "maxcount": cfg.reports_pi_maxcount,
                }
                payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)

                raw_path = out_raw / f"ins_{ins_id}__{rt}.json.gz"
                gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

                nrows = gz_jsonl_append(
                    out_norm,
                    [{
                        "dataset": "reports_per_instrument",
                        "rt": rt,
                        "ins_id": ins_id,
                        "fetched_at_utc": utc_now().isoformat(timespec="seconds"),
                        "payload": payload,
                    }],
                )

                state.mark_done(
                    key,
                    {"raw_file": str(raw_path), "normalized_rows": nrows, "done_at_utc": utc_now().isoformat(timespec="seconds")},
                )
                state.save(state_path)

    # ------------------ Stockprices last (latest snapshot) ------------------
    if cfg.do_prices_last:
        endpoint = f"{API_BASE}/instruments/stockprices/last"
        out_raw = raw_dir / "stockprices_last"
        out_norm = norm_dir / "stockprices_last.jsonl.gz"
        ensure_dir(out_raw)

        for batch_no, batch_ids in chunks(ins_ids, cfg.batch_size):
            key = f"prices_last|batch={batch_no}|n={len(batch_ids)}"
            if state.is_done(key):
                continue

            inst_list = ",".join(str(x) for x in batch_ids)
            params = {"authKey": auth, "instList": inst_list}
            payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)

            raw_path = out_raw / f"batch_{batch_no:04d}.json.gz"
            gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

            nrows = gz_jsonl_append(out_norm, [{"dataset": "stockprices_last", "fetched_at_utc": utc_now().isoformat(timespec="seconds"), "payload": payload}])

            state.mark_done(
                key,
                {"raw_file": str(raw_path), "normalized_rows": nrows, "done_at_utc": utc_now().isoformat(timespec="seconds")},
            )
            state.save(state_path)

    # ------------------ KPI History (optional) ------------------
    if cfg.do_kpi_history:
        out_raw = raw_dir / "kpi_history"
        out_norm = norm_dir / "kpi_history.jsonl.gz"
        ensure_dir(out_raw)

        for kpi_id in cfg.kpi_ids:
            endpoint = f"{API_BASE}/Instruments/kpis/{kpi_id}/{cfg.kpi_period}/{cfg.kpi_value_type}/history"
            for batch_no, batch_ids in chunks(ins_ids, cfg.batch_size):
                key = f"kpi|kpi={kpi_id}|period={cfg.kpi_period}|vt={cfg.kpi_value_type}|batch={batch_no}|n={len(batch_ids)}"
                if state.is_done(key):
                    continue

                inst_list = ",".join(str(x) for x in batch_ids)
                params = {"authKey": auth, "instList": inst_list}
                payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)

                raw_path = out_raw / f"kpi_{kpi_id}__{cfg.kpi_period}_{cfg.kpi_value_type}__batch_{batch_no:04d}.json.gz"
                gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

                nrows = gz_jsonl_append(out_norm, normalize_kpi_history(payload, kpi_id, cfg.kpi_period, cfg.kpi_value_type))

                state.mark_done(
                    key,
                    {"raw_file": str(raw_path), "normalized_rows": nrows, "done_at_utc": utc_now().isoformat(timespec="seconds")},
                )
                state.save(state_path)

    # ------------------ Holdings (optional, big) ------------------
    if cfg.do_holdings:
        out_raw = raw_dir / "holdings"
        out_norm = norm_dir / "holdings.jsonl.gz"
        ensure_dir(out_raw)

        for name in ("insider", "shorts", "buyback"):
            endpoint = f"{API_BASE}/holdings/{name}"
            key = f"holdings|{name}"
            if state.is_done(key):
                continue

            params = {"authKey": auth}
            payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)
            raw_path = out_raw / f"{name}.json.gz"
            gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

            nrows = gz_jsonl_append(
                out_norm,
                [{"dataset": f"holdings_{name}", "fetched_at_utc": utc_now().isoformat(timespec="seconds"), "payload": payload}],
            )

            state.mark_done(
                key,
                {"raw_file": str(raw_path), "normalized_rows": nrows, "done_at_utc": utc_now().isoformat(timespec="seconds")},
            )
            state.save(state_path)

    # Summary
    summary = {
        "asof": cfg.asof,
        "finished_at_utc": utc_now().isoformat(timespec="seconds"),
        "done_keys": len(state.done),
        "base_dir": str(base_dir),
    }
    (base_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Borsdata freeze: reports + prices + optional KPI/holdings. Resumable.")
    ap.add_argument("--auth-env", default="BORSDATA_AUTHKEY")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--ids-csv", required=True)
    ap.add_argument("--ids-col", default="ins_id")
    ap.add_argument("--out-root", default="data/freeze/borsdata")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--pace-s", type=float, default=0.2)

    # reports
    ap.add_argument("--reports-original", type=int, default=1, choices=[0, 1])
    ap.add_argument("--reports-maxcount", type=int, default=20)
    ap.add_argument(
        "--reports-per-instrument",
        action="store_true",
        help="Download reports per instrument using /v1/instruments/{id}/reports with &maxcount (needed for true 20y).",
    )
    ap.add_argument(
        "--reports-pi-maxcount",
        type=int,
        default=20,
        help="maxcount for per-instrument reports (20y year reports / 40 interim).",
    )

    # prices
    ap.add_argument("--prices-years-back", type=int, default=20)
    ap.add_argument("--prices-window-days", type=int, default=365)
    ap.add_argument("--no-prices-last", action="store_true")

    # kpi
    ap.add_argument("--kpi-ids", default="", help="Comma-separated KPI IDs, e.g. 37,33,31")
    ap.add_argument("--kpi-period", default="year", choices=["year", "quarter", "r12"])
    ap.add_argument("--kpi-value-type", default="mean")

    # holdings
    ap.add_argument("--holdings", action="store_true")

    # subset
    ap.add_argument(
        "--only",
        default="",
        help="Run only specific steps: comma-separated among reports,reports_pi,prices,prices_last,kpi,holdings",
    )

    args = ap.parse_args()
    if args.only.strip():
        args.only = {x.strip() for x in args.only.split(",") if x.strip()}
    else:
        args.only = set()
    return args


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    # if user says no-prices-last, override
    if args.no_prices_last:
        cfg.do_prices_last = False
    run_freeze(cfg)


if __name__ == "__main__":
    main()
