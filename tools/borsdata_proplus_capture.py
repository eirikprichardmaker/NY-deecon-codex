from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests

API_BASE = "https://apiservice.borsdata.se/v1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chunks(values: Sequence[int], size: int) -> Iterable[Tuple[int, List[int]]]:
    i = 0
    batch = 0
    n = len(values)
    while i < n:
        batch += 1
        yield batch, list(values[i : i + size])
        i += size


def write_json_gz(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def first_list_len(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return len(v)
        for v in payload.values():
            if isinstance(v, dict):
                out = first_list_len(v)
                if out >= 0:
                    return out
    return -1


class ApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureConfig:
    asof: str
    out_root: Path
    auth_env: str
    batch_size: int
    pace_s: float
    max_attempts: int
    global_date_start: str
    global_date_end: str
    kpi_ids: Tuple[int, ...]
    include_calendar_batches: bool
    include_global_date_series: bool


class BorsdataClient:
    def __init__(self, auth_key: str, max_attempts: int, pace_s: float):
        self.auth_key = auth_key
        self.max_attempts = max(1, int(max_attempts))
        self.pace_s = max(0.0, float(pace_s))
        self.session = requests.Session()

    def get_json(self, path: str, params: Dict[str, Any] | None = None) -> Tuple[Any, Dict[str, Any]]:
        p = dict(params or {})
        p["authKey"] = self.auth_key
        url = f"{API_BASE}{path}"
        delay = 1.0
        last_err: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                resp = self.session.get(url, params=p, timeout=90)
                elapsed = round(time.time() - t0, 3)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = delay
                    if retry_after and retry_after.isdigit():
                        sleep_s = max(float(retry_after), delay)
                    time.sleep(min(sleep_s, 60.0))
                    delay = min(delay * 1.8, 60.0)
                    continue

                if 500 <= resp.status_code < 600:
                    time.sleep(min(delay, 60.0))
                    delay = min(delay * 1.8, 60.0)
                    continue

                if resp.status_code >= 400:
                    raise ApiError(f"{path} HTTP {resp.status_code}: {resp.text[:200]}")

                js = resp.json()
                meta = {
                    "path": path,
                    "status_code": resp.status_code,
                    "elapsed_s": elapsed,
                    "attempt": attempt,
                    "fetched_at_utc": utc_now_iso(),
                }
                if self.pace_s > 0:
                    time.sleep(self.pace_s)
                return js, meta

            except Exception as exc:  # requests + json decode
                last_err = exc
                time.sleep(min(delay, 60.0))
                delay = min(delay * 1.8, 60.0)

        raise ApiError(f"GET failed after {self.max_attempts} attempts: {path} err={last_err}")


def build_global_ids(payload: Any) -> List[int]:
    rows: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        maybe = payload.get("instruments")
        if isinstance(maybe, list):
            rows = [x for x in maybe if isinstance(x, dict)]
    elif isinstance(payload, list):
        rows = [x for x in payload if isinstance(x, dict)]
    out: List[int] = []
    seen = set()
    for row in rows:
        raw = row.get("insId") or row.get("id")
        if raw is None:
            continue
        try:
            ins_id = int(raw)
        except Exception:
            continue
        if ins_id in seen:
            continue
        seen.add(ins_id)
        out.append(ins_id)
    return out


def write_global_ids_csv(path: Path, ids: Sequence[int]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ins_id"])
        for ins_id in ids:
            w.writerow([int(ins_id)])


def weekday_dates(start_s: str, end_s: str) -> List[str]:
    start_dt = datetime.strptime(start_s, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_s, "%Y-%m-%d").date()
    if end_dt < start_dt:
        raise ValueError(f"global-date-end {end_s} is before global-date-start {start_s}")
    out: List[str] = []
    d = start_dt
    while d <= end_dt:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def capture_meta_and_updated(cfg: CaptureConfig, client: BorsdataClient, run_dir: Path, global_ids: Sequence[int]) -> Dict[str, int]:
    out_dir = run_dir / "meta_updated"
    ensure_dir(out_dir)
    stats = {"files": 0, "calls": 0}

    static_calls = [
        ("countries", "/countries", {}),
        ("markets", "/markets", {}),
        ("sectors", "/sectors", {}),
        ("branches", "/branches", {}),
        ("translationmetadata", "/translationmetadata", {}),
        ("instruments_updated", "/instruments/updated", {}),
        ("kpis_metadata", "/instruments/kpis/metadata", {}),
        ("kpis_updated", "/instruments/kpis/updated", {}),
        ("reports_metadata", "/instruments/reports/metadata", {}),
    ]

    for name, path, params in static_calls:
        payload, meta = client.get_json(path, params)
        row_count = first_list_len(payload)
        write_json_gz(
            out_dir / f"{name}.json.gz",
            {
                "meta": meta,
                "params": params,
                "row_count_hint": row_count,
                "payload": payload,
            },
        )
        print(f"[meta] {name}: rows={row_count}")
        stats["files"] += 1
        stats["calls"] += 1

    if not cfg.include_calendar_batches:
        return stats

    cal_dir = out_dir / "calendar_batches"
    ensure_dir(cal_dir)
    n_batches = (len(global_ids) + cfg.batch_size - 1) // cfg.batch_size

    for batch_no, batch_ids in chunks(global_ids, cfg.batch_size):
        inst_list = ",".join(str(x) for x in batch_ids)
        for name, path in [
            ("report_calendar", "/instruments/report/calendar"),
            ("dividend_calendar", "/instruments/dividend/calendar"),
            ("description", "/instruments/description"),
        ]:
            params = {"instList": inst_list}
            payload, meta = client.get_json(path, params)
            row_count = first_list_len(payload)
            write_json_gz(
                cal_dir / f"{name}__batch_{batch_no:04d}.json.gz",
                {
                    "meta": meta,
                    "params": {"instList_count": len(batch_ids)},
                    "row_count_hint": row_count,
                    "payload": payload,
                },
            )
            if batch_no == 1 or batch_no % 50 == 0 or batch_no == n_batches:
                print(f"[meta] {name} batch {batch_no}/{n_batches}: rows={row_count}")
            stats["files"] += 1
            stats["calls"] += 1

    return stats


def capture_global_price_snapshots(cfg: CaptureConfig, client: BorsdataClient, run_dir: Path) -> Dict[str, int]:
    out_dir = run_dir / "global_prices"
    ensure_dir(out_dir)
    stats = {"files": 0, "calls": 0, "dates": 0}

    last_payload, last_meta = client.get_json("/instruments/stockprices/global/last", {})
    last_rows = first_list_len(last_payload)
    write_json_gz(
        out_dir / "global_last.json.gz",
        {
            "meta": last_meta,
            "row_count_hint": last_rows,
            "payload": last_payload,
        },
    )
    print(f"[global_prices] global_last rows={last_rows}")
    stats["files"] += 1
    stats["calls"] += 1

    if not cfg.include_global_date_series:
        return stats

    dates = weekday_dates(cfg.global_date_start, cfg.global_date_end)
    n_dates = len(dates)
    date_dir = out_dir / "by_date"
    ensure_dir(date_dir)
    for idx, d in enumerate(dates, start=1):
        payload, meta = client.get_json("/instruments/stockprices/global/date", {"date": d})
        row_count = first_list_len(payload)
        write_json_gz(
            date_dir / f"date_{d}.json.gz",
            {
                "meta": meta,
                "date": d,
                "row_count_hint": row_count,
                "payload": payload,
            },
        )
        if idx == 1 or idx % 25 == 0 or idx == n_dates:
            print(f"[global_prices] date {idx}/{n_dates} ({d}) rows={row_count}")
        stats["files"] += 1
        stats["calls"] += 1
        stats["dates"] += 1

    return stats


def capture_global_kpi_summary(cfg: CaptureConfig, client: BorsdataClient, run_dir: Path) -> Dict[str, int]:
    out_dir = run_dir / "global_kpi_summary"
    ensure_dir(out_dir)
    stats = {"files": 0, "calls": 0}
    combos = [("1year", "mean"), ("last", "latest")]

    for kpi_id in cfg.kpi_ids:
        for calc_group, calc in combos:
            path = f"/instruments/global/kpis/{int(kpi_id)}/{calc_group}/{calc}"
            payload, meta = client.get_json(path, {})
            row_count = first_list_len(payload)
            write_json_gz(
                out_dir / f"kpi_{int(kpi_id)}__{calc_group}_{calc}.json.gz",
                {
                    "meta": meta,
                    "kpi_id": int(kpi_id),
                    "calc_group": calc_group,
                    "calc": calc,
                    "row_count_hint": row_count,
                    "payload": payload,
                },
            )
            print(f"[kpi_summary] kpi={int(kpi_id)} {calc_group}/{calc} rows={row_count}")
            stats["files"] += 1
            stats["calls"] += 1

    return stats


def parse_kpi_ids(raw: str) -> Tuple[int, ...]:
    out: List[int] = []
    seen = set()
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        v = int(p)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return tuple(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture Pro+ metadata, global master/prices and global KPI summaries for audit-safe archival."
    )
    p.add_argument("--asof", default=date.today().isoformat(), help="YYYY-MM-DD")
    p.add_argument("--out-root", default="data/freeze/borsdata_proplus")
    p.add_argument("--auth-env", default="BORSDATA_AUTHKEY")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--pace-s", type=float, default=0.05)
    p.add_argument("--max-attempts", type=int, default=8)
    p.add_argument("--global-date-start", default="2025-01-01")
    p.add_argument("--global-date-end", default=None)
    p.add_argument(
        "--kpi-ids",
        default="37,42,49,50,53,55,57,60,62,63,66",
        help="Comma-separated KPI IDs for global KPI summary capture.",
    )
    p.add_argument(
        "--steps",
        default="meta,global_prices,kpi_summary",
        help="Comma-separated: meta,global_prices,kpi_summary",
    )
    p.add_argument("--skip-calendar-batches", action="store_true")
    p.add_argument("--skip-global-date-series", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(Path(".env"))
    auth_key = os.getenv(args.auth_env) or os.getenv("BORSDATA_API_KEY")
    if not auth_key:
        raise SystemExit(f"Missing API key. Set {args.auth_env} (or BORSDATA_API_KEY).")

    asof = str(args.asof)
    out_root = Path(args.out_root)
    run_dir = out_root / asof / "meta_global_capture"
    ensure_dir(run_dir)

    global_date_end = str(args.global_date_end).strip() if args.global_date_end else asof
    cfg = CaptureConfig(
        asof=asof,
        out_root=out_root,
        auth_env=str(args.auth_env),
        batch_size=max(1, int(args.batch_size)),
        pace_s=max(0.0, float(args.pace_s)),
        max_attempts=max(1, int(args.max_attempts)),
        global_date_start=str(args.global_date_start),
        global_date_end=global_date_end,
        kpi_ids=parse_kpi_ids(str(args.kpi_ids)),
        include_calendar_batches=not bool(args.skip_calendar_batches),
        include_global_date_series=not bool(args.skip_global_date_series),
    )

    steps = {s.strip() for s in str(args.steps).split(",") if s.strip()}
    allowed = {"meta", "global_prices", "kpi_summary"}
    bad_steps = sorted(steps - allowed)
    if bad_steps:
        raise SystemExit(f"Unsupported --steps: {bad_steps}. Allowed: {sorted(allowed)}")

    client = BorsdataClient(auth_key=auth_key, max_attempts=cfg.max_attempts, pace_s=cfg.pace_s)

    instruments_global, meta_global = client.get_json("/instruments/global", {})
    instruments_nordic, meta_nordic = client.get_json("/instruments", {"includeDelisted": "false"})
    write_json_gz(
        run_dir / "instrument_master_global.json.gz",
        {"meta": meta_global, "row_count_hint": first_list_len(instruments_global), "payload": instruments_global},
    )
    write_json_gz(
        run_dir / "instrument_master_nordic.json.gz",
        {"meta": meta_nordic, "row_count_hint": first_list_len(instruments_nordic), "payload": instruments_nordic},
    )

    global_ids = build_global_ids(instruments_global)
    ids_csv = run_dir / "global_ids.csv"
    write_global_ids_csv(ids_csv, global_ids)
    print(f"[master] global instruments={len(global_ids)} csv={ids_csv}")

    summary: Dict[str, Any] = {
        "asof": asof,
        "created_at_utc": utc_now_iso(),
        "out_dir": str(run_dir),
        "global_ids_csv": str(ids_csv),
        "global_instrument_count": len(global_ids),
        "steps": sorted(steps),
        "kpi_ids": list(cfg.kpi_ids),
        "stats": {},
    }

    if "meta" in steps:
        summary["stats"]["meta"] = capture_meta_and_updated(cfg, client, run_dir, global_ids)
    if "global_prices" in steps:
        summary["stats"]["global_prices"] = capture_global_price_snapshots(cfg, client, run_dir)
    if "kpi_summary" in steps:
        summary["stats"]["kpi_summary"] = capture_global_kpi_summary(cfg, client, run_dir)

    summary["finished_at_utc"] = utc_now_iso()
    (run_dir / "capture_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
