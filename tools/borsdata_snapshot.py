import os, json, time, gzip, hashlib, argparse
from datetime import datetime
from pathlib import Path
import requests

API_BASE = "https://apiservice.borsdata.se/v1"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_ids_csv(path: Path, col: str) -> list[int]:
    # minimal CSV reader (no pandas dependency)
    import csv
    ids = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            v = (row.get(col) or "").strip()
            if v:
                try:
                    ids.append(int(float(v)))
                except ValueError:
                    pass
    # uniq + stable order
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def request_json(session: requests.Session, url: str, params: dict, tries=6, timeout=45):
    delay = 1.5
    last_err = None
    for attempt in range(1, tries+1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                # rate limited
                time.sleep(delay)
                delay = min(delay * 1.8, 30)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 1.8, 30)
    raise RuntimeError(f"Failed after {tries} tries: {url} err={last_err}")

def parse_points(payload: dict, kpi_id: int, period: str, value_type: str):
    """
    Best-effort parsing. Uansett lagres rådata; parsing er bonus.
    Returnerer liste av dict-rader.
    """
    rows = []
    kpis_list = payload.get("kpisList") or []
    for item in kpis_list:
        ins_id = item.get("insId") or item.get("instrumentId") or item.get("insid") or item.get("ins_id")
        points = None
        for kk in ("values", "history", "data", "items", "points", "kpiHistory"):
            if isinstance(item.get(kk), list):
                points = item[kk]
                break
        if not ins_id or not points:
            continue
        for p in points:
            # ulike skjema i ulike endepunkt → heuristikk
            year = p.get("year") or p.get("y") or p.get("x") or p.get("period")
            val = p.get("value") or p.get("v") or p.get("yValue") or p.get("val")
            if year is None or val is None:
                continue
            rows.append({
                "ins_id": int(ins_id),
                "kpi_id": int(kpi_id),
                "period": period,
                "value_type": value_type,
                "point": year,
                "value": val,
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth-env", default="BORSDATA_AUTHKEY")
    ap.add_argument("--ids-csv", required=True, help="Path to CSV containing ins_id column")
    ap.add_argument("--ids-col", default="ins_id")
    ap.add_argument("--kpis", required=True, help="Comma-separated KPI IDs, e.g. 37,8,12")
    ap.add_argument("--period", default="year", choices=["year","quarter","r12"])
    ap.add_argument("--value-type", default="mean")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--out", default="data/snapshots/borsdata")
    ap.add_argument("--state", default="data/snapshots/borsdata/state.json")
    ap.add_argument("--parse-out", default="data/snapshots/borsdata/parsed.jsonl.gz")
    args = ap.parse_args()

    auth = os.getenv(args.auth_env)
    if not auth:
        raise SystemExit(f"Missing API key in env var: {args.auth_env}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"done": {}}  # key -> {file, fetched_at}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    ins_ids = load_ids_csv(Path(args.ids_csv), args.ids_col)
    kpi_ids = [int(x.strip()) for x in args.kpis.split(",") if x.strip()]

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session = requests.Session()

    parsed_rows_total = 0
    parse_out = Path(args.parse_out)
    parse_out.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(parse_out, "at", encoding="utf-8") as parsed_f:
        for kpi_id in kpi_ids:
            url = f"{API_BASE}/Instruments/kpis/{kpi_id}/{args.period}/{args.value_type}/history"
            for bi, batch in enumerate(chunked(ins_ids, args.batch_size), start=1):
                inst_list = ",".join(str(x) for x in batch)
                key = f"kpi={kpi_id}|period={args.period}|vt={args.value_type}|batch={bi}|n={len(batch)}"
                if key in state["done"]:
                    continue

                params = {"authKey": auth, "instList": inst_list}
                payload = request_json(session, url, params=params)

                raw_path = out_dir / f"{run_id}__{key.replace('|','__').replace('=','-')}.json.gz"
                with gzip.open(raw_path, "wt", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)

                # best-effort parse
                rows = parse_points(payload, kpi_id, args.period, args.value_type)
                for r in rows:
                    r["fetched_at_utc"] = datetime.utcnow().isoformat(timespec="seconds")
                    parsed_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                parsed_rows_total += len(rows)

                state["done"][key] = {
                    "file": str(raw_path),
                    "fetched_at_utc": datetime.utcnow().isoformat(timespec="seconds")
                }
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

                # mild pacing to be nice to API (juster ved behov)
                time.sleep(0.2)

    # manifest
    manifest = {
        "run_id": run_id,
        "kpi_ids": kpi_ids,
        "period": args.period,
        "value_type": args.value_type,
        "instrument_count": len(ins_ids),
        "parsed_rows_appended": parsed_rows_total,
        "state_file": str(state_path),
        "parse_out": str(parse_out),
    }
    (out_dir / f"{run_id}__manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
