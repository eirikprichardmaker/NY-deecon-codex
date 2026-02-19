from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from src.common.utils import get_first_env


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_py(script: Path, args: list[str], extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, str(script)] + args
    print("[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr.rstrip())
        raise RuntimeError(f"Exit={r.returncode} ved {script}")
    if r.stderr:
        print(r.stderr.rstrip())


def _safe_url(url: str, params: dict) -> str:
    if not params:
        return url
    safe = dict(params)
    if "authKey" in safe:
        safe["authKey"] = "***"
    req = requests.Request("GET", url, params=safe).prepare()
    return req.url or url


def _get_with_retries(url: str, params: dict, timeout: int = 30, tries: int = 4, backoff_s: float = 1.5, verify: bool | None = None) -> requests.Response:
    last_err: Exception | None = None
    for i in range(tries):
        try:
            return requests.get(url, params=params, timeout=timeout, verify=verify if verify is not None else True)
        except requests.RequestException as e:
            last_err = e
            if i < tries - 1:
                time.sleep(backoff_s * (2 ** i))
    safe_url = _safe_url(url, params)
    err_type = type(last_err).__name__ if last_err else "Error"
    raise RuntimeError(f"Request failed after {tries} tries: {safe_url} ({err_type})") from None

def _verify_tls() -> bool:
    return os.environ.get("BORSDATA_TLS_INSECURE", "").strip() != "1"



def pick_instruments(authkey: str, n: int = 3) -> list[int]:
    base = "https://apiservice.borsdata.se/v1"
    r = _get_with_retries(f"{base}/instruments", params={"authKey": authkey}, timeout=30, verify=_verify_tls())
    r.raise_for_status()
    j = r.json()

    lst = j if isinstance(j, list) else (j.get("instruments") or j.get("data") or j.get("items") or [])
    ins_ids = []
    for item in lst[: max(n, 1)]:
        if isinstance(item, dict):
            iid = item.get("insId") or item.get("ins_id") or item.get("id")
            if iid is not None:
                ins_ids.append(int(iid))
    return ins_ids


def probe_reports(authkey: str, ins_ids: list[int]) -> dict:
    base = "https://apiservice.borsdata.se/v1"
    out = {"probe_ins_ids": ins_ids, "reports": []}
    for iid in ins_ids:
        r = _get_with_retries(
            f"{base}/instruments/{iid}/reports",
            params={"authKey": authkey, "maxCount": 5},
            timeout=30,
            verify=_verify_tls(),
        )
        entry = {"ins_id": iid, "status": r.status_code, "reason": r.reason, "len": len(r.content)}
        try:
            j = r.json()
            entry["json_type"] = type(j).__name__
            if isinstance(j, list):
                entry["list_len"] = len(j)
                if j and isinstance(j[0], dict):
                    entry["first_keys"] = list(j[0].keys())[:30]
                    entry["first_date_fields"] = {k: j[0].get(k) for k in j[0].keys() if "date" in k.lower() or "end" in k.lower() or "period" in k.lower()}
            elif isinstance(j, dict):
                entry["dict_keys"] = list(j.keys())[:30]
                # prøv vanlige containere
                for cand in ("reports", "data", "items", "result"):
                    if cand in j and isinstance(j[cand], list):
                        entry["container_key"] = cand
                        entry["list_len"] = len(j[cand])
                        if j[cand] and isinstance(j[cand][0], dict):
                            entry["first_keys"] = list(j[cand][0].keys())[:30]
                        break
        except Exception:
            entry["json_type"] = "non-json"
        out["reports"].append(entry)
    return out


def ensure_period_end(df: pd.DataFrame) -> pd.DataFrame:
    if "period_end" not in df.columns:
        if "date" in df.columns:
            df = df.copy()
            df["period_end"] = df["date"]
        else:
            raise RuntimeError("fundamentals mangler både 'period_end' og 'date'")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True)
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / args.asof
    raw_f = raw_dir / "fundamentals_history.parquet"
    raw_meta = raw_dir / "ingest_meta.json"

    ingest_py = root / "src" / "ingest_fundamentals_history.py"
    build_panel_py = root / "src" / "build_panel.py"

    golden_dir = root / "data" / "golden"
    golden_hist = golden_dir / "history"
    golden_f = golden_dir / "fundamentals_history.parquet"
    golden_f_snap = golden_hist / f"fundamentals_history_{args.asof}.parquet"
    golden_prices = golden_dir / "prices_panel.parquet"

    raw_prices_a = root / "data" / "raw" / "prices" / "prices_panel.parquet"
    raw_prices_b = root / "data" / "raw" / "_synced" / args.asof / "prices_panel.parquet"

    processed_panel = root / "data" / "processed" / "panel.parquet"

    load_dotenv(override=True)
    authkey = get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY", "BORSDATA_KEY"])
    if not authkey:
        raise RuntimeError("BORSDATA_AUTHKEY ikke lastet. Sjekk .env (repo-root).")

    # preflight
    ins_ids = pick_instruments(authkey, n=3)
    if not ins_ids:
        raise RuntimeError("Fikk ingen insId fra /instruments (uventet når preflight var OK).")

    # (re)run ingest alltid her, slik at RAW faktisk matcher gjeldende key/kode
    run_py(ingest_py, ["--asof", args.asof], extra_env={"DEBUG_HTTP": "1"})

    if not raw_f.exists():
        raise RuntimeError(f"Ingest skrev ikke {raw_f}")

    df = pd.read_parquet(raw_f)
    meta = json.load(open(raw_meta, "r", encoding="utf-8")) if raw_meta.exists() else {}

    print("[CHECK] ingest_meta rows=", meta.get("rows"), "api_calls=", meta.get("api_calls"), "tickers_used=", meta.get("tickers_used"))
    print("[CHECK] parquet rows=", len(df))

    if len(df) == 0:
        probe = probe_reports(authkey, ins_ids)
        write_json(raw_dir / "api_probe.json", probe)
        raise RuntimeError(f"RAW fundamentals fortsatt 0 rader. Se {raw_dir / 'api_probe.json'} og ingest_meta.json")

    # freeze golden fundamentals (med period_end)
    golden_hist.mkdir(parents=True, exist_ok=True)
    df2 = ensure_period_end(df)
    df2.to_parquet(golden_f, index=False)
    df2.to_parquet(golden_f_snap, index=False)
    write_json(raw_dir / "freeze_golden_fundamentals_history_meta.json", {"rows": int(len(df2)), "cols": df2.columns.tolist()})

    # ensure golden prices
    if not golden_prices.exists():
        src = raw_prices_a if raw_prices_a.exists() else raw_prices_b
        if not src.exists():
            raise RuntimeError("Fant ingen raw prices_panel.parquet (data/raw/prices eller data/raw/_synced/...).")
        p = pd.read_parquet(src)
        if "yahoo_ticker" not in p.columns and "ticker" in p.columns:
            p = p.rename(columns={"ticker": "yahoo_ticker"})
        need = ["yahoo_ticker", "date", "adj_close"]
        miss = [c for c in need if c not in p.columns]
        if miss:
            raise RuntimeError(f"prices_panel mangler {miss}. cols={p.columns.tolist()}")
        keep = ["yahoo_ticker", "date", "adj_close"] + (["volume"] if "volume" in p.columns else [])
        p[keep].to_parquet(golden_prices, index=False)

    # build panel
    run_py(build_panel_py, ["--asof", args.asof, "--prices", str(golden_prices)])

    if not processed_panel.exists():
        raise RuntimeError(f"build_panel skrev ikke {processed_panel}")

    print("[OK] panel.parquet rows=", len(pd.read_parquet(processed_panel)))


if __name__ == "__main__":
    main()


