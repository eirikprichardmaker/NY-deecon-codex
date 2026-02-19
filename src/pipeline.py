# src/pipeline.py
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml


REQUIRED_MASTER_COLS = [
    "yahoo_ticker", "Company", "Info - Country",
    "Market Cap - Current", "OCF - Millions", "Capex - Millions", "FCF - Millions",
    "Net Debt - Current", "N.Debt/Ebitda - Current", "ROIC - Current", "EV/EBIT - Current",
    "P/E - Current", "P/S - Current",
    "price_date", "price", "ma200", "mad", "above_ma200",
    "fundamental_ok", "technical_ok", "decision",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pick_first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def read_csv_auto(path: Path) -> pd.DataFrame:
    best = None
    best_cols = -1
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > best_cols:
                best = df
                best_cols = df.shape[1]
        except Exception:
            continue
    if best is None:
        raise ValueError(f"Could not read CSV: {path}")
    return best


def load_thresholds(repo: Path) -> dict:
    p = repo / "configs" / "thresholds.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run bootstrap_repo.py first.")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def load_sources(repo: Path) -> dict:
    p = repo / "configs" / "sources.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _as_path(val: Optional[str]) -> Optional[Path]:
    if not val:
        return None
    # keep Windows drive paths intact; allow relative paths
    return Path(val)


def sync_inputs(repo: Path, sources: dict, as_of: str) -> list[str]:
    """
    Copies upstream inputs defined in configs/sources.yaml into data/raw/... with canonical filenames,
    so pipeline remains reproducible and manifest hashing stays correct.
    """
    notes: list[str] = []
    sync_enabled = bool((sources.get("sync") or {}).get("enabled", True))
    if not sync_enabled:
        return notes

    # Canonical destinations (what the pipeline reads)
    dst_fund_mapped = repo / "data/raw/borsdata/fundamentals_mapped.csv"
    dst_fund_clean  = repo / "data/raw/borsdata/fundamentals_clean.csv"
    dst_fund_latest = repo / "data/raw/borsdata/fundamentals_export_latest.csv"

    dst_prices_panel = repo / "data/raw/prices/prices_panel.parquet"
    dst_prices_latest = repo / "data/raw/prices/prices_latest.csv"

    ensure_dir(dst_fund_mapped.parent)
    ensure_dir(dst_prices_panel.parent)

    b = sources.get("borsdata") or {}
    p = sources.get("prices") or {}

    # fundamentals: prefer mapped, then clean, then latest
    cand_f = [
        _as_path(b.get("fundamentals_input")),
        _as_path(b.get("fundamentals_clean_input")),
        _as_path(b.get("fundamentals_latest_input")),
    ]
    src_f = next((x for x in cand_f if x and x.exists()), None)
    if src_f:
        if src_f.suffix.lower() == ".csv":
            # Decide canonical destination based on filename hint
            name = src_f.name.lower()
            if "mapped" in name:
                shutil.copy2(src_f, dst_fund_mapped)
                notes.append(f"synced fundamentals -> {dst_fund_mapped}")
            elif "clean" in name:
                shutil.copy2(src_f, dst_fund_clean)
                notes.append(f"synced fundamentals -> {dst_fund_clean}")
            else:
                shutil.copy2(src_f, dst_fund_latest)
                notes.append(f"synced fundamentals -> {dst_fund_latest}")
        else:
            # if someone points to parquet, keep it in borsdata as mapped.parquet (not used by default reader)
            dst = repo / "data/raw/borsdata/fundamentals_mapped.parquet"
            shutil.copy2(src_f, dst)
            notes.append(f"synced fundamentals -> {dst}")

    # prices panel (preferred)
    src_panel = _as_path(p.get("prices_panel_input"))
    if src_panel and src_panel.exists():
        shutil.copy2(src_panel, dst_prices_panel)
        notes.append(f"synced prices panel -> {dst_prices_panel}")
    else:
        # fallback to latest CSV
        src_latest = _as_path(p.get("prices_latest_input"))
        if src_latest and src_latest.exists():
            shutil.copy2(src_latest, dst_prices_latest)
            notes.append(f"synced prices latest -> {dst_prices_latest}")

    # Optional: keep dated copies for audit (still ignored by git)
    stamp_dir = repo / "data/raw/_synced" / as_of
    ensure_dir(stamp_dir)
    for f in [dst_fund_mapped, dst_fund_clean, dst_fund_latest, dst_prices_panel, dst_prices_latest]:
        if f.exists():
            shutil.copy2(f, stamp_dir / f.name)

    return notes


def build_fundamentals(repo: Path) -> pd.DataFrame:
    candidates = [
        repo / "data/raw/borsdata/fundamentals_mapped.csv",
        repo / "data/raw/borsdata/fundamentals_clean.csv",
        repo / "data/raw/borsdata/fundamentals_export_latest.csv",
    ]
    fpath = pick_first_existing(candidates)
    if fpath is None:
        raise FileNotFoundError("No fundamentals input found in data/raw/borsdata/")

    df = read_csv_auto(fpath).copy()

    # Normalize ticker column name
    if "yahoo_ticker" not in df.columns:
        for alt in ["Yahoo", "yahoo", "ticker", "Ticker", "yahooTicker"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "yahoo_ticker"})
                break
    if "yahoo_ticker" not in df.columns:
        raise ValueError(f"Fundamentals file missing yahoo_ticker column: {fpath}")

    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str)
    df = df.dropna(subset=["yahoo_ticker"])
    df = df.sort_index().groupby("yahoo_ticker", as_index=False).tail(1)

    required = [
        "Company", "Info - Country",
        "Market Cap - Current", "OCF - Millions", "Capex - Millions", "FCF - Millions",
        "Net Debt - Current", "N.Debt/Ebitda - Current", "ROIC - Current", "EV/EBIT - Current",
        "P/E - Current", "P/S - Current",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fundamentals missing required columns: {missing} (file={fpath})")

    return df


def build_price_features(repo: Path, ma200_window: int, ma21_window: int) -> pd.DataFrame:
    # Prefer panel parquet if present
    p_panel = repo / "data/raw/prices/prices_panel.parquet"
    p_latest = repo / "data/raw/prices/prices_latest.csv"

    if p_panel.exists():
        df = pd.read_parquet(p_panel).copy()
    elif p_latest.exists():
        df = read_csv_auto(p_latest).copy()
    else:
        raise FileNotFoundError("No prices input found in data/raw/prices/")

    # Normalize index/columns (robust)
    # - handles ticker stored as index
    # - handles casing/whitespace differences in column names
    df = df.reset_index()
    df.columns = [str(c).strip() for c in df.columns]

    def _rename_ci(target: str, candidates: list[str]) -> None:
        if target in df.columns:
            return
        lower_to_orig = {str(c).strip().lower(): c for c in df.columns}
        for cand in candidates:
            key = cand.strip().lower()
            if key in lower_to_orig:
                df.rename(columns={lower_to_orig[key]: target}, inplace=True)
                # keep columns clean after rename
                df.columns = [str(c).strip() for c in df.columns]
                return

    _rename_ci("yahoo_ticker", ["yahoo_ticker", "ticker", "symbol", "ric", "yahoo"])
    _rename_ci("price", ["price", "close", "adj_close", "adj close", "last", "px_last"])
    _rename_ci("date", ["date", "pricedate", "price_date", "timestamp", "datetime", "time"])

    if "yahoo_ticker" not in df.columns or "price" not in df.columns:
        raise ValueError("Prices must have yahoo_ticker + price")

    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str)
    df = df.dropna(subset=["yahoo_ticker", "price"])

    # If no date => cannot compute MA; set technical false
    if "date" not in df.columns:
        out = df.groupby("yahoo_ticker", as_index=False).agg(price=("price", "last"))
        out["price_date"] = pd.NaT
        out["ma200"] = np.nan
        out["ma21"] = np.nan
        out["mad"] = np.nan
        out["above_ma200"] = False
        return out[["yahoo_ticker", "price_date", "price", "ma200", "mad", "above_ma200"]]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["yahoo_ticker", "date"])

    def add_ma(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["ma200"] = g["price"].rolling(ma200_window, min_periods=max(20, ma200_window // 4)).mean()
        g["ma21"] = g["price"].rolling(ma21_window, min_periods=max(5, ma21_window // 3)).mean()
        return g

    df = df.groupby("yahoo_ticker", group_keys=False).apply(add_ma)
    last = df.groupby("yahoo_ticker", as_index=False).tail(1).copy()
    last = last.rename(columns={"date": "price_date"})
    last["above_ma200"] = (last["price"] > last["ma200"]).fillna(False)
    last["mad"] = ((last["ma21"] - last["ma200"]) / last["ma200"]).replace([np.inf, -np.inf], np.nan)
    return last[["yahoo_ticker", "price_date", "price", "ma200", "mad", "above_ma200"]]


def apply_screen(master: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    roic_min = float(thresholds["fundamentals"]["roic_min"])
    ev_ebit_min = float(thresholds["fundamentals"]["ev_ebit_min"])
    nd_ebitda_max = float(thresholds["fundamentals"]["nd_ebitda_max"])

    m = master.copy()

    m["fundamental_ok"] = (
        (m["ROIC - Current"] > roic_min)
        & (m["EV/EBIT - Current"] > ev_ebit_min)
        & (m["N.Debt/Ebitda - Current"].notna())
        & (m["N.Debt/Ebitda - Current"] <= nd_ebitda_max)
    )
    m["technical_ok"] = m["above_ma200"].astype(bool)

    m["decision"] = np.where(
        m["fundamental_ok"] & m["technical_ok"], "CANDIDATE",
        np.where(m["fundamental_ok"] & ~m["technical_ok"], "WAIT", "HOLD")
    )
    return m


def build_shortlist(screen: pd.DataFrame) -> pd.DataFrame:
    cand = screen.loc[screen["decision"] == "CANDIDATE"].copy()
    if cand.empty:
        return pd.DataFrame(columns=[
            "yahoo_ticker","Company","ROIC - Current","EV/EBIT - Current","N.Debt/Ebitda - Current",
            "price","ma200","mad","score","rank"
        ])

    def z(x: pd.Series, higher_better: bool) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if sd == 0 or np.isnan(sd):
            s = x * 0.0
        else:
            s = (x - mu) / sd
        return s if higher_better else -s

    cand["score"] = (
        0.5 * z(cand["ROIC - Current"], True)
        + 0.3 * z(cand["EV/EBIT - Current"], False)
        + 0.2 * z(cand["N.Debt/Ebitda - Current"], False)
    )
    cand = cand.sort_values(["score", "ROIC - Current"], ascending=[False, False])
    cand["rank"] = np.arange(1, len(cand) + 1)

    keep = [
        "yahoo_ticker","Company","ROIC - Current","EV/EBIT - Current","N.Debt/Ebitda - Current",
        "price","ma200","mad","score","rank"
    ]
    return cand[keep]


def validate_contract(master: pd.DataFrame) -> None:
    if "yahoo_ticker" not in master.columns:
        raise SystemExit("Missing yahoo_ticker")
    if master["yahoo_ticker"].isna().any():
        raise SystemExit("yahoo_ticker contains nulls")
    if master["yahoo_ticker"].duplicated().any():
        raise SystemExit("yahoo_ticker is not unique")

    missing = [c for c in REQUIRED_MASTER_COLS if c not in master.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    forbidden = [c for c in master.columns if c.endswith("_x") or c.endswith("_y")]
    if forbidden:
        raise SystemExit(f"Forbidden merge leftovers present: {forbidden}")

    if not master["decision"].isin(["CANDIDATE","WAIT","HOLD"]).all():
        raise SystemExit("Invalid decision values")


def write_decision(repo: Path, as_of: str, shortlist: pd.DataFrame, notes: list[str]) -> None:
    ensure_dir(repo / "reports")
    if shortlist.empty:
        payload = {"action":"HOLD_CASH","as_of":as_of,"reason":{"notes":notes}}
        md = f"# Decision ({as_of})\n\n**Action:** HOLD_CASH\n"
    else:
        top = shortlist.iloc[0].to_dict()
        payload = {"action":"BUY","ticker":top["yahoo_ticker"],"as_of":as_of,"reason":{"notes":notes,"rank":int(top["rank"])}}
        md = f"# Decision ({as_of})\n\n**Action:** BUY\n\n- Ticker: **{top['yahoo_ticker']}**\n- Company: {top['Company']}\n- Score: {top['score']:.3f}\n"

    (repo / "reports/decision_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (repo / "reports/decision_latest.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("--as-of", default=date.today().isoformat())
    sub.add_parser("validate")
    sub.add_parser("sync")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    thresholds = load_thresholds(repo)
    sources = load_sources(repo)

    as_of = date.today().isoformat()
    if getattr(args, "as_of", None):
        as_of = args.as_of

    # Sync from upstream (if configured) -> data/raw
    notes_sync = sync_inputs(repo, sources, as_of)

    if args.cmd == "sync":
        print("OK: sync completed")
        if notes_sync:
            for n in notes_sync:
                print(f"- {n}")
        return

    ma200 = int(thresholds["technical"]["ma200_window"])
    ma21 = int(thresholds["technical"]["ma21_window"])

    fundamentals = build_fundamentals(repo)
    prices = build_price_features(repo, ma200, ma21)

    master = fundamentals.merge(prices, on="yahoo_ticker", how="left", validate="one_to_one")
    screen = apply_screen(master, thresholds)

    validate_contract(screen)

    latest = repo / "data/latest"
    ensure_dir(latest)
    screen.to_parquet(latest / "master.parquet", index=False)
    screen.to_parquet(latest / "screen.parquet", index=False)

    shortlist = build_shortlist(screen)
    shortlist.to_csv(latest / "shortlist.csv", index=False)

    notes = []
    notes.extend(notes_sync)
    if screen["ma200"].isna().any():
        notes.append("Missing MA200 for some tickers (prices panel likely incomplete).")
    if shortlist.empty:
        notes.append("No candidates passed fundamental_ok AND technical_ok.")

    write_decision(repo, as_of, shortlist, notes)

    # runs/YYYY-MM-DD
    run_dir = repo / "runs" / as_of
    ensure_dir(run_dir / "outputs")
    for fn in ["master.parquet","screen.parquet","shortlist.csv"]:
        src = latest / fn
        (run_dir / "outputs" / fn).write_bytes(src.read_bytes())

    thresholds_text = (repo / "configs/thresholds.yaml").read_text(encoding="utf-8")
    manifest = {
        "as_of": as_of,
        "thresholds_sha256": sha256_text(thresholds_text),
        "inputs": [],
        "upstream_inputs": [],
        "outputs": [],
    }

    # Hash local raw (reproducible)
    for p in (repo / "data/raw").rglob("*"):
        if p.is_file():
            manifest["inputs"].append({"path": str(p.relative_to(repo)), "sha256": sha256_file(p)})

    # Hash upstream (traceability)
    b = sources.get("borsdata") or {}
    pr = sources.get("prices") or {}
    for k in ["fundamentals_input","fundamentals_clean_input","fundamentals_latest_input"]:
        sp = b.get(k)
        if sp:
            pp = Path(sp)
            if pp.exists():
                manifest["upstream_inputs"].append({"path": str(pp), "sha256": sha256_file(pp)})
    for k in ["prices_panel_input","prices_latest_input"]:
        sp = pr.get(k)
        if sp:
            pp = Path(sp)
            if pp.exists():
                manifest["upstream_inputs"].append({"path": str(pp), "sha256": sha256_file(pp)})

    for p in (run_dir / "outputs").glob("*"):
        manifest["outputs"].append({"path": str(p.relative_to(repo)), "sha256": sha256_file(p)})

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.cmd == "validate":
        print("OK: validated")
        return

    print(f"OK: run completed ({as_of})")


if __name__ == "__main__":
    main()
