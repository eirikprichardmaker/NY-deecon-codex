from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def rows_parquet(p: Path) -> int | None:
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        return int(len(df))
    except Exception:
        return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, obj: dict) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_and_write_golden_fundamentals(
    raw_path: Path, asof: str, out_current: Path, out_snapshot: Path
) -> dict:
    df = pd.read_parquet(raw_path)
    if len(df) == 0:
        raise RuntimeError(f"RAW fundamentals er tom: {raw_path}")

    # build_panel forventer period_end; ingest gir 'date'
    if "period_end" not in df.columns:
        if "date" in df.columns:
            df["period_end"] = df["date"]
        else:
            raise RuntimeError(f"Fant verken 'period_end' eller 'date' i {raw_path}")

    # fallback for ticker-felt
    if "yahoo_ticker" not in df.columns and "ticker" in df.columns:
        df["yahoo_ticker"] = df["ticker"]

    ensure_dir(out_current.parent)
    ensure_dir(out_snapshot.parent)
    df.to_parquet(out_current, index=False)
    df.to_parquet(out_snapshot, index=False)

    return {
        "asof": asof,
        "rows": int(len(df)),
        "cols": df.columns.tolist(),
        "path_current": str(out_current),
        "path_snapshot": str(out_snapshot),
    }


def normalize_and_write_golden_prices(in_path: Path, out_path: Path) -> dict:
    df = pd.read_parquet(in_path)

    # raw har ticker, golden forventer yahoo_ticker
    if "yahoo_ticker" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "yahoo_ticker"})

    required = ["yahoo_ticker", "date", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"prices_panel mangler {missing}. Fant cols={df.columns.tolist()} i {in_path}"
        )

    keep = ["yahoo_ticker", "date", "adj_close"] + (["volume"] if "volume" in df.columns else [])
    df = df[keep].copy()

    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)

    return {"rows": int(len(df)), "cols": df.columns.tolist(), "src": str(in_path), "dst": str(out_path)}


def run_py(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script)] + args
    print(f"[RUN] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout.rstrip())
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr.rstrip())
        raise RuntimeError(f"Feil ved kjøring: {script} (exit={r.returncode})")
    if r.stderr:
        print(r.stderr.rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="YYYY-MM-DD")
    ap.add_argument("--horizon-months", type=int, default=12)
    ap.add_argument("--skip-labels", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    raw_dir = root / "data" / "raw" / args.asof
    raw_f = raw_dir / "fundamentals_history.parquet"
    raw_meta = raw_dir / "ingest_meta.json"
    raw_missing_instr = raw_dir / "missing_instruments.csv"

    golden_dir = root / "data" / "golden"
    golden_hist = golden_dir / "history"
    golden_f = golden_dir / "fundamentals_history.parquet"
    golden_f_snap = golden_hist / f"fundamentals_history_{args.asof}.parquet"

    golden_prices = golden_dir / "prices_panel.parquet"
    raw_prices_a = root / "data" / "raw" / "prices" / "prices_panel.parquet"
    raw_prices_b = root / "data" / "raw" / "_synced" / args.asof / "prices_panel.parquet"

    processed_dir = root / "data" / "processed"
    panel_out = processed_dir / "panel.parquet"
    panel_labeled_out = processed_dir / "panel_labeled.parquet"

    build_panel_py = root / "src" / "build_panel.py"
    labels_py = root / "src" / "compute_labels.py"

    print("\n=== STATUS (before) ===")
    print("RAW fundamentals:", raw_f, "rows=", rows_parquet(raw_f))
    print("RAW meta:", raw_meta, "exists=", raw_meta.exists())
    print("RAW missing instruments:", raw_missing_instr, "exists=", raw_missing_instr.exists())
    print("GOLDEN fundamentals:", golden_f, "rows=", rows_parquet(golden_f))
    print("GOLDEN prices:", golden_prices, "rows=", rows_parquet(golden_prices))
    print("PROCESSED panel:", panel_out, "rows=", rows_parquet(panel_out))
    print("PROCESSED labeled:", panel_labeled_out, "rows=", rows_parquet(panel_labeled_out))

    if not raw_f.exists():
        raise RuntimeError(f"Mangler RAW fundamentals: {raw_f}. Kjør ingest først.")

    print("\n=== ACTION: freeze golden fundamentals (with period_end) ===")
    ensure_dir(golden_hist)
    fmeta = normalize_and_write_golden_fundamentals(
        raw_path=raw_f,
        asof=args.asof,
        out_current=golden_f,
        out_snapshot=golden_f_snap,
    )
    write_json(raw_dir / "freeze_golden_fundamentals_history_meta.json", fmeta)
    print("[OK] golden fundamentals updated:", fmeta["rows"], "rows")

    if not golden_prices.exists() or rows_parquet(golden_prices) in (None, 0):
        print("\n=== ACTION: build golden prices_panel.parquet ===")
        src = raw_prices_a if raw_prices_a.exists() else raw_prices_b
        if not src.exists():
            raise RuntimeError("Fant ingen raw prices_panel.parquet (data/raw/prices eller data/raw/_synced/...).")
        pmeta = normalize_and_write_golden_prices(src, golden_prices)
        write_json(raw_dir / "normalize_prices_meta.json", pmeta)
        print("[OK] golden prices updated:", pmeta["rows"], "rows")
    else:
        print("\n=== SKIP: golden prices already exists ===")

    print("\n=== ACTION: build_panel ===")
    if not build_panel_py.exists():
        raise RuntimeError(f"Fant ikke {build_panel_py}")
    run_py(build_panel_py, ["--asof", args.asof, "--prices", str(golden_prices)])

    if not panel_out.exists():
        raise RuntimeError(f"build_panel fullførte uten å skrive {panel_out} (sjekk output over).")
    print("[OK] panel exists:", panel_out, "rows=", rows_parquet(panel_out))

    if not args.skip_labels and labels_py.exists():
        print("\n=== ACTION: compute_labels ===")
        run_py(labels_py, ["--horizon-months", str(args.horizon_months)])
        if panel_labeled_out.exists():
            print("[OK] panel_labeled exists:", panel_labeled_out, "rows=", rows_parquet(panel_labeled_out))
        else:
            print("[WARN] compute_labels kjørte, men fant ikke", panel_labeled_out)
    else:
        print("\n=== SKIP: labels ===")

    print("\n=== STATUS (after) ===")
    print("GOLDEN fundamentals rows=", rows_parquet(golden_f))
    print("GOLDEN prices rows=", rows_parquet(golden_prices))
    print("PROCESSED panel rows=", rows_parquet(panel_out))
    print("PROCESSED labeled rows=", rows_parquet(panel_labeled_out))


if __name__ == "__main__":
    main()
