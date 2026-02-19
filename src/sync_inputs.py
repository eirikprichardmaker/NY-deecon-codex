# src/sync_inputs.py
from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import yaml


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_sources(repo: Path) -> dict:
    p = repo / "configs" / "sources.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _as_path(val: str | None) -> Path | None:
    if not val:
        return None
    return Path(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat())
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sources = load_sources(repo)
    sync_enabled = bool((sources.get("sync") or {}).get("enabled", True))
    if not sync_enabled:
        print("Sync disabled in configs/sources.yaml (sync.enabled=false)")
        return

    dst_fund_mapped = repo / "data/raw/borsdata/fundamentals_mapped.csv"
    dst_fund_clean  = repo / "data/raw/borsdata/fundamentals_clean.csv"
    dst_fund_latest = repo / "data/raw/borsdata/fundamentals_export_latest.csv"

    dst_prices_panel = repo / "data/raw/prices/prices_panel.parquet"
    dst_prices_latest = repo / "data/raw/prices/prices_latest.csv"

    ensure_dir(dst_fund_mapped.parent)
    ensure_dir(dst_prices_panel.parent)

    b = sources.get("borsdata") or {}
    p = sources.get("prices") or {}

    cand_f = [
        _as_path(b.get("fundamentals_input")),
        _as_path(b.get("fundamentals_clean_input")),
        _as_path(b.get("fundamentals_latest_input")),
    ]
    src_f = next((x for x in cand_f if x and x.exists()), None)
    if src_f:
        name = src_f.name.lower()
        if "mapped" in name:
            shutil.copy2(src_f, dst_fund_mapped)
            print(f"Synced: {src_f} -> {dst_fund_mapped}")
        elif "clean" in name:
            shutil.copy2(src_f, dst_fund_clean)
            print(f"Synced: {src_f} -> {dst_fund_clean}")
        else:
            shutil.copy2(src_f, dst_fund_latest)
            print(f"Synced: {src_f} -> {dst_fund_latest}")
    else:
        print("No upstream fundamentals found (check configs/sources.yaml).")

    src_panel = _as_path(p.get("prices_panel_input"))
    if src_panel and src_panel.exists():
        shutil.copy2(src_panel, dst_prices_panel)
        print(f"Synced: {src_panel} -> {dst_prices_panel}")
    else:
        src_latest = _as_path(p.get("prices_latest_input"))
        if src_latest and src_latest.exists():
            shutil.copy2(src_latest, dst_prices_latest)
            print(f"Synced: {src_latest} -> {dst_prices_latest}")
        else:
            print("No upstream prices found (check configs/sources.yaml).")

    stamp_dir = repo / "data/raw/_synced" / args.as_of
    ensure_dir(stamp_dir)
    for f in [dst_fund_mapped, dst_fund_clean, dst_fund_latest, dst_prices_panel, dst_prices_latest]:
        if f.exists():
            shutil.copy2(f, stamp_dir / f.name)

    print("OK: sync completed")


if __name__ == "__main__":
    main()
