from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asof", required=True, help="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    asof = args.asof

    src_path = data_dir / "raw" / asof / "fundamentals_history.parquet"
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    golden_path = data_dir / "golden" / "fundamentals_history.parquet"
    hist_dir = data_dir / "golden" / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    version_path = hist_dir / f"fundamentals_history_{asof}.parquet"

    # kopier ved å lese/skriv (enkelt og plattformuavhengig)
    df = pd.read_parquet(src_path)
    df.to_parquet(golden_path, index=False)
    df.to_parquet(version_path, index=False)

    # manifest
    manifest = hist_dir / "manifest_fundamentals_history.csv"
    row = {
        "asof": asof,
        "raw_path": str(src_path).replace("\\", "/"),
        "golden_path": str(golden_path).replace("\\", "/"),
        "version_path": str(version_path).replace("\\", "/"),
        "rows": len(df),
        "sha256": _sha256(version_path),
    }
    if manifest.exists():
        m = pd.read_csv(manifest)
        m = pd.concat([m, pd.DataFrame([row])], ignore_index=True)
    else:
        m = pd.DataFrame([row])
    m.to_csv(manifest, index=False)

    print(f"OK: golden  -> {golden_path}")
    print(f"OK: version -> {version_path}")
    print(f"OK: manifest-> {manifest}")


if __name__ == "__main__":
    main()
