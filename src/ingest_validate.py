from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.common.config import resolve_paths
from src.common.errors import ProviderError, SchemaError


DEFAULT_CANDIDATES = {
    "master_input": [
        "master_with_price.csv",
        "master_latest.csv",
        "master_with_price.parquet",
        "master_latest.parquet",
        "master.parquet",
    ],
    "fundamentals": [
        "fundamentals_latest.csv",
        "fundamentals_clean.csv",
        "fundamentals_mapped.csv",
        "fundamentals_latest.parquet",
        "fundamentals.parquet",
    ],
    "prices": [
        "prices_panel.parquet",
        "prices_latest.csv",
        "prices_latest_one_row.csv",
        "prices.parquet",
    ],
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_newest(root: Path, names: list[str]) -> Path | None:
    hits: list[Path] = []
    for n in names:
        hits.extend(root.rglob(n))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    raw_asof_dir = paths["raw_dir"] / ctx.asof
    raw_asof_dir.mkdir(parents=True, exist_ok=True)

    sources = ctx.cfg.get("sources", {}) or {}
    inputs_cfg = (sources.get("inputs", {}) or {})

    copied: dict[str, dict] = {}

    for key, defaults in DEFAULT_CANDIDATES.items():
        explicit = inputs_cfg.get(key)

        if explicit:
            src = (ctx.project_root / explicit).resolve()
            if not src.exists():
                raise ProviderError(f"Input '{key}' not found at: {src}")
        else:
            src = _find_newest(ctx.project_root, defaults)

            # Fallback: golden fundamentals if no explicit or candidate found
            if src is None and key == "fundamentals":
                golden_dir = Path(ctx.cfg.get("golden_dir", "data/golden"))
                if not golden_dir.is_absolute():
                    golden_dir = ctx.project_root / golden_dir
                golden = golden_dir / "fundamentals.parquet"
                if golden.exists():
                    src = golden
                    log.info(f"INGEST: fundamentals fallback -> {golden}")
                else:
                    log.info("INGEST: no candidate found for fundamentals and no golden fallback found.")
                    continue

            if src is None:
                log.info(f"INGEST: no candidate found for {key} (ok if not needed yet).")
                continue

        if src.stat().st_size == 0:
            raise SchemaError(f"Input '{key}' is empty: {src}")

        dst = raw_asof_dir / f"{key}{src.suffix}"
        shutil.copy2(src, dst)

        copied[key] = {
            "src": str(src),
            "dst": str(dst),
            "sha256": _sha256_file(dst),
            "bytes": dst.stat().st_size,
        }
        log.info(f"INGEST: {key} -> {dst.name}")

    if not copied:
        raise ProviderError(
            "No inputs found to ingest. Put files in repo root OR set sources.inputs.* OR create golden fundamentals."
        )

    # Update manifest in run_dir
    manifest_path = ctx.run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"run_id": ctx.run_id, "asof": ctx.asof}

    manifest["ingest"] = {"raw_dir": str(raw_asof_dir), "files": copied}
    _write_json(manifest_path, manifest)

    log.info("INGEST: OK")
    return 0
