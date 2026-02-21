from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pandas as pd

import src.freeze_golden_fundamentals_history as mod


def _cfg(tmp_path):
    return mod.FreezeConfig(
        asof="2026-02-16",
        data_dir=tmp_path / "data",
        markets=("NO",),
        include_delisted=True,
        skip_existing=True,
        refresh_stale_days=0,
        refetch_invalid_cache=False,
        force=False,
        datasets=("prices",),
        timeout=10,
        max_attempts=3,
    )


def test_target_path_for_partitioned_layout(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "2026-02-16"
    path = mod._target_path_for(raw_dir, dataset="prices", market="NO", ins_id=123)
    assert path == raw_dir / "prices" / "market=NO" / "ins_id=123.parquet"

    path_with_kpi = mod._target_path_for(
        raw_dir,
        dataset="kpis",
        market="SE",
        ins_id=456,
        extra_dims={"kpi_id": 7},
    )
    assert path_with_kpi == raw_dir / "kpis" / "kpi_id=7" / "market=SE" / "ins_id=456.parquet"


def test_bootstrap_manifest_registers_existing_flat_parquet(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "2026-02-16"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"v": 1}]).to_parquet(raw_dir / "prices__NO__123.parquet", index=False)

    manifest = mod._bootstrap_manifest_if_missing(raw_dir)
    assert (raw_dir / "manifest.parquet").exists()
    assert set(mod.MANIFEST_COLUMNS).issubset(set(manifest.columns))

    rows = manifest.set_index("file_path")
    assert rows.loc["prices__NO__123.parquet", "status"] == "cached"
    assert rows.loc["prices__NO__123.parquet", "source"] == "cached"
    assert int(rows.loc["prices__NO__123.parquet", "rows"]) == 1

def test_bootstrap_manifest_prefers_partitioned_over_flat_when_both_exist(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "2026-02-16"
    (raw_dir / "prices" / "market=NO").mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"v": 1}]).to_parquet(raw_dir / "prices__NO__123.parquet", index=False)
    pd.DataFrame([{"v": 2}]).to_parquet(raw_dir / "prices" / "market=NO" / "ins_id=123.parquet", index=False)

    manifest = mod._bootstrap_manifest_if_missing(raw_dir)
    selected = manifest[
        (manifest["dataset"] == "prices")
        & (manifest["market"] == "NO")
        & (manifest["ins_id"] == 123)
    ]
    assert len(selected) == 1
    assert selected.iloc[0]["file_path"] == "prices/market=NO/ins_id=123.parquet"
    assert selected.iloc[0]["status"] == "cached"


def test_run_task_skips_existing_without_fetch(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    raw_dir = cfg.data_dir / "raw" / cfg.asof
    file_path = raw_dir / "prices" / "market=NO" / "ins_id=101.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"dataset": "prices", "v": 1}]).to_parquet(file_path, index=False)

    task = mod.FreezeTask(
        dataset="prices",
        market="NO",
        ins_id=101,
        file_path=file_path,
        file_path_rel="prices/market=NO/ins_id=101.parquet",
    )
    called = {"n": 0}

    def fake_fetch_dataset_payload(_client, _dataset, _asof, _ins_id):
        called["n"] += 1
        return {"items": [{"v": 1}]}, 1

    monkeypatch.setattr(mod, "_fetch_dataset_payload", fake_fetch_dataset_payload)
    row = mod._run_task(task, cfg, client=object(), now_utc="2026-02-16T00:00:00+00:00")

    assert called["n"] == 0
    assert row["status"] == "cached"
    assert row["source"] == "cached"
    assert row["rows"] == 1


def test_invalid_cache_empty_parquet_marked_cache_invalid(tmp_path):
    raw_dir = tmp_path / "data" / "raw" / "2026-02-16"
    bad_path = raw_dir / "prices" / "market=NO" / "ins_id=777.parquet"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"v": []}).to_parquet(bad_path, index=False)

    manifest = mod._bootstrap_manifest_if_missing(raw_dir)
    rows = manifest.set_index("file_path")
    row = rows.loc["prices/market=NO/ins_id=777.parquet"]
    assert row["status"] == "missing"
    assert row["source"] == "cache_invalid"
    assert "empty parquet" in str(row["last_error"]).lower()


def test_run_task_does_not_refetch_invalid_cache_by_default(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    raw_dir = cfg.data_dir / "raw" / cfg.asof
    bad_path = raw_dir / "prices" / "market=NO" / "ins_id=202.parquet"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"v": []}).to_parquet(bad_path, index=False)

    task = mod.FreezeTask(
        dataset="prices",
        market="NO",
        ins_id=202,
        file_path=bad_path,
        file_path_rel="prices/market=NO/ins_id=202.parquet",
    )
    called = {"n": 0}

    def fake_fetch_dataset_payload(_client, _dataset, _asof, _ins_id):
        called["n"] += 1
        return {"items": [{"v": 1}]}, 1

    monkeypatch.setattr(mod, "_fetch_dataset_payload", fake_fetch_dataset_payload)
    row = mod._run_task(task, cfg, client=object(), now_utc="2026-02-16T00:00:00+00:00")

    assert called["n"] == 0
    assert row["status"] == "missing"
    assert row["source"] == "cache_invalid"
    assert "empty parquet" in str(row["last_error"]).lower()


def test_retry_after_seconds_parses_numeric_and_http_date():
    assert mod._retry_after_seconds("3", default=1.0) >= 3.0

    dt = datetime.now(timezone.utc) + timedelta(seconds=2)
    retry_after = format_datetime(dt)
    parsed = mod._retry_after_seconds(retry_after, default=1.0)
    assert parsed >= 0.1


def test_write_coverage_md_contains_expected_and_missing_sections(tmp_path):
    coverage = pd.DataFrame(
        [
            {
                "dataset": "prices",
                "market": "NO",
                "expected": 10,
                "cached": 7,
                "ok": 2,
                "failed": 0,
                "missing": 1,
                "completed": 9,
                "coverage_ratio": 0.9,
                "rows_total": 100,
            }
        ]
    )
    out = tmp_path / "runs" / "2026-02-16" / "freeze_coverage.md"
    mod._write_coverage_md(
        out,
        asof="2026-02-16",
        coverage=coverage,
        markets=("NO", "SE"),
        include_delisted=False,
    )
    text = out.read_text(encoding="utf-8")
    assert "## Expected calculation" in text
    assert "## Missing semantics" in text
    assert "expected = antall instrumenter i scope" in text
    assert "coverage_ratio = completed / expected" in text
