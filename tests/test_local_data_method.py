from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.local_data_method as mod


def _mk_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_build_local_inventory_picks_latest_asof(tmp_path):
    data_dir = tmp_path / "data"
    old_dir = data_dir / "raw" / "2026-02-15"
    new_dir = data_dir / "raw" / "2026-02-16"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"date": "2026-02-16", "ticker": "EQNR", "adj_close": 100.0, "volume": 10_000}]
    ).to_parquet(new_dir / "prices.parquet", index=False)
    pd.DataFrame(
        [{"ins_id": 1, "ticker": "EQNR", "market": "NO", "currency": "NOK", "sector": "Energy"}]
    ).to_csv(new_dir / "ticker_map.csv", index=False)

    asof, inv, _ = mod.build_local_inventory(data_dir=data_dir, asof=None)

    assert asof == "2026-02-16"
    assert "daglige_priser_aksjer" in set(inv["freeze_item"])


def test_build_local_inventory_reports_required_missing_fields(tmp_path):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw" / "2026-02-16"
    raw.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"date": "2026-02-16", "ticker": "EQNR", "adj_close": 100.0, "volume": 10_000}]
    ).to_parquet(raw / "prices.parquet", index=False)
    # Missing currency/sector to force partial
    pd.DataFrame([{"ins_id": 1, "ticker": "EQNR", "market": "NO"}]).to_csv(raw / "ticker_map.csv", index=False)

    pd.DataFrame(
        [
            {
                "period_end": "2025-12-31",
                "report_period": "Q4",
                "available_date": "2026-02-15",
                "metric": "revenue",
                "value": 10.0,
            }
        ]
    ).to_parquet(raw / "fundamentals_history.parquet", index=False)

    _, inv, _ = mod.build_local_inventory(data_dir=data_dir, asof="2026-02-16")
    inv = inv.set_index("freeze_item")

    assert inv.loc["daglige_priser_aksjer", "status"] == "ready"
    assert inv.loc["rapportdata_kvartal_ar", "status"] == "ready"
    assert inv.loc["instrument_universliste", "status"] == "partial"
    assert "currency" in inv.loc["instrument_universliste", "missing_fields"]


def test_build_local_inventory_uses_manifest_when_partitioned_sources_exist(tmp_path):
    data_dir = tmp_path / "data"
    raw = data_dir / "raw" / "2026-02-16"
    raw.mkdir(parents=True, exist_ok=True)

    _mk_manifest(
        raw / "manifest.parquet",
        [
            {
                "dataset": "reports_q",
                "status": "ok",
                "rows": 123,
                "file_path": "reports_q/market=NO/ins_id=1.parquet",
                "source": "api",
            },
            {
                "dataset": "prices",
                "status": "ok",
                "rows": 456,
                "file_path": "prices/market=NO/ins_id=1.parquet",
                "source": "api",
            },
        ],
    )

    _, inv, sources = mod.build_local_inventory(data_dir=data_dir, asof="2026-02-16")
    inv = inv.set_index("freeze_item")

    assert inv.loc["rapportdata_kvartal_ar", "status"] == "partial"
    assert int(inv.loc["rapportdata_kvartal_ar", "rows"]) == 123
    assert str(inv.loc["rapportdata_kvartal_ar", "source"]).startswith("manifest:")

    assert "daglige_priser_aksjer" in set(sources["freeze_item"])


def test_write_local_inventory_outputs_files(tmp_path):
    inv = pd.DataFrame(
        [
            {
                "freeze_item": "daglige_priser_aksjer",
                "status": "ready",
                "source": "data/raw/2026-02-16/prices.parquet",
                "rows": 1,
                "required_fields": "date,close_or_adj_close,volume",
                "missing_fields": "",
                "comment": "ok",
            }
        ]
    )
    src = pd.DataFrame(
        [
            {
                "freeze_item": "daglige_priser_aksjer",
                "path_or_pattern": "data/raw/2026-02-16/prices.parquet",
                "read_hint": "pd.read_parquet",
                "note": "test",
            }
        ]
    )

    out = mod.write_local_inventory(asof="2026-02-16", out_dir=tmp_path / "runs" / "2026-02-16", inventory=inv, source_df=src)

    assert out["inventory_csv"].exists()
    assert out["sources_csv"].exists()
    assert out["inventory_md"].exists()
