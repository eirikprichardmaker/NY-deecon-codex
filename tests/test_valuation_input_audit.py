from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from src import valuation


class _Log:
    def info(self, _msg: str) -> None:
        return


def test_valuation_writes_fundamental_input_audit(tmp_path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "runs" / "20260302_000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    master = pd.DataFrame(
        {
            "yahoo_ticker": ["AAA.ST"],
            "ticker": ["AAA"],
            "company": ["Acme"],
            "fcf_millions": [100.0],
            "net_debt_current": [10.0],
            "wacc_used": [0.09],
            "coe_used": [0.09],
        }
    )
    master.to_parquet(processed / "master_cost.parquet", index=False)

    ctx = SimpleNamespace(
        cfg={
            "paths": {
                "processed_dir": "data/processed",
                "runs_dir": "runs",
            }
        },
        project_root=tmp_path,
        run_id="20260302_000000",
        run_dir=run_dir,
        asof="2026-03-02",
    )

    rc = valuation.run(ctx, _Log())
    assert rc == 0

    audit_path = run_dir / "valuation_input_audit.json"
    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["fundamental_only_guard_passed"] is True
    assert "fcf_millions" in payload["fundamental_inputs_selected"]


def test_valuation_prefers_quarterly_reports_r12_when_enabled(tmp_path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "runs" / "20260302_000001"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_r12_dir = tmp_path / "data" / "raw" / "2026-03-02" / "reports_r12" / "market=SE"
    raw_r12_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    master = pd.DataFrame(
        {
            "yahoo_ticker": ["AAA.ST"],
            "ticker": ["AAA"],
            "company": ["Acme"],
            "fcf_millions": [100.0],
            "net_debt_current": [10.0],
            "wacc_used": [0.09],
            "coe_used": [0.09],
        }
    )
    master.to_parquet(processed / "master_cost.parquet", index=False)

    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "yahoo_ticker": ["AAA.ST"],
            "country": ["Sweden"],
            "company": ["Acme"],
            "sector": [""],
            "borsdata_ticker": [""],
            "ins_id": [1],
        }
    ).to_csv(cfg_dir / "tickers_with_insid_clean.csv", index=False)

    pd.DataFrame(
        {
            "year": [2025],
            "period": [4],
            "free_Cash_Flow": [200.0],
        }
    ).to_parquet(raw_r12_dir / "ins_id=1.parquet", index=False)

    ctx = SimpleNamespace(
        cfg={
            "paths": {
                "processed_dir": "data/processed",
                "runs_dir": "runs",
                "raw_dir": "data/raw",
            },
            "valuation": {
                "prefer_quarterly_reports": True,
                "require_quarterly_reports": True,
                "max_quarterly_report_age_days": 400,
            },
        },
        project_root=tmp_path,
        run_id="20260302_000001",
        run_dir=run_dir,
        asof="2026-03-02",
    )

    rc = valuation.run(ctx, _Log())
    assert rc == 0

    val = pd.read_csv(run_dir / "valuation.csv")
    assert len(val) == 1
    assert str(val.loc[0, "fcf_source"]) == "reports_r12"
    assert float(val.loc[0, "fcf_used_millions"]) == 200.0
    assert pd.isna(val.loc[0, "reason"]) or str(val.loc[0, "reason"]) == ""


def test_valuation_requires_quarterly_reports_when_configured(tmp_path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "runs" / "20260302_000002"
    run_dir.mkdir(parents=True, exist_ok=True)

    master = pd.DataFrame(
        {
            "yahoo_ticker": ["AAA.ST"],
            "ticker": ["AAA"],
            "company": ["Acme"],
            "fcf_millions": [100.0],
            "net_debt_current": [10.0],
            "wacc_used": [0.09],
            "coe_used": [0.09],
        }
    )
    master.to_parquet(processed / "master_cost.parquet", index=False)

    ctx = SimpleNamespace(
        cfg={
            "paths": {
                "processed_dir": "data/processed",
                "runs_dir": "runs",
                "raw_dir": "data/raw",
            },
            "valuation": {
                "prefer_quarterly_reports": True,
                "require_quarterly_reports": True,
                "max_quarterly_report_age_days": 400,
            },
        },
        project_root=tmp_path,
        run_id="20260302_000002",
        run_dir=run_dir,
        asof="2026-03-02",
    )

    rc = valuation.run(ctx, _Log())
    assert rc == 0

    val = pd.read_csv(run_dir / "valuation.csv")
    assert len(val) == 1
    assert str(val.loc[0, "reason"]) == "missing_quarterly_fcf"
    assert pd.isna(val.loc[0, "intrinsic_equity"])
