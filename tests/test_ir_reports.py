from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ir_reports, run_weekly


class _Log:
    def info(self, *_args, **_kwargs):
        return None


@dataclass
class _Ctx:
    asof: str
    project_root: Path
    run_dir: Path
    cfg: dict


def _make_ctx(tmp_path: Path, mapping_path: Path) -> _Ctx:
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "paths": {
            "runs_dir": "runs",
            "data_dir": "data",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
        },
        "ir_reports": {
            "mapping_csv": str(mapping_path),
            "rate_limit_sec": 0,
            "timeout_sec": 1,
        },
    }
    return _Ctx(asof="2026-02-16", project_root=tmp_path, run_dir=run_dir, cfg=cfg)


def test_ir_reports_network_error_fail_soft(tmp_path, monkeypatch):
    mapping = tmp_path / "config" / "ir_sources.csv"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text("ticker,url,source,period,report_date\nEQNR,https://example.com/r1.html,example,Q4,2026-02-15\n", encoding="utf-8")

    ctx = _make_ctx(tmp_path, mapping)

    monkeypatch.setattr(ir_reports, "_can_fetch", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(ir_reports, "_fetch_url_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))

    rc = ir_reports.run(ctx, _Log())
    assert rc == 0

    idx = pd.read_parquet(tmp_path / "data" / "raw" / "ir" / "index.parquet")
    facts = pd.read_parquet(tmp_path / "data" / "golden" / "ir_facts.parquet")

    assert len(idx) == 1
    assert idx.iloc[0]["status"] == "ERROR"
    assert idx.iloc[0]["error_code"] == "NETWORK"

    assert len(facts) == 1
    assert facts.iloc[0]["status"] == "ERROR"
    assert facts.iloc[0]["error_code"] == "NETWORK"


def test_ir_reports_duplicate_report_detected_and_skipped(tmp_path, monkeypatch):
    mapping = tmp_path / "config" / "ir_sources.csv"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text("ticker,url,source,period,report_date\nEQNR,https://example.com/r1.html,example,Q4,2026-02-15\n", encoding="utf-8")

    ctx = _make_ctx(tmp_path, mapping)

    calls = {"n": 0}

    def _fetch(*_args, **_kwargs):
        calls["n"] += 1
        return (b"<html><body>Revenue 123 EBIT 45 EPS 1.2</body></html>", "text/html")

    monkeypatch.setattr(ir_reports, "_can_fetch", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(ir_reports, "_fetch_url_bytes", _fetch)

    assert ir_reports.run(ctx, _Log()) == 0
    assert ir_reports.run(ctx, _Log()) == 0

    idx = pd.read_parquet(tmp_path / "data" / "raw" / "ir" / "index.parquet")
    assert calls["n"] == 1
    assert "DUPLICATE" in set(idx["error_code"].astype(str).tolist())


def test_ir_reports_unparsable_pdf_sets_unparsable(tmp_path, monkeypatch):
    mapping = tmp_path / "config" / "ir_sources.csv"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_text("ticker,url,source,period,report_date\nEQNR,https://example.com/r1.pdf,example,Q4,2026-02-15\n", encoding="utf-8")

    ctx = _make_ctx(tmp_path, mapping)

    monkeypatch.setattr(ir_reports, "_can_fetch", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(ir_reports, "_fetch_url_bytes", lambda *_args, **_kwargs: (b"%PDF-1.4 bad", "application/pdf"))
    monkeypatch.setattr(ir_reports, "_extract_pdf_text", lambda _b: "")

    assert ir_reports.run(ctx, _Log()) == 0

    idx = pd.read_parquet(tmp_path / "data" / "raw" / "ir" / "index.parquet")
    assert idx.iloc[0]["error_code"] == "UNPARSABLE"


def test_ir_reports_is_optional_and_pipeline_continues_on_fail(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config" / "config.yaml").write_text(
        "paths:\n  runs_dir: runs\n  data_dir: data\n  raw_dir: data/raw\n  processed_dir: data/processed\n"
        "ir_reports:\n  mapping_csv: config/ir_sources.csv\n  rate_limit_sec: 0\n  timeout_sec: 1\n",
        encoding="utf-8",
    )
    (root / "config" / "ir_sources.csv").write_text(
        "ticker,url,source,period,report_date\nEQNR,https://example.com/r1.html,example,Q4,2026-02-15\n",
        encoding="utf-8",
    )

    dummy = types.ModuleType("src.dummy_decision")

    def _run_decision(ctx, _log):
        (ctx.run_dir / "decision.md").write_text("# Decision\n\nCASH", encoding="utf-8")
        return 0

    dummy.run = _run_decision
    sys.modules["src.dummy_decision"] = dummy

    from src.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "project_root_from_file", lambda: root)
    monkeypatch.setattr(ir_reports, "_can_fetch", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(ir_reports, "_fetch_url_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))

    monkeypatch.setattr(run_weekly, "DEFAULT_STEPS", [("decision", "src.dummy_decision")])
    monkeypatch.setattr(run_weekly, "OPTIONAL_STEPS", [("ir_reports", "src.ir_reports")])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_weekly",
            "--asof",
            "2026-02-16",
            "--config",
            "config/config.yaml",
            "--steps",
            "ir_reports,decision",
        ],
    )

    assert "ir_reports" not in run_weekly.resolve_steps_arg("all")

    rc = run_weekly.main()
    assert rc == 0

    run_dirs = sorted((root / "runs").glob("20260216_*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    assert (run_dir / "decision.md").exists()
    assert (root / "data" / "raw" / "ir" / "index.parquet").exists()
    idx = pd.read_parquet(root / "data" / "raw" / "ir" / "index.parquet")
    assert len(idx) >= 1
    assert "NETWORK" in set(idx["error_code"].astype(str).tolist())
