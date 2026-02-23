import time
from pathlib import Path

from src.gui import build_run_weekly_command, validate_asof
from src.gui import find_latest_decision_md


def test_validate_asof_accepts_iso_date():
    assert validate_asof("2026-02-22") == "2026-02-22"


def test_validate_asof_rejects_invalid_date():
    try:
        validate_asof("22-02-2026")
        assert False, "Expected ValueError for invalid date format"
    except ValueError:
        assert True


def test_build_run_weekly_command_basic():
    cmd = build_run_weekly_command(
        asof="2026-02-22",
        config_path=r"config\config.yaml",
        run_dir=None,
        dry_run=False,
        steps=["valuation", "decision"],
    )
    assert cmd[:3] == [cmd[0], "-m", "src.run_weekly"]
    assert "--asof" in cmd and "2026-02-22" in cmd
    assert "--config" in cmd and r"config\config.yaml" in cmd
    assert "--steps" in cmd and "valuation,decision" in cmd


def test_build_run_weekly_command_optional_flags():
    cmd = build_run_weekly_command(
        asof="2026-02-22",
        config_path=r"config\config.yaml",
        run_dir=r"runs\manual",
        dry_run=True,
        steps=[],
    )
    assert "--dry-run" in cmd
    assert "--run-dir" in cmd and r"runs\manual" in cmd
    assert "--steps" not in cmd


def test_find_latest_decision_md_returns_none_when_missing(tmp_path: Path):
    assert find_latest_decision_md(tmp_path / "runs") is None


def test_find_latest_decision_md_picks_most_recent(tmp_path: Path):
    run_a = tmp_path / "runs" / "run_a"
    run_b = tmp_path / "runs" / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    a = run_a / "decision.md"
    b = run_b / "decision.md"
    a.write_text("A", encoding="utf-8")
    time.sleep(0.02)
    b.write_text("B", encoding="utf-8")

    latest = find_latest_decision_md(tmp_path / "runs")
    assert latest == b
