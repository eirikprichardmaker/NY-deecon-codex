from __future__ import annotations

import sqlite3
from pathlib import Path

from src import report_watch


def test_extract_candidate_dates_and_pick() -> None:
    txt = "Q1 release 2026-05-07, prior date 15.02.2026 and archive 2024-01-01"
    cands = report_watch._extract_candidate_dates(
        txt,
        min_date=report_watch.date(2026, 1, 1),
        max_date=report_watch.date(2026, 12, 31),
    )
    assert report_watch.date(2026, 5, 7) in cands
    assert report_watch.date(2026, 2, 15) in cands
    picked = report_watch._pick_expected_date(cands, today=report_watch.date(2026, 2, 20))
    assert picked == report_watch.date(2026, 5, 7)


def test_extract_report_links_absolute_and_relative() -> None:
    html = b"""
    <html><body>
      <a href="/files/q1_report.pdf">Quarterly report Q1</a>
      <a href="https://example.com/other">Other</a>
      <a href="/files/press-release">Interim results</a>
    </body></html>
    """
    links = report_watch._extract_report_links("https://example.com/ir", html)
    urls = [u for u, _ in links]
    assert "https://example.com/files/q1_report.pdf" in urls
    assert "https://example.com/files/press-release" in urls
    assert "https://example.com/other" not in urls


def test_run_watch_detects_date_change_and_downloads(tmp_path, monkeypatch) -> None:
    sources_csv = tmp_path / "sources.csv"
    sources_csv.write_text(
        "ticker,calendar_url,reports_url,source\n"
        "TEST,https://example.com/cal,https://example.com/reports,company_ir\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "watch.db"
    downloads = tmp_path / "downloads"

    calls = {"n": 0}

    def fake_http_get(url: str, timeout_sec: int, user_agent: str):
        if url.endswith("/cal"):
            calls["n"] += 1
            if calls["n"] == 1:
                return b"<html>Next report date 2026-05-07</html>", "text/html"
            return b"<html>Next report date 2026-05-10</html>", "text/html"
        if url.endswith("/reports"):
            return (
                b"<html><a href='/r/q1_report.pdf'>Quarterly report</a></html>",
                "text/html",
            )
        if url.endswith("q1_report.pdf"):
            return b"%PDF-1.4 fake", "application/pdf"
        raise AssertionError(f"Unexpected url: {url}")

    monkeypatch.setattr(report_watch, "_http_get", fake_http_get)

    rc1 = report_watch.run_watch(
        sources_csv=sources_csv,
        db_path=db_path,
        downloads_dir=downloads,
        watch_days_before=500,
        watch_days_after=500,
        download_reports=True,
    )
    assert rc1 == 0

    rc2 = report_watch.run_watch(
        sources_csv=sources_csv,
        db_path=db_path,
        downloads_dir=downloads,
        watch_days_before=500,
        watch_days_after=500,
        download_reports=True,
    )
    assert rc2 == 0

    con = sqlite3.connect(str(db_path))
    try:
        changes = con.execute("SELECT old_expected_date, new_expected_date, reason FROM schedule_changes").fetchall()
        assert len(changes) >= 2
        assert ("2026-05-07", "2026-05-10", "DATE_CHANGED") in changes

        reports = con.execute("SELECT status, local_path FROM report_files WHERE ticker='TEST'").fetchall()
        assert len(reports) == 1
        status, local_path = reports[0]
        assert status == "DOWNLOADED"
        assert local_path
        assert Path(local_path).exists()
    finally:
        con.close()

