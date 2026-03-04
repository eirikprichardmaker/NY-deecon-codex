import time
import datetime as dt
from pathlib import Path

import src.gui_qt as gui_qt
from src.gui import (
    build_run_weekly_command,
    build_result_preview,
    build_test_result_html,
    extract_decision_sections_from_markdown,
    extract_pytest_summary,
    find_result_highlight_spans,
    find_latest_result_file,
    find_latest_decision_md,
    find_latest_test_report,
    format_result_label,
    is_previewable_result,
    list_recent_result_files,
    validate_asof,
    write_test_result_page,
)


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


def test_extract_pytest_summary_prefers_summary_line():
    output = (
        "line 1\n"
        "line 2\n"
        "138 passed in 149.54s (0:02:29)\n"
    )
    assert extract_pytest_summary(output) == "138 passed in 149.54s (0:02:29)"


def test_write_test_result_page_creates_html(tmp_path: Path):
    out_dir = tmp_path / "runs" / "gui_reports"
    started = dt.datetime(2026, 3, 3, 12, 0, 0)
    finished = dt.datetime(2026, 3, 3, 12, 0, 5)
    path = write_test_result_page(
        out_dir=out_dir,
        command=["python", "-m", "pytest", "-q"],
        exit_code=0,
        output="138 passed in 149.54s",
        started_at=started,
        finished_at=finished,
    )
    assert path.exists()
    txt = path.read_text(encoding="utf-8")
    assert "Deecon GUI Test Report" in txt
    assert "138 passed in 149.54s" in txt
    assert "PASS" in txt


def test_find_latest_test_report_picks_most_recent(tmp_path: Path):
    base = tmp_path / "runs"
    a = base / "gui_reports" / "test_report_20260303_120000.html"
    b = base / "gui_reports" / "test_report_20260303_120010.html"
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_text("A", encoding="utf-8")
    time.sleep(0.02)
    b.write_text("B", encoding="utf-8")
    assert find_latest_test_report(base) == b


def test_build_test_result_html_includes_command_and_exit():
    started = dt.datetime(2026, 3, 3, 12, 0, 0)
    finished = dt.datetime(2026, 3, 3, 12, 0, 2)
    txt = build_test_result_html(
        command=["python", "-m", "pytest", "-q"],
        exit_code=1,
        output="1 failed, 137 passed in 149.54s",
        started_at=started,
        finished_at=finished,
    )
    assert "FAIL" in txt
    assert "1 failed, 137 passed in 149.54s" in txt
    assert "python -m pytest -q" in txt


def test_list_recent_result_files_sorts_and_limits(tmp_path: Path):
    base = tmp_path / "runs"
    a = base / "run_a" / "result.txt"
    b = base / "run_b" / "decision.md"
    c = base / "run_c" / "test_report_20260303_120000.html"
    a.parent.mkdir(parents=True, exist_ok=True)
    b.parent.mkdir(parents=True, exist_ok=True)
    c.parent.mkdir(parents=True, exist_ok=True)
    a.write_text("A", encoding="utf-8")
    time.sleep(0.02)
    b.write_text("B", encoding="utf-8")
    time.sleep(0.02)
    c.write_text("C", encoding="utf-8")

    hits = list_recent_result_files(base, limit=2)
    assert len(hits) == 2
    assert hits[0] == c
    assert hits[1] == b


def test_find_latest_result_file_reads_supported_patterns(tmp_path: Path):
    base = tmp_path / "runs"
    old = base / "x" / "decision.md"
    new = base / "y" / "result.html"
    old.parent.mkdir(parents=True, exist_ok=True)
    new.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("old", encoding="utf-8")
    time.sleep(0.02)
    new.write_text("new", encoding="utf-8")
    assert find_latest_result_file(base) == new


def test_format_result_label_uses_relative_path(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    p = root / "runs" / "abc" / "result.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    label = format_result_label(p, root)
    assert "runs/abc/result.txt" in label


def test_is_previewable_result_by_file_name(tmp_path: Path):
    decision = tmp_path / "decision.md"
    decision.write_text("# hi", encoding="utf-8")
    html_file = tmp_path / "result.html"
    html_file.write_text("<html></html>", encoding="utf-8")
    assert is_previewable_result(decision) is True
    assert is_previewable_result(html_file) is True


def test_is_previewable_result_by_glob_pattern(tmp_path: Path):
    rep = tmp_path / "test_report_20260303_120000.html"
    rep.write_text("<html><body>ok</body></html>", encoding="utf-8")
    assert is_previewable_result(rep) is True


def test_build_result_preview_reads_result_txt(tmp_path: Path):
    p = tmp_path / "result.txt"
    p.write_text("line1\nline2\n", encoding="utf-8")
    txt = build_result_preview(p)
    assert "line1" in txt
    assert "File:" in txt


def test_build_result_preview_html_is_rendered_in_app(tmp_path: Path):
    p = tmp_path / "result.html"
    p.write_text("<html><head><title>R</title></head><body><h1>x</h1><p>warning and pass</p></body></html>", encoding="utf-8")
    txt = build_result_preview(p)
    assert "Title: R" in txt
    assert "x" in txt.lower()
    assert "warning" in txt.lower()


def test_build_result_preview_html_fallback_when_bs4_missing(tmp_path: Path, monkeypatch):
    p = tmp_path / "result.html"
    p.write_text("<html><head><title>Fallback</title></head><body><p>warning and pass</p></body></html>", encoding="utf-8")
    monkeypatch.setattr(gui_qt, "BeautifulSoup", None)
    txt = build_result_preview(p)
    assert "Title: Fallback" in txt
    assert "warning and pass" in txt.lower()


def test_build_result_preview_html_large_input_uses_fast_parser(tmp_path: Path):
    p = tmp_path / "result.html"
    body = "<p>warning block</p>" * ((gui_qt.HTML_PARSE_SOFT_LIMIT_CHARS // 20) + 20)
    p.write_text(f"<html><head><title>Big</title></head><body>{body}</body></html>", encoding="utf-8")
    txt = build_result_preview(p)
    assert "Title: Big" in txt
    assert "warning block" in txt.lower()


def test_find_result_highlight_spans_marks_red_and_green():
    txt = "status: PASS\nwarning: missing prices\ndrop=-12.5%\nready candidate\n"
    spans = find_result_highlight_spans(txt)
    tags = [tag for _, _, tag in spans]
    assert "flag_red" in tags
    assert "flag_green" in tags


def test_find_result_highlight_spans_avoids_substring_false_positives():
    txt = "compass value, ready signal\n drawdown -4.0%"
    spans = find_result_highlight_spans(txt)
    has_red_number = any(tag == "flag_red" and txt[s:e] == "-4.0%" for s, e, tag in spans)
    has_green_ready = any(tag == "flag_green" and txt[s:e].lower() == "ready" for s, e, tag in spans)
    has_false_pass = any(tag == "flag_green" and txt[s:e].lower() == "pass" for s, e, tag in spans)
    assert has_red_number is True
    assert has_green_ready is True
    assert has_false_pass is False


def test_find_result_highlight_spans_suppresses_green_on_red_line():
    txt = "status: stabil men mangler data\nready candidate\n"
    spans = find_result_highlight_spans(txt)
    first_line_green = any(tag == "flag_green" and s < txt.find("\n") for s, _, tag in spans)
    first_line_red = any(tag == "flag_red" and s < txt.find("\n") for s, _, tag in spans)
    second_line_green = any(tag == "flag_green" and s > txt.find("\n") for s, _, tag in spans)
    assert first_line_red is True
    assert first_line_green is False
    assert second_line_green is True


def test_extract_decision_sections_from_markdown_splits_main_sections():
    md = (
        "# Decision\n\n"
        "## Oversikt\nx\n\n"
        "## 1) Fundamental analyse\nf\n\n"
        "## 2) Aksje analyse\na\n\n"
        "## 3) Produkter og markedsforventning\np\n\n"
        "## 4) Nyheter og vurdering\nm\n\n"
        "## Beslutningsskjema\ns\n\n"
        "## Kvalitetssikring Av Verdier\nq\n"
    )
    out = extract_decision_sections_from_markdown(md)
    assert "Fundamental" in out["fundamental"]
    assert "Aksje" in out["stock"]
    assert "Produkter" in out["products"]
    assert "Nyheter" in out["media"]
    assert "Beslutningsskjema" in out["schema"]
    assert "Kvalitetssikring" in out["quality"]
