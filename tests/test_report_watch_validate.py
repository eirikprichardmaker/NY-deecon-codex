from __future__ import annotations

from pathlib import Path

from src import report_watch_validate as rwv


def test_candidate_urls_include_seed_and_base_paths() -> None:
    urls = rwv._candidate_urls("https://example.com/ir/page", ["", "/investors"])
    assert "https://example.com/ir/page" in urls
    assert "https://example.com/investors" in urls


def test_validate_sources_picks_best_urls(tmp_path: Path, monkeypatch) -> None:
    inp = tmp_path / "sources.csv"
    out = tmp_path / "out.csv"
    inp.write_text(
        "ticker,calendar_url,reports_url,source\n"
        "AAA,https://a.example/cal,https://a.example/reports,company_ir\n",
        encoding="utf-8",
    )

    def fake_eval(url: str, timeout_sec: int, user_agent: str, today):
        if url.endswith("/investors/reports-and-presentations"):
            return rwv.UrlEval(url=url, ok=True, status="OK", date_count=3, future_date_count=2, report_link_count=5, score=10.0)
        if url.endswith("/reports"):
            return rwv.UrlEval(url=url, ok=True, status="OK", date_count=0, future_date_count=0, report_link_count=1, score=1.0)
        return rwv.UrlEval(url=url, ok=False, status="ERROR", score=-1.0)

    monkeypatch.setattr(rwv, "_eval_url", fake_eval)
    rc = rwv.validate_sources(inp, out, timeout_sec=1)
    assert rc == 0
    txt = out.read_text(encoding="utf-8")
    assert "suggested_calendar_url" in txt
    assert "investors/reports-and-presentations" in txt

