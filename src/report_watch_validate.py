from __future__ import annotations

import argparse
import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src import report_watch


COMMON_CALENDAR_PATHS = [
    "",
    "/investors",
    "/investor-relations",
    "/investors/reports-and-presentations",
    "/investor-relations/reports-and-presentations",
    "/financial-reports-and-presentations",
]

COMMON_REPORT_PATHS = [
    "",
    "/investors/reports-and-presentations",
    "/investor-relations/reports-and-presentations",
    "/financial-reports-and-presentations",
    "/reports-and-presentations",
    "/press-releases",
]


@dataclass
class UrlEval:
    url: str
    ok: bool
    status: str
    date_count: int = 0
    future_date_count: int = 0
    report_link_count: int = 0
    score: float = -1.0


def _base_site(url: str) -> str:
    p = urllib.parse.urlparse(url)
    if not p.scheme or not p.netloc:
        return ""
    return f"{p.scheme}://{p.netloc}"


def _candidate_urls(seed: str, paths: list[str]) -> list[str]:
    seed = report_watch._normalize_url(seed)
    if not seed:
        return []
    base = _base_site(seed)
    out: list[str] = [seed]
    if base:
        for p in paths:
            if not p:
                continue
            out.append(base + p)
    seen = set()
    uniq = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def _eval_url(url: str, timeout_sec: int, user_agent: str, today: pd.Timestamp) -> UrlEval:
    try:
        b, _ct = report_watch._http_get(url, timeout_sec=timeout_sec, user_agent=user_agent)
        txt = report_watch._html_text(b)
        min_d = (today - timedelta(days=30)).date()
        max_d = (today + timedelta(days=400)).date()
        dates = report_watch._extract_candidate_dates(txt, min_date=min_d, max_date=max_d)
        future = [d for d in dates if d >= today.date()]
        links = report_watch._extract_report_links(url, b)

        score = 0.0
        score += min(len(future), 3) * 2.0
        score += min(len(dates), 5) * 0.5
        score += min(len(links), 10) * 0.3
        if len(txt) > 500:
            score += 0.5
        return UrlEval(
            url=url,
            ok=True,
            status="OK",
            date_count=len(dates),
            future_date_count=len(future),
            report_link_count=len(links),
            score=score,
        )
    except Exception:
        return UrlEval(url=url, ok=False, status="ERROR")


def validate_sources(
    input_csv: Path,
    output_csv: Path,
    timeout_sec: int = 20,
    user_agent: str = "Deecon-ReportWatchValidator/1.0",
) -> int:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")
    df = pd.read_csv(input_csv)
    for c in ["ticker", "calendar_url", "reports_url", "source"]:
        if c not in df.columns:
            df[c] = ""

    today = pd.Timestamp.utcnow().tz_localize(None)
    rows: list[dict] = []

    for _, r in df.iterrows():
        ticker = report_watch._safe_text(r.get("ticker")).upper()
        calendar_url = report_watch._normalize_url(r.get("calendar_url"))
        reports_url = report_watch._normalize_url(r.get("reports_url")) or calendar_url
        source = report_watch._safe_text(r.get("source")) or "manual"

        cal_candidates = _candidate_urls(calendar_url, COMMON_CALENDAR_PATHS)
        rep_candidates = _candidate_urls(reports_url, COMMON_REPORT_PATHS)

        cal_evals = [_eval_url(u, timeout_sec=timeout_sec, user_agent=user_agent, today=today) for u in cal_candidates]
        rep_evals = [_eval_url(u, timeout_sec=timeout_sec, user_agent=user_agent, today=today) for u in rep_candidates]

        best_cal = max(cal_evals, key=lambda x: x.score) if cal_evals else UrlEval(url=calendar_url, ok=False, status="NO_CANDIDATES")
        best_rep = max(rep_evals, key=lambda x: x.score) if rep_evals else UrlEval(url=reports_url, ok=False, status="NO_CANDIDATES")

        status = "OK" if best_cal.ok and best_rep.ok else "WARN"
        if best_cal.future_date_count == 0:
            status = "WARN"
        if best_rep.report_link_count == 0:
            status = "WARN"

        rows.append(
            {
                "ticker": ticker,
                "source": source,
                "input_calendar_url": calendar_url,
                "input_reports_url": reports_url,
                "suggested_calendar_url": best_cal.url,
                "suggested_reports_url": best_rep.url,
                "calendar_status": best_cal.status,
                "calendar_score": best_cal.score,
                "calendar_date_count": best_cal.date_count,
                "calendar_future_date_count": best_cal.future_date_count,
                "reports_status": best_rep.status,
                "reports_score": best_rep.score,
                "reports_link_count": best_rep.report_link_count,
                "status": status,
            }
        )

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="config/report_watch_sources.csv")
    p.add_argument("--output", default="tmp/report_watch_sources_validated.csv")
    p.add_argument("--timeout-sec", type=int, default=20)
    p.add_argument("--user-agent", default="Deecon-ReportWatchValidator/1.0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return validate_sources(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        timeout_sec=int(args.timeout_sec),
        user_agent=str(args.user_agent),
    )


if __name__ == "__main__":
    raise SystemExit(main())

