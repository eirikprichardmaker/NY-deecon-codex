from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),
]

REPORT_KEYWORDS = (
    "quarter",
    "q1",
    "q2",
    "q3",
    "q4",
    "interim",
    "report",
    "results",
    "delars",
    "delårs",
    "kvartal",
    "financial",
)


@dataclass(frozen=True)
class SourceRow:
    ticker: str
    calendar_url: str
    reports_url: str
    source: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _normalize_url(url: str) -> str:
    u = _safe_text(url)
    if not u:
        return ""
    p = urllib.parse.urlparse(u)
    if not p.scheme:
        u = "https://" + u
    return u


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _http_get(url: str, timeout_sec: int, user_agent: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


def _html_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")
    txt = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt).strip()


def _extract_candidate_dates(text: str, min_date: date, max_date: date) -> list[date]:
    out: list[date] = []
    seen: set[date] = set()
    for pat in DATE_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
                dt = pd.to_datetime(raw, errors="coerce", format="%Y-%m-%d")
            else:
                dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
            if pd.isna(dt):
                continue
            d = dt.date()
            if d < min_date or d > max_date:
                continue
            if d in seen:
                continue
            seen.add(d)
            out.append(d)
    out.sort()
    return out


def _pick_expected_date(candidates: list[date], today: date) -> date | None:
    if not candidates:
        return None
    future = [d for d in candidates if d >= today]
    if future:
        return min(future)
    return max(candidates)


def _extract_report_links(page_url: str, html_bytes: bytes) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html_bytes, "html.parser")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a"):
        href = _safe_text(a.get("href"))
        if not href:
            continue
        abs_url = urllib.parse.urljoin(page_url, href)
        low_url = abs_url.lower()
        text = _safe_text(a.get_text(" ", strip=True))
        low_text = text.lower()
        is_pdf = low_url.endswith(".pdf")
        has_kw = any(k in low_text or k in low_url for k in REPORT_KEYWORDS)
        if not is_pdf and not has_kw:
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append((abs_url, text))
    return out


def _safe_filename_from_url(url: str, content_type: str = "") -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "")
    if not name:
        name = "report.pdf" if "pdf" in content_type.lower() else "report.bin"
    if "." not in name:
        name += ".pdf" if "pdf" in content_type.lower() else ".bin"
    return name


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_state (
            ticker TEXT PRIMARY KEY,
            expected_date TEXT,
            source_url TEXT NOT NULL,
            last_checked_at TEXT NOT NULL,
            last_changed_at TEXT NOT NULL,
            raw_match TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            source_url TEXT NOT NULL,
            expected_date TEXT,
            raw_match TEXT,
            status TEXT NOT NULL,
            error_code TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schedule_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            old_expected_date TEXT,
            new_expected_date TEXT,
            source_url TEXT NOT NULL,
            reason TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS report_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            discovered_at TEXT NOT NULL,
            report_url TEXT NOT NULL,
            title TEXT,
            status TEXT NOT NULL,
            local_path TEXT,
            content_type TEXT,
            bytes INTEGER,
            sha256 TEXT,
            error_code TEXT,
            UNIQUE(ticker, report_url)
        )
        """
    )
    conn.commit()


def _load_sources(path: Path) -> list[SourceRow]:
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")
    df = pd.read_csv(path)
    for c in ["ticker", "calendar_url", "reports_url", "source"]:
        if c not in df.columns:
            df[c] = ""
    out: list[SourceRow] = []
    for _, r in df.iterrows():
        t = _safe_text(r.get("ticker")).upper()
        cal = _normalize_url(r.get("calendar_url"))
        rep = _normalize_url(r.get("reports_url"))
        src = _safe_text(r.get("source")) or "manual"
        if not t or not cal:
            continue
        out.append(SourceRow(ticker=t, calendar_url=cal, reports_url=rep or cal, source=src))
    return out


def _get_state(conn: sqlite3.Connection, ticker: str) -> tuple[str | None, str | None]:
    row = conn.execute(
        "SELECT expected_date, raw_match FROM schedule_state WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


def _upsert_state(
    conn: sqlite3.Connection,
    ticker: str,
    source_url: str,
    expected_date: str | None,
    raw_match: str | None,
    now_iso: str,
) -> None:
    conn.execute(
        """
        INSERT INTO schedule_state(ticker, expected_date, source_url, last_checked_at, last_changed_at, raw_match)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            expected_date=excluded.expected_date,
            source_url=excluded.source_url,
            last_checked_at=excluded.last_checked_at,
            raw_match=excluded.raw_match
        """,
        (ticker, expected_date, source_url, now_iso, now_iso, raw_match),
    )


def _mark_changed(conn: sqlite3.Connection, ticker: str, now_iso: str) -> None:
    conn.execute(
        "UPDATE schedule_state SET last_changed_at = ? WHERE ticker = ?",
        (now_iso, ticker),
    )


def _record_observation(
    conn: sqlite3.Connection,
    ticker: str,
    source_url: str,
    expected_date: str | None,
    raw_match: str | None,
    status: str,
    error_code: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO schedule_observations(
            ticker, observed_at, source_url, expected_date, raw_match, status, error_code
        ) VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, _utc_now().isoformat(), source_url, expected_date, raw_match, status, error_code or None),
    )


def _record_change(
    conn: sqlite3.Connection,
    ticker: str,
    old_expected_date: str | None,
    new_expected_date: str | None,
    source_url: str,
    reason: str,
) -> None:
    conn.execute(
        """
        INSERT INTO schedule_changes(
            ticker, changed_at, old_expected_date, new_expected_date, source_url, reason
        ) VALUES(?, ?, ?, ?, ?, ?)
        """,
        (ticker, _utc_now().isoformat(), old_expected_date, new_expected_date, source_url, reason),
    )


def _within_watch_window(expected_date: date | None, now_d: date, days_before: int, days_after: int) -> bool:
    if expected_date is None:
        return True
    return (expected_date - timedelta(days=days_before)) <= now_d <= (expected_date + timedelta(days=days_after))


def _already_seen_report(conn: sqlite3.Connection, ticker: str, report_url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM report_files WHERE ticker = ? AND report_url = ?",
        (ticker, report_url),
    ).fetchone()
    return row is not None


def _insert_report_discovery(conn: sqlite3.Connection, ticker: str, report_url: str, title: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO report_files(
            ticker, discovered_at, report_url, title, status
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (ticker, _utc_now().isoformat(), report_url, title, "DISCOVERED"),
    )


def _update_report_download(
    conn: sqlite3.Connection,
    ticker: str,
    report_url: str,
    status: str,
    local_path: str = "",
    content_type: str = "",
    nbytes: int = 0,
    sha256: str = "",
    error_code: str = "",
) -> None:
    conn.execute(
        """
        UPDATE report_files
        SET status = ?, local_path = ?, content_type = ?, bytes = ?, sha256 = ?, error_code = ?
        WHERE ticker = ? AND report_url = ?
        """,
        (
            status,
            local_path or None,
            content_type or None,
            int(nbytes) if nbytes else None,
            sha256 or None,
            error_code or None,
            ticker,
            report_url,
        ),
    )


def run_watch(
    sources_csv: Path,
    db_path: Path,
    downloads_dir: Path,
    timeout_sec: int = 20,
    user_agent: str = "Deecon-ReportWatch/1.0",
    schedule_lookback_days: int = 30,
    schedule_lookahead_days: int = 400,
    watch_days_before: int = 5,
    watch_days_after: int = 5,
    download_reports: bool = True,
) -> int:
    sources = _load_sources(sources_csv)
    if not sources:
        print(f"No valid sources in {sources_csv}")
        return 0

    db_path.parent.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        _init_db(conn)
        now = _utc_now()
        now_iso = now.isoformat()
        today = now.date()
        min_d = today - timedelta(days=schedule_lookback_days)
        max_d = today + timedelta(days=schedule_lookahead_days)

        for src in sources:
            try:
                cal_bytes, _ct = _http_get(src.calendar_url, timeout_sec=timeout_sec, user_agent=user_agent)
                text = _html_text(cal_bytes)
                cands = _extract_candidate_dates(text, min_date=min_d, max_date=max_d)
                picked = _pick_expected_date(cands, today=today)
                picked_iso = picked.isoformat() if picked else None
                raw_match = picked_iso

                prev_date, _prev_raw = _get_state(conn, src.ticker)
                _upsert_state(
                    conn,
                    ticker=src.ticker,
                    source_url=src.calendar_url,
                    expected_date=picked_iso,
                    raw_match=raw_match,
                    now_iso=now_iso,
                )
                _record_observation(
                    conn,
                    ticker=src.ticker,
                    source_url=src.calendar_url,
                    expected_date=picked_iso,
                    raw_match=raw_match,
                    status="OK",
                )
                if prev_date != picked_iso:
                    reason = "INIT" if prev_date is None else "DATE_CHANGED"
                    _record_change(
                        conn,
                        ticker=src.ticker,
                        old_expected_date=prev_date,
                        new_expected_date=picked_iso,
                        source_url=src.calendar_url,
                        reason=reason,
                    )
                    _mark_changed(conn, src.ticker, now_iso=now_iso)

                if not _within_watch_window(picked, today, days_before=watch_days_before, days_after=watch_days_after):
                    continue

                rep_bytes, rep_ct = _http_get(src.reports_url, timeout_sec=timeout_sec, user_agent=user_agent)
                links = _extract_report_links(src.reports_url, rep_bytes)
                for report_url, title in links:
                    if _already_seen_report(conn, src.ticker, report_url):
                        continue
                    _insert_report_discovery(conn, src.ticker, report_url, title)
                    if not download_reports:
                        continue
                    try:
                        b, ct = _http_get(report_url, timeout_sec=timeout_sec, user_agent=user_agent)
                        fn = _safe_filename_from_url(report_url, content_type=ct or rep_ct)
                        out_dir = downloads_dir / src.ticker / today.isoformat()
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / fn
                        out_path.write_bytes(b)
                        _update_report_download(
                            conn,
                            ticker=src.ticker,
                            report_url=report_url,
                            status="DOWNLOADED",
                            local_path=str(out_path),
                            content_type=(ct or rep_ct),
                            nbytes=len(b),
                            sha256=_sha256_bytes(b),
                        )
                    except Exception:
                        _update_report_download(
                            conn,
                            ticker=src.ticker,
                            report_url=report_url,
                            status="ERROR",
                            error_code="DOWNLOAD_FAILED",
                        )
            except Exception:
                _record_observation(
                    conn,
                    ticker=src.ticker,
                    source_url=src.calendar_url,
                    expected_date=None,
                    raw_match=None,
                    status="ERROR",
                    error_code="CALENDAR_FETCH_FAILED",
                )
        conn.commit()
    finally:
        conn.close()
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sources-csv", default="config/report_watch_sources.csv")
    p.add_argument("--db", default="data/processed/report_watch/report_watch.db")
    p.add_argument("--downloads-dir", default="data/raw/ir_auto")
    p.add_argument("--timeout-sec", type=int, default=20)
    p.add_argument("--user-agent", default="Deecon-ReportWatch/1.0")
    p.add_argument("--schedule-lookback-days", type=int, default=30)
    p.add_argument("--schedule-lookahead-days", type=int, default=400)
    p.add_argument("--watch-days-before", type=int, default=5)
    p.add_argument("--watch-days-after", type=int, default=5)
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    return run_watch(
        sources_csv=Path(args.sources_csv),
        db_path=Path(args.db),
        downloads_dir=Path(args.downloads_dir),
        timeout_sec=int(args.timeout_sec),
        user_agent=str(args.user_agent),
        schedule_lookback_days=int(args.schedule_lookback_days),
        schedule_lookahead_days=int(args.schedule_lookahead_days),
        watch_days_before=int(args.watch_days_before),
        watch_days_after=int(args.watch_days_after),
        download_reports=not bool(args.no_download),
    )


if __name__ == "__main__":
    raise SystemExit(main())
