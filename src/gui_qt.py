from __future__ import annotations

import datetime as _dt
import html
import re
import subprocess
import sys
import threading
import traceback
from collections import deque
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from src.run_weekly import DEFAULT_STEPS, OPTIONAL_STEPS

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - dependency/runtime specific
    BeautifulSoup = None

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover - environment-specific
    QtCore = None
    QtGui = None
    QtWidgets = None


def validate_asof(value: str) -> str:
    raw = (value or "").strip()
    try:
        _dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("asof must be YYYY-MM-DD") from exc
    return raw


def build_run_weekly_command(
    asof: str,
    config_path: str,
    run_dir: str | None,
    dry_run: bool,
    steps: Iterable[str],
) -> List[str]:
    cmd = [sys.executable, "-m", "src.run_weekly", "--asof", validate_asof(asof), "--config", config_path]
    if run_dir:
        cmd.extend(["--run-dir", run_dir])
    if dry_run:
        cmd.append("--dry-run")
    steps_list = [s.strip() for s in steps if s and s.strip()]
    if steps_list:
        cmd.extend(["--steps", ",".join(steps_list)])
    return cmd


def find_latest_decision_md(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    hits = list(runs_dir.rglob("decision.md"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def find_latest_decision_report(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    hits = list(runs_dir.rglob("decision_report.html"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def find_latest_decision_artifact(runs_dir: Path) -> Path | None:
    report = find_latest_decision_report(runs_dir)
    if report is not None:
        return report
    return find_latest_decision_md(runs_dir)


def find_latest_test_report(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    hits = list(base_dir.rglob("test_report_*.html"))
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def find_latest_data_quality_report(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    patterns = ("wft_data_quality_report.md", "data_quality_report.md", "data_qc_report.html")
    hits: List[Path] = []
    for pat in patterns:
        hits.extend([p for p in base_dir.rglob(pat) if p.is_file()])
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


RESULT_FILE_PATTERNS = (
    "decision.md",
    "decision_report.html",
    "top_candidates.md",
    "result.html",
    "result.txt",
    "test_report_*.html",
    "data_qc_report.html",
    "data_quality_report.md",
    "wft_data_quality_report.md",
)

PREVIEWABLE_RESULT_FILES = {
    "decision.md",
    "decision_report.html",
    "top_candidates.md",
    "result.html",
    "result.txt",
    "data_qc_report.html",
    "data_quality_report.md",
    "wft_data_quality_report.md",
}

PREVIEWABLE_RESULT_GLOBS = (
    "test_report_*.html",
)

RED_FLAG_TERMS = (
    "fail",
    "failed",
    "fatal",
    "error",
    "warning",
    "warn",
    "advarsel",
    "feil",
    "kritisk",
    "missing",
    "mangler",
    "stale",
    "blocked",
    "block",
    "dq_fail",
    "data_quality_fail",
    "data_invalid",
    "cash",
    "kontant",
    "negative",
    "negativ",
    "downside",
    "drawdown",
    "decline",
    "fall",
    "drop",
    "red flag",
    "risk",
    "risiko",
    "svak",
)

GREEN_FLAG_TERMS = (
    "pass",
    "ok",
    "ready",
    "klar",
    "candidate",
    "kandidat",
    "buy",
    "kjop",
    "kjÃ¸p",
    "strong",
    "sterk",
    "solid",
    "stabil",
    "sunn",
    "healthy",
    "lav risiko",
    "improved",
    "forbedret",
    "opp",
    "vekst",
    "growth",
    "success",
    "suksess",
    "very good",
    "veldig bra",
)

RED_NUMBER_PATTERN = re.compile(r"(?<!\w)-\d+(?:\.\d+)?%?")


def find_result_highlight_spans(text: str) -> List[Tuple[int, int, str]]:
    def _find_term_spans(terms: Iterable[str], tag: str) -> List[Tuple[int, int, str]]:
        out: List[Tuple[int, int, str]] = []
        for term in terms:
            pat = re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
            for match in pat.finditer(text):
                out.append((match.start(), match.end(), tag))
        return out

    red = _find_term_spans(RED_FLAG_TERMS, "flag_red")
    green_all = _find_term_spans(GREEN_FLAG_TERMS, "flag_green")
    for match in RED_NUMBER_PATTERN.finditer(text):
        red.append((match.start(), match.end(), "flag_red"))

    def _overlaps(a: Tuple[int, int, str], b: Tuple[int, int, str]) -> bool:
        return not (a[1] <= b[0] or b[1] <= a[0])

    def _line_bounds(span: Tuple[int, int, str]) -> Tuple[int, int]:
        start = span[0]
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        if line_end < 0:
            line_end = len(text)
        return line_start, line_end

    red_line_bounds = [_line_bounds(r) for r in red]

    green: List[Tuple[int, int, str]] = []
    for g in green_all:
        if any(_overlaps(g, r) for r in red):
            continue
        g0, g1 = _line_bounds(g)
        has_red_on_same_line = any(not (g1 <= r0 or r1 <= g0) for (r0, r1) in red_line_bounds)
        if has_red_on_same_line:
            continue
        green.append(g)

    return sorted(red + green, key=lambda x: (x[0], x[1]))


def list_recent_result_files(base_dir: Path, limit: int = 40) -> List[Path]:
    if limit <= 0 or not base_dir.exists():
        return []

    unique: Dict[str, Path] = {}
    for pattern in RESULT_FILE_PATTERNS:
        for path in base_dir.rglob(pattern):
            if path.is_file():
                unique[str(path.resolve())] = path

    out = sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return out[:limit]


def find_latest_result_file(base_dir: Path) -> Path | None:
    hits = list_recent_result_files(base_dir, limit=1)
    return hits[0] if hits else None


def format_result_label(path: Path, root_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    ts = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} | {rel}"


def is_previewable_result(path: Path) -> bool:
    name = path.name.lower()
    if name in PREVIEWABLE_RESULT_FILES:
        return True
    return any(fnmatch(name, pat.lower()) for pat in PREVIEWABLE_RESULT_GLOBS)


def _simple_html_to_text(raw_html: str) -> str:
    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", raw_html)
    txt = re.sub(r"(?is)<br\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</(p|div|h1|h2|h3|h4|h5|h6|li|tr)>", "\n", txt)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _table_to_lines(table) -> List[str]:
    lines: List[str] = []
    rows = table.find_all("tr")
    for row in rows[:40]:
        cells = row.find_all(["th", "td"])
        vals = [c.get_text(" ", strip=True) for c in cells]
        if vals:
            lines.append(" | ".join(vals))
    return lines


def _html_to_preview_text(raw_html: str) -> str:
    if BeautifulSoup is None:
        return _simple_html_to_text(raw_html)

    soup = BeautifulSoup(raw_html, "html.parser")
    lines: List[str] = []

    if soup.title and soup.title.get_text(strip=True):
        lines.append(f"Title: {soup.title.get_text(strip=True)}")
        lines.append("")

    body = soup.body if soup.body else soup
    tags = body.find_all(["h1", "h2", "h3", "p", "li", "table"])
    for tag in tags[:250]:
        if tag.name in {"h1", "h2", "h3"}:
            txt = tag.get_text(" ", strip=True)
            if txt:
                lines.append(txt.upper() if tag.name == "h1" else txt)
                lines.append("")
            continue
        if tag.name == "p":
            txt = tag.get_text(" ", strip=True)
            if txt:
                lines.append(txt)
            continue
        if tag.name == "li":
            txt = tag.get_text(" ", strip=True)
            if txt:
                lines.append(f"- {txt}")
            continue
        if tag.name == "table":
            table_lines = _table_to_lines(tag)
            if table_lines:
                lines.append("[Table]")
                lines.extend(table_lines)
                lines.append("")

    out = "\n".join(lines).strip()
    if out:
        return out
    return _simple_html_to_text(raw_html)


def _read_preview_text(path: Path, max_bytes: int = 2_000_000) -> Tuple[str, bool]:
    with path.open("rb") as fh:
        raw = fh.read(max_bytes + 1)
    truncated = len(raw) > max_bytes
    if truncated:
        raw = raw[:max_bytes]
    try:
        return raw.decode("utf-8"), truncated
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), truncated


def build_result_preview(path: Path, max_chars: int = 120_000, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return f"File not found:\n{path}"
    if not is_previewable_result(path):
        return (
            f"This file type is not previewed in-app:\n{path.name}\n\n"
            "Use a previewable file (for example decision_report.html, decision.md or result.txt)."
        )

    try:
        raw, truncated_by_bytes = _read_preview_text(path, max_bytes=max_bytes)
    except Exception as exc:
        return f"Could not read file:\n{path}\n\nError: {exc}"

    if path.suffix.lower() == ".html":
        rendered = _html_to_preview_text(raw)
    else:
        rendered = raw

    truncated_by_chars = len(rendered) > max_chars
    txt = rendered if not truncated_by_chars else rendered[:max_chars] + "\n\n... [truncated]"
    truncate_notice = ""
    if truncated_by_bytes:
        truncate_notice += f"\nPreview limited to first {max_bytes:,} bytes."
    if truncated_by_chars:
        truncate_notice += f"\nRendered output limited to first {max_chars:,} chars."
    mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return f"File: {path}\nModified: {mtime}{truncate_notice}\n\n{txt}"


def extract_decision_sections_from_markdown(text: str) -> Dict[str, str]:
    raw = text or ""
    lines = raw.splitlines()
    headings: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m:
            headings.append((i, m.group(1).strip()))

    sections: Dict[str, str] = {}
    if not headings:
        return {
            "all": raw.strip(),
            "overview": raw.strip(),
            "fundamental": "",
            "stock": "",
            "products": "",
            "media": "",
            "quality": "",
            "schema": "",
        }

    def _slice(start_idx: int, end_idx: int) -> str:
        if start_idx < 0 or end_idx <= start_idx:
            return ""
        return "\n".join(lines[start_idx:end_idx]).strip()

    intro = _slice(0, headings[0][0])
    blocks: Dict[str, str] = {}
    for j, (line_idx, heading) in enumerate(headings):
        end_line = headings[j + 1][0] if j + 1 < len(headings) else len(lines)
        block = _slice(line_idx, end_line)
        h = heading.lower()
        key = heading
        if "1)" in h and "fundamental" in h:
            key = "fundamental"
        elif "2)" in h and "aksje" in h:
            key = "stock"
        elif "3)" in h and ("produkt" in h or "marked" in h):
            key = "products"
        elif "4)" in h and ("nyhet" in h or "media" in h):
            key = "media"
        elif "beslutningsskjema" in h:
            key = "schema"
        elif "kvalitetssikring" in h or "quality" in h:
            key = "quality"
        elif "beslutningskommentar" in h:
            key = "overview"
        elif "oversikt" in h:
            key = "overview"
        blocks[key] = (blocks.get(key, "").strip() + "\n\n" + block).strip()

    overview_parts = [p for p in [intro, blocks.get("overview", "")] if p.strip()]
    sections["all"] = raw.strip()
    sections["overview"] = "\n\n".join(overview_parts).strip() if overview_parts else raw.strip()
    sections["fundamental"] = blocks.get("fundamental", "")
    sections["stock"] = blocks.get("stock", "")
    sections["products"] = blocks.get("products", "")
    sections["media"] = blocks.get("media", "")
    sections["quality"] = blocks.get("quality", "")
    sections["schema"] = blocks.get("schema", "")
    return sections


def extract_pytest_summary(output: str) -> str:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return "No pytest output captured."

    summary_pattern = re.compile(
        r"\b(\d+\s+passed|\d+\s+failed|\d+\s+errors?|\d+\s+skipped|no tests ran)\b",
        flags=re.IGNORECASE,
    )
    for line in reversed(lines):
        if summary_pattern.search(line):
            return line
    return lines[-1]


def build_test_result_html(
    command: List[str],
    exit_code: int,
    output: str,
    started_at: _dt.datetime,
    finished_at: _dt.datetime,
) -> str:
    summary = extract_pytest_summary(output)
    status = "PASS" if int(exit_code) == 0 else "FAIL"
    tokens = build_ui_tokens() if "build_ui_tokens" in globals() else {
        "text": "#111111",
        "surface": "#ffffff",
        "bg": "#fafafa",
        "border": "#dddddd",
        "success": "#0b6b0b",
        "error": "#8b0000",
        "log_bg": "#111111",
        "log_text": "#eeeeee",
    }
    status_color = tokens["success"] if status == "PASS" else tokens["error"]
    cmd_text = " ".join(command)
    duration_sec = max(0.0, (finished_at - started_at).total_seconds())

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deecon Test Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: {tokens["text"]}; background: {tokens["bg"]}; }}
    .card {{ background: {tokens["surface"]}; border: 1px solid {tokens["border"]}; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    .status {{ font-weight: 700; color: {status_color}; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: {tokens["log_bg"]}; color: {tokens["log_text"]}; border-radius: 8px; padding: 12px; }}
    code {{ font-family: Consolas, Menlo, monospace; }}
  </style>
</head>
<body>
  <h1>Deecon GUI Test Report</h1>
  <div class="card">
    <p><strong>Status:</strong> <span class="status">{html.escape(status)}</span></p>
    <p><strong>Summary:</strong> {html.escape(summary)}</p>
    <p><strong>Exit code:</strong> {int(exit_code)}</p>
    <p><strong>Started:</strong> {html.escape(started_at.isoformat(timespec="seconds"))}</p>
    <p><strong>Finished:</strong> {html.escape(finished_at.isoformat(timespec="seconds"))}</p>
    <p><strong>Duration:</strong> {duration_sec:.2f}s</p>
    <p><strong>Command:</strong> <code>{html.escape(cmd_text)}</code></p>
  </div>
  <div class="card">
    <h2>Raw Output</h2>
    <pre>{html.escape(output or "(no output)")}</pre>
  </div>
</body>
</html>
"""


def write_test_result_page(
    out_dir: Path,
    command: List[str],
    exit_code: int,
    output: str,
    started_at: _dt.datetime,
    finished_at: _dt.datetime,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = finished_at.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"test_report_{stamp}.html"
    content = build_test_result_html(
        command=command,
        exit_code=exit_code,
        output=output,
        started_at=started_at,
        finished_at=finished_at,
    )
    path.write_text(content, encoding="utf-8")
    return path

"""CHANGELOG (2026-03-04)
- AppShell layout + consistent 8px spacing.
- Token-based styling for colors/components.
- Robust process stop and streamed logging improvements.
"""

UI_SPACING = 8
LOG_FLUSH_INTERVAL_MS = 40
LOG_MAX_CHARS = 600_000
PREVIEW_SECTION_MAX_CHARS = 120_000


def build_ui_tokens() -> Dict[str, str]:
    return {
        "bg": "#f4f6f9",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "border": "#d8e0ea",
        "text": "#102133",
        "muted": "#5f7388",
        "primary": "#1f6feb",
        "primary_hover": "#1a5ec8",
        "primary_soft": "#d8e8ff",
        "success": "#0e8a4f",
        "success_soft": "#dbf5e7",
        "warn": "#b77100",
        "warn_soft": "#fff2d1",
        "error": "#b4233f",
        "error_soft": "#ffe1e7",
        "log_bg": "#0f1722",
        "log_text": "#d9e4f3",
        "disabled_bg": "#f1f4f8",
        "disabled_text": "#9cadbe",
        "disabled_border": "#d9e0e7",
        "destructive_hover": "#95203a",
        "log_border": "#243445",
        "on_primary": "#ffffff",
    }


def build_light_qss(tokens: Dict[str, str]) -> str:
    return f"""
* {{
  color: {tokens["text"]};
  font-family: "Segoe UI";
  font-size: 13px;
}}
QMainWindow {{
  background: {tokens["bg"]};
}}
QFrame[card="true"], QGroupBox {{
  background: {tokens["surface"]};
  border: 1px solid {tokens["border"]};
  border-radius: 12px;
}}
QGroupBox {{
  margin-top: 14px;
  padding-top: 12px;
}}
QGroupBox::title {{
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px;
  color: {tokens["text"]};
  font-size: 13px;
  font-weight: 600;
}}
QLabel[role="h1"] {{
  font-size: 24px;
  font-weight: 700;
}}
QLabel[role="h2"] {{
  font-size: 16px;
  font-weight: 600;
}}
QLabel[role="muted"], QLabel#subTitleLabel {{
  color: {tokens["muted"]};
}}
QLabel[role="help"] {{
  color: {tokens["muted"]};
  font-size: 12px;
}}
QLabel[role="error"] {{
  color: {tokens["error"]};
  font-size: 12px;
}}
QLabel#statusChip {{
  border-radius: 12px;
  padding: 5px 12px;
  font-size: 13px;
  font-weight: 700;
  background: {tokens["surface_alt"]};
  border: 1px solid {tokens["border"]};
}}
QLabel#statusChip[statusTone="running"], QLabel#statusChip[tone="running"] {{
  background: {tokens["primary_soft"]};
  color: {tokens["primary"]};
  border: 1px solid {tokens["primary"]};
}}
QLabel#statusChip[statusTone="ok"], QLabel#statusChip[tone="ok"] {{
  background: {tokens["success_soft"]};
  color: {tokens["success"]};
  border: 1px solid {tokens["success"]};
}}
QLabel#statusChip[statusTone="warn"], QLabel#statusChip[tone="warn"] {{
  background: {tokens["warn_soft"]};
  color: {tokens["warn"]};
  border: 1px solid {tokens["warn"]};
}}
QLabel#statusChip[statusTone="error"], QLabel#statusChip[tone="error"] {{
  background: {tokens["error_soft"]};
  color: {tokens["error"]};
  border: 1px solid {tokens["error"]};
}}
QLineEdit, QComboBox, QListWidget, QTextEdit {{
  background: {tokens["surface_alt"]};
  border: 1px solid {tokens["border"]};
  border-radius: 10px;
  padding: 8px;
}}
QLineEdit:focus, QComboBox:focus, QListWidget:focus, QTextEdit:focus {{
  border: 1px solid {tokens["primary"]};
}}
QListWidget#sideNav {{
  border: none;
  background: transparent;
  padding: 0;
}}
QListWidget#sideNav::item {{
  padding: 8px 10px;
  margin: 2px 0;
  border-radius: 8px;
}}
QListWidget#sideNav::item:selected {{
  background: {tokens["primary_soft"]};
  color: {tokens["primary"]};
  font-weight: 700;
}}
QPushButton {{
  border-radius: 10px;
  padding: 8px 12px;
  border: 1px solid {tokens["border"]};
  background: {tokens["surface_alt"]};
  font-size: 13px;
  font-weight: 600;
}}
QPushButton:hover:!disabled {{
  background: {tokens["primary_soft"]};
}}
QPushButton:focus {{
  border: 1px solid {tokens["primary"]};
}}
QPushButton:disabled {{
  background: {tokens["disabled_bg"]};
  color: {tokens["disabled_text"]};
  border: 1px solid {tokens["disabled_border"]};
}}
QPushButton#primaryBtn, QPushButton[variant="primary"] {{
  background: {tokens["primary"]};
  border: 1px solid {tokens["primary"]};
  color: {tokens["on_primary"]};
}}
QPushButton#primaryBtn:hover:!disabled, QPushButton[variant="primary"]:hover:!disabled {{
  background: {tokens["primary_hover"]};
}}
QPushButton[variant="secondary"] {{
  background: {tokens["surface_alt"]};
  color: {tokens["text"]};
}}
QPushButton[variant="destructive"] {{
  background: {tokens["error"]};
  border: 1px solid {tokens["error"]};
  color: {tokens["on_primary"]};
}}
QPushButton[variant="destructive"]:hover:!disabled {{
  background: {tokens["destructive_hover"]};
}}
QPushButton#subNavBtn[active="true"], QPushButton[subnav="true"][active="true"] {{
  background: {tokens["primary_soft"]};
  border: 1px solid {tokens["primary"]};
  color: {tokens["primary"]};
  font-weight: 700;
}}
QTextEdit#logEdit {{
  background: {tokens["log_bg"]};
  color: {tokens["log_text"]};
  border: 1px solid {tokens["log_border"]};
  border-radius: 10px;
  font-family: Consolas, Menlo, monospace;
  font-size: 12px;
}}
QTextEdit#previewEdit {{
  background: {tokens["surface_alt"]};
  color: {tokens["text"]};
  border: 1px solid {tokens["border"]};
  border-radius: 10px;
  font-family: Consolas, Menlo, monospace;
  font-size: 12px;
}}
QProgressBar {{
  border-radius: 8px;
  border: 1px solid {tokens["border"]};
  background: {tokens["surface_alt"]};
  text-align: center;
  min-height: 12px;
  max-height: 12px;
}}
QProgressBar::chunk {{
  border-radius: 7px;
  background: {tokens["primary"]};
}}
"""


if QtCore is not None and QtGui is not None and QtWidgets is not None:
    class ProcessWorker(QtCore.QThread):
        output_line = QtCore.Signal(str)
        finished_meta = QtCore.Signal(dict)

        def __init__(self, cmd: List[str], task: str, report_out_dir: Optional[Path] = None, parent=None) -> None:
            super().__init__(parent)
            self.cmd = list(cmd)
            self.task = str(task or "")
            self.report_out_dir = report_out_dir
            self._proc: Optional[subprocess.Popen[str]] = None
            self._stop_requested = False
            self._force_killed = False

        def stop(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None or self._stop_requested:
                return
            self._stop_requested = True
            self.output_line.emit("[control] stop requested, sending terminate...\n")

            def _terminate_then_kill() -> None:
                try:
                    proc.terminate()
                except Exception as exc:  # pragma: no cover - runtime/platform specific
                    self.output_line.emit(f"[error] terminate failed: {exc}\n")
                    return
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._force_killed = True
                    self.output_line.emit("[control] process still running after 2s, sending kill...\n")
                    try:
                        proc.kill()
                    except Exception as exc:  # pragma: no cover - runtime/platform specific
                        self.output_line.emit(f"[error] kill failed: {exc}\n")

            threading.Thread(target=_terminate_then_kill, daemon=True).start()

        def run(self) -> None:
            started_at = _dt.datetime.now()
            report_path: Optional[Path] = None
            rc = 99
            out_parts: List[str] = []
            try:
                self._proc = subprocess.Popen(
                    self.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert self._proc.stdout is not None
                while True:
                    line = self._proc.stdout.readline()
                    if line:
                        out_parts.append(line)
                        self.output_line.emit(line)
                        continue
                    if self._proc.poll() is not None:
                        break
                rc = int(self._proc.wait())
            except Exception as exc:  # pragma: no cover
                self.output_line.emit(f"[error] failed to execute command: {exc}\n")
                rc = 99
            finally:
                finished_at = _dt.datetime.now()
                if self.task == "tests":
                    report_dir = self.report_out_dir if self.report_out_dir is not None else (Path("runs") / "gui_reports")
                    try:
                        report_path = write_test_result_page(
                            out_dir=report_dir,
                            command=self.cmd,
                            exit_code=int(rc),
                            output="".join(out_parts),
                            started_at=started_at,
                            finished_at=finished_at,
                        )
                        self.output_line.emit(f"[test report] {report_path}\n")
                    except Exception as exc:  # pragma: no cover
                        self.output_line.emit(f"[error] could not write test report: {exc}\n")
                self.output_line.emit(f"\n[exit code: {rc}]\n")
                self.finished_meta.emit(
                    {
                        "task": self.task,
                        "rc": int(rc),
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "report_path": str(report_path) if report_path else "",
                        "stopped": bool(self._stop_requested),
                        "force_killed": bool(self._force_killed),
                    }
                )


    class DeeconGui(QtWidgets.QMainWindow):
        page_keys = ["overview", "decision", "tests", "dq"]

        def __init__(self) -> None:
            super().__init__()
            self.tokens = build_ui_tokens()
            self.setWindowTitle("Deecon Control Center (Qt)")
            self.resize(1360, 900)
            self.setMinimumSize(1150, 760)

            self.worker: Optional[ProcessWorker] = None
            self.last_test_report: Optional[Path] = None
            self.result_choices: Dict[str, Path] = {}
            self.section_path_labels: Dict[str, QtWidgets.QLabel] = {}
            self.section_previews: Dict[str, QtWidgets.QTextEdit] = {}
            self.decision_sections: Dict[str, str] = {}
            self.decision_section_buttons: Dict[str, QtWidgets.QPushButton] = {}
            self.decision_source_path: str = ""
            self.field_error_labels: Dict[str, QtWidgets.QLabel] = {}
            self._log_queue: deque[str] = deque()
            self._log_timer = QtCore.QTimer(self)
            self._log_timer.setInterval(LOG_FLUSH_INTERVAL_MS)
            self._log_timer.timeout.connect(self._flush_log_queue)
            self._previous_excepthook = sys.excepthook
            sys.excepthook = self._handle_unhandled_exception

            self._build_ui()
            self._wire_events()
            self._select_default_steps()
            self._refresh_results()
            self._set_status("Ready", tone="ok")
            self._update_command_preview()

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - UI lifecycle
            sys.excepthook = self._previous_excepthook
            super().closeEvent(event)

        def _build_ui(self) -> None:
            root = QtWidgets.QWidget(self)
            self.setCentralWidget(root)
            outer = QtWidgets.QVBoxLayout(root)
            outer.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            outer.setSpacing(UI_SPACING * 2)

            top = QtWidgets.QFrame()
            top.setObjectName("topBar")
            top.setProperty("card", True)
            top_l = QtWidgets.QVBoxLayout(top)
            top_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            top_l.setSpacing(UI_SPACING)
            top_row = QtWidgets.QHBoxLayout()
            top_row.setSpacing(UI_SPACING * 2)
            txt_col = QtWidgets.QVBoxLayout()
            txt_col.setSpacing(UI_SPACING // 2)
            title = QtWidgets.QLabel("Deecon Control Center")
            title.setObjectName("titleLabel")
            title.setProperty("role", "h1")
            subtitle = QtWidgets.QLabel("Kjor pipeline/model/tests med live log og resultat-preview.")
            subtitle.setObjectName("subTitleLabel")
            subtitle.setProperty("role", "muted")
            txt_col.addWidget(title)
            txt_col.addWidget(subtitle)
            self.status_chip = QtWidgets.QLabel("Ready")
            self.status_chip.setObjectName("statusChip")
            self.status_chip.setProperty("tone", "ok")
            self.task_status_label = QtWidgets.QLabel("Idle")
            self.task_status_label.setProperty("role", "muted")
            self.task_status_label.setAlignment(QtCore.Qt.AlignRight)
            status_col = QtWidgets.QVBoxLayout()
            status_col.setSpacing(UI_SPACING // 2)
            status_col.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
            status_col.addWidget(self.status_chip, 0, QtCore.Qt.AlignRight)
            status_col.addWidget(self.task_status_label, 0, QtCore.Qt.AlignRight)
            top_row.addLayout(txt_col, 1)
            top_row.addLayout(status_col)
            top_l.addLayout(top_row)
            self.run_progress = QtWidgets.QProgressBar()
            self.run_progress.setRange(0, 100)
            self.run_progress.setValue(0)
            self.run_progress.setTextVisible(False)
            top_l.addWidget(self.run_progress)
            outer.addWidget(top)

            body = QtWidgets.QHBoxLayout()
            body.setSpacing(UI_SPACING * 2)
            nav_card = QtWidgets.QFrame()
            nav_card.setProperty("card", True)
            nav_card.setFixedWidth(220)
            nav_l = QtWidgets.QVBoxLayout(nav_card)
            nav_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            nav_l.setSpacing(UI_SPACING)
            nav_title = QtWidgets.QLabel("Resultater")
            nav_title.setProperty("role", "h2")
            nav_l.addWidget(nav_title)
            self.side_nav = QtWidgets.QListWidget()
            self.side_nav.setObjectName("sideNav")
            self.side_nav.addItems(["Oversikt", "Decision", "Tests", "DQ"])
            self.side_nav.setCurrentRow(0)
            nav_l.addWidget(self.side_nav, 1)
            body.addWidget(nav_card, 0)

            right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            right_split.setChildrenCollapsible(False)
            right_split.addWidget(self._build_run_page())
            self.pages = QtWidgets.QStackedWidget()
            right_split.addWidget(self.pages)
            right_split.setStretchFactor(0, 5)
            right_split.setStretchFactor(1, 7)
            body.addWidget(right_split, 1)
            outer.addLayout(body, 1)

            self._build_overview_page()
            self._build_decision_page()
            self._build_tests_page()
            self._build_dq_page()

        def _wire_events(self) -> None:
            self.side_nav.currentRowChanged.connect(self.pages.setCurrentIndex)
            self.asof_edit.textChanged.connect(self._update_command_preview)
            self.config_edit.textChanged.connect(self._update_command_preview)
            self.run_dir_edit.textChanged.connect(self._update_command_preview)
            self.dry_run_box.toggled.connect(self._update_command_preview)
            self.steps_list.itemSelectionChanged.connect(self._update_command_preview)
            self.result_filter_edit.textChanged.connect(self._refresh_results)
            self.results_combo.currentTextChanged.connect(lambda _txt: self._preview_selected_result())

            QtGui.QShortcut(QtGui.QKeySequence("F5"), self, activated=self._run_model)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self, activated=self._run_pipeline)
            QtGui.QShortcut(QtGui.QKeySequence("Ctrl+T"), self, activated=self._run_tests)
            QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self._stop_pipeline)

        def _set_button_variant(self, btn: QtWidgets.QPushButton, variant: str) -> None:
            btn.setProperty("variant", str(variant or "secondary").strip().lower())

        def _add_labeled_field(
            self,
            parent: QtWidgets.QVBoxLayout,
            key: str,
            label_text: str,
            help_text: str,
            edit: QtWidgets.QLineEdit,
            browse_handler: Optional[Callable[[], None]] = None,
        ) -> None:
            block = QtWidgets.QWidget()
            block_l = QtWidgets.QVBoxLayout(block)
            block_l.setContentsMargins(0, 0, 0, 0)
            block_l.setSpacing(UI_SPACING // 2)
            label = QtWidgets.QLabel(label_text)
            block_l.addWidget(label)
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(UI_SPACING)
            row.addWidget(edit, 1)
            if browse_handler is not None:
                browse_btn = QtWidgets.QPushButton("Browse")
                self._set_button_variant(browse_btn, "secondary")
                browse_btn.clicked.connect(browse_handler)
                row.addWidget(browse_btn)
            block_l.addLayout(row)
            help_label = QtWidgets.QLabel(help_text)
            help_label.setProperty("role", "help")
            block_l.addWidget(help_label)
            err_label = QtWidgets.QLabel("")
            err_label.setProperty("role", "error")
            err_label.setVisible(False)
            block_l.addWidget(err_label)
            self.field_error_labels[key] = err_label
            parent.addWidget(block)

        def _build_run_page(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(UI_SPACING * 2)

            left = QtWidgets.QWidget()
            left_l = QtWidgets.QVBoxLayout(left)
            left_l.setContentsMargins(0, 0, 0, 0)
            left_l.setSpacing(UI_SPACING * 2)
            layout.addWidget(left, 0)

            right = QtWidgets.QWidget()
            right_l = QtWidgets.QVBoxLayout(right)
            right_l.setContentsMargins(0, 0, 0, 0)
            right_l.setSpacing(UI_SPACING * 2)
            layout.addWidget(right, 1)

            setup = QtWidgets.QGroupBox("Run Setup")
            setup_l = QtWidgets.QVBoxLayout(setup)
            setup_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            setup_l.setSpacing(UI_SPACING)
            self.asof_edit = QtWidgets.QLineEdit(_dt.date.today().isoformat())
            self.config_edit = QtWidgets.QLineEdit(r"config\config.yaml")
            self.run_dir_edit = QtWidgets.QLineEdit("")
            self._add_labeled_field(
                setup_l,
                key="asof",
                label_text="Asof (YYYY-MM-DD)",
                help_text="ISO-format dato for deterministisk as-of kjoring.",
                edit=self.asof_edit,
            )
            self._add_labeled_field(
                setup_l,
                key="config",
                label_text="Config",
                help_text="Velg config-fil som brukes av CLI-kjoringen.",
                edit=self.config_edit,
                browse_handler=self._pick_config,
            )
            self._add_labeled_field(
                setup_l,
                key="run_dir",
                label_text="Run dir (valgfri, ma eksistere)",
                help_text="Hvis satt ma mappen finnes. Bruk Browse.",
                edit=self.run_dir_edit,
                browse_handler=self._pick_run_dir,
            )
            self.dry_run_box = QtWidgets.QCheckBox("Dry run")
            setup_l.addWidget(self.dry_run_box)
            left_l.addWidget(setup)

            steps_group = QtWidgets.QGroupBox("Steps")
            steps_l = QtWidgets.QVBoxLayout(steps_group)
            steps_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            steps_l.setSpacing(UI_SPACING)
            preset_row = QtWidgets.QHBoxLayout()
            preset_row.setSpacing(UI_SPACING)
            b_def = QtWidgets.QPushButton("Default")
            b_all = QtWidgets.QPushButton("All")
            b_clear = QtWidgets.QPushButton("Clear")
            for btn in (b_def, b_all, b_clear):
                self._set_button_variant(btn, "secondary")
            b_def.clicked.connect(self._select_default_steps)
            b_all.clicked.connect(self._select_all_steps)
            b_clear.clicked.connect(self._clear_steps_selection)
            preset_row.addWidget(b_def)
            preset_row.addWidget(b_all)
            preset_row.addWidget(b_clear)
            preset_row.addStretch(1)
            steps_l.addLayout(preset_row)
            self.steps_list = QtWidgets.QListWidget()
            self.steps_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            for name, _ in DEFAULT_STEPS:
                self.steps_list.addItem(name)
            for name, _ in OPTIONAL_STEPS:
                self.steps_list.addItem(f"{name} (optional)")
            steps_l.addWidget(self.steps_list, 1)
            left_l.addWidget(steps_group, 1)

            cmd_group = QtWidgets.QGroupBox("Command Preview")
            cmd_l = QtWidgets.QVBoxLayout(cmd_group)
            cmd_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            cmd_l.setSpacing(UI_SPACING)
            self.command_preview_label = QtWidgets.QLabel("")
            self.command_preview_label.setWordWrap(True)
            self.command_preview_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            cmd_l.addWidget(self.command_preview_label)
            left_l.addWidget(cmd_group)

            act_group = QtWidgets.QGroupBox("Actions")
            act_l = QtWidgets.QGridLayout(act_group)
            act_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            act_l.setHorizontalSpacing(UI_SPACING)
            act_l.setVerticalSpacing(UI_SPACING)
            self.run_model_btn = QtWidgets.QPushButton("Run model (F5)")
            self._set_button_variant(self.run_model_btn, "primary")
            self.run_model_btn.clicked.connect(self._run_model)
            self.run_btn = QtWidgets.QPushButton("Run pipeline (Ctrl+R)")
            self._set_button_variant(self.run_btn, "secondary")
            self.run_btn.clicked.connect(self._run_pipeline)
            self.run_tests_btn = QtWidgets.QPushButton("Run tests (Ctrl+T)")
            self._set_button_variant(self.run_tests_btn, "secondary")
            self.run_tests_btn.clicked.connect(self._run_tests)
            self.stop_btn = QtWidgets.QPushButton("Stop (Esc)")
            self._set_button_variant(self.stop_btn, "destructive")
            self.stop_btn.clicked.connect(self._stop_pipeline)
            self.stop_btn.setEnabled(False)
            show_dec_btn = QtWidgets.QPushButton("Latest decision")
            show_test_btn = QtWidgets.QPushButton("Latest test")
            show_dq_btn = QtWidgets.QPushButton("Latest DQ")
            clear_log_btn = QtWidgets.QPushButton("Clear log")
            for btn in (show_dec_btn, show_test_btn, show_dq_btn, clear_log_btn):
                self._set_button_variant(btn, "secondary")
            show_dec_btn.clicked.connect(self._show_latest_decision_in_gui)
            show_test_btn.clicked.connect(self._show_latest_test_report_in_gui)
            show_dq_btn.clicked.connect(self._show_latest_dq_report_in_gui)
            clear_log_btn.clicked.connect(lambda: self.log_edit.clear())
            act_l.addWidget(self.run_model_btn, 0, 0)
            act_l.addWidget(self.run_btn, 0, 1)
            act_l.addWidget(self.run_tests_btn, 1, 0)
            act_l.addWidget(self.stop_btn, 1, 1)
            act_l.addWidget(show_dec_btn, 2, 0)
            act_l.addWidget(show_test_btn, 2, 1)
            act_l.addWidget(show_dq_btn, 3, 0)
            act_l.addWidget(clear_log_btn, 3, 1)
            left_l.addWidget(act_group)

            log_group = QtWidgets.QGroupBox("Live Log")
            log_l = QtWidgets.QVBoxLayout(log_group)
            log_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            log_l.setSpacing(UI_SPACING)
            self.log_edit = QtWidgets.QTextEdit()
            self.log_edit.setObjectName("logEdit")
            self.log_edit.setReadOnly(True)
            log_l.addWidget(self.log_edit)
            right_l.addWidget(log_group, 1)
            return panel

        def _create_preview_editor(self) -> QtWidgets.QTextEdit:
            edit = QtWidgets.QTextEdit()
            edit.setObjectName("previewEdit")
            edit.setReadOnly(True)
            return edit
        def _build_overview_page(self) -> None:
            page = QtWidgets.QWidget()
            self.pages.addWidget(page)
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(UI_SPACING * 2)
            controls = QtWidgets.QGroupBox("Resultatoversikt")
            c_l = QtWidgets.QVBoxLayout(controls)
            c_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            c_l.setSpacing(UI_SPACING)
            filter_row = QtWidgets.QHBoxLayout()
            filter_row.setSpacing(UI_SPACING)
            filter_row.addWidget(QtWidgets.QLabel("Filter"))
            self.result_filter_edit = QtWidgets.QLineEdit("")
            filter_row.addWidget(self.result_filter_edit, 1)
            self.result_stats_label = QtWidgets.QLabel("0 files")
            self.result_stats_label.setProperty("role", "muted")
            filter_row.addWidget(self.result_stats_label)
            c_l.addLayout(filter_row)

            self.results_combo = QtWidgets.QComboBox()
            c_l.addWidget(self.results_combo)
            btns = QtWidgets.QHBoxLayout()
            btns.setSpacing(UI_SPACING)
            refresh_btn = QtWidgets.QPushButton("Refresh")
            refresh_btn.clicked.connect(self._refresh_results)
            preview_btn = QtWidgets.QPushButton("Preview")
            preview_btn.clicked.connect(self._preview_selected_result)
            latest_btn = QtWidgets.QPushButton("Latest result")
            latest_btn.clicked.connect(self._show_latest_result_in_gui)
            latest_dec_btn = QtWidgets.QPushButton("Latest decision")
            latest_dec_btn.clicked.connect(self._show_latest_decision_in_gui)
            latest_test_btn = QtWidgets.QPushButton("Latest test")
            latest_test_btn.clicked.connect(self._show_latest_test_report_in_gui)
            latest_dq_btn = QtWidgets.QPushButton("Latest DQ")
            latest_dq_btn.clicked.connect(self._show_latest_dq_report_in_gui)
            for b in (refresh_btn, preview_btn, latest_btn, latest_dec_btn, latest_test_btn, latest_dq_btn):
                self._set_button_variant(b, "secondary")
                btns.addWidget(b)
            btns.addStretch(1)
            c_l.addLayout(btns)
            layout.addWidget(controls)

            prev_group = QtWidgets.QGroupBox("Preview")
            p_l = QtWidgets.QVBoxLayout(prev_group)
            p_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            p_l.setSpacing(UI_SPACING)
            label = QtWidgets.QLabel("No result selected")
            label.setProperty("role", "muted")
            p_l.addWidget(label)
            edit = self._create_preview_editor()
            p_l.addWidget(edit, 1)
            layout.addWidget(prev_group, 1)
            self.section_path_labels["overview"] = label
            self.section_previews["overview"] = edit

        def _build_decision_page(self) -> None:
            page = QtWidgets.QWidget()
            self.pages.addWidget(page)
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(UI_SPACING * 2)
            head = QtWidgets.QGroupBox("Decision")
            h_l = QtWidgets.QVBoxLayout(head)
            h_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            h_l.setSpacing(UI_SPACING)
            h_l.addWidget(QtWidgets.QLabel("Decision-seksjoner kan veksles under."))
            btn_row = QtWidgets.QHBoxLayout()
            btn_row.setSpacing(UI_SPACING)
            latest_dec_btn = QtWidgets.QPushButton("Latest decision")
            latest_dec_btn.clicked.connect(self._show_latest_decision_in_gui)
            refresh_btn = QtWidgets.QPushButton("Refresh")
            refresh_btn.clicked.connect(self._refresh_results)
            self._set_button_variant(latest_dec_btn, "secondary")
            self._set_button_variant(refresh_btn, "secondary")
            btn_row.addWidget(latest_dec_btn)
            btn_row.addWidget(refresh_btn)
            btn_row.addStretch(1)
            h_l.addLayout(btn_row)

            sub_row = QtWidgets.QHBoxLayout()
            sub_row.setSpacing(UI_SPACING)
            items = [
                ("overview", "Oversikt"),
                ("fundamental", "Fundamental"),
                ("stock", "Aksje"),
                ("products", "Produkter"),
                ("media", "Media"),
                ("quality", "Vurdering"),
                ("schema", "Skjema"),
            ]
            for key, label in items:
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName("subNavBtn")
                btn.setProperty("subnav", "true")
                btn.clicked.connect(lambda _checked=False, k=key: self._show_decision_subsection(k))
                sub_row.addWidget(btn)
                self.decision_section_buttons[key] = btn
            sub_row.addStretch(1)
            h_l.addLayout(sub_row)
            layout.addWidget(head)

            prev_group = QtWidgets.QGroupBox("Decision Preview")
            p_l = QtWidgets.QVBoxLayout(prev_group)
            p_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            p_l.setSpacing(UI_SPACING)
            label = QtWidgets.QLabel("No decision selected")
            label.setProperty("role", "muted")
            p_l.addWidget(label)
            edit = self._create_preview_editor()
            p_l.addWidget(edit, 1)
            layout.addWidget(prev_group, 1)
            self.section_path_labels["decision"] = label
            self.section_previews["decision"] = edit
            self._show_decision_subsection("overview", silent=True)

        def _build_tests_page(self) -> None:
            page = QtWidgets.QWidget()
            self.pages.addWidget(page)
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(UI_SPACING * 2)
            top = QtWidgets.QGroupBox("Test Reports")
            t_l = QtWidgets.QHBoxLayout(top)
            t_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            t_l.setSpacing(UI_SPACING)
            latest_btn = QtWidgets.QPushButton("Latest test report")
            latest_btn.clicked.connect(self._show_latest_test_report_in_gui)
            refresh_btn = QtWidgets.QPushButton("Refresh")
            refresh_btn.clicked.connect(self._refresh_results)
            self._set_button_variant(latest_btn, "secondary")
            self._set_button_variant(refresh_btn, "secondary")
            t_l.addWidget(latest_btn)
            t_l.addWidget(refresh_btn)
            t_l.addStretch(1)
            layout.addWidget(top)

            prev = QtWidgets.QGroupBox("Test Preview")
            p_l = QtWidgets.QVBoxLayout(prev)
            p_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            p_l.setSpacing(UI_SPACING)
            label = QtWidgets.QLabel("No test report selected")
            label.setProperty("role", "muted")
            p_l.addWidget(label)
            edit = self._create_preview_editor()
            p_l.addWidget(edit, 1)
            layout.addWidget(prev, 1)
            self.section_path_labels["tests"] = label
            self.section_previews["tests"] = edit

        def _build_dq_page(self) -> None:
            page = QtWidgets.QWidget()
            self.pages.addWidget(page)
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(UI_SPACING * 2)
            top = QtWidgets.QGroupBox("Data Quality")
            t_l = QtWidgets.QHBoxLayout(top)
            t_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            t_l.setSpacing(UI_SPACING)
            latest_btn = QtWidgets.QPushButton("Latest DQ report")
            latest_btn.clicked.connect(self._show_latest_dq_report_in_gui)
            refresh_btn = QtWidgets.QPushButton("Refresh")
            refresh_btn.clicked.connect(self._refresh_results)
            self._set_button_variant(latest_btn, "secondary")
            self._set_button_variant(refresh_btn, "secondary")
            t_l.addWidget(latest_btn)
            t_l.addWidget(refresh_btn)
            t_l.addStretch(1)
            layout.addWidget(top)

            prev = QtWidgets.QGroupBox("Data Quality Preview")
            p_l = QtWidgets.QVBoxLayout(prev)
            p_l.setContentsMargins(UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2, UI_SPACING * 2)
            p_l.setSpacing(UI_SPACING)
            label = QtWidgets.QLabel("No data-quality report selected")
            label.setProperty("role", "muted")
            p_l.addWidget(label)
            edit = self._create_preview_editor()
            p_l.addWidget(edit, 1)
            layout.addWidget(prev, 1)
            self.section_path_labels["dq"] = label
            self.section_previews["dq"] = edit

        def _set_status(self, text: str, tone: str = "neutral") -> None:
            t = str(tone).strip().lower()
            mapped = "neutral"
            if t in {"running", "run", "info"}:
                mapped = "running"
            elif t in {"ok", "success", "done"}:
                mapped = "ok"
            elif t in {"warn", "warning"}:
                mapped = "warn"
            elif t in {"bad", "error", "fail", "failed"}:
                mapped = "error"
            self.status_chip.setProperty("statusTone", mapped)
            self.status_chip.setProperty("tone", mapped)
            self.status_chip.setText(text)
            self.status_chip.style().unpolish(self.status_chip)
            self.status_chip.style().polish(self.status_chip)
            self.task_status_label.setText(text)

        def _set_field_error(self, key: str, message: str) -> None:
            label = self.field_error_labels.get(str(key).strip())
            if label is None:
                return
            msg = str(message or "").strip()
            label.setText(msg)
            label.setVisible(bool(msg))

        def _set_busy(self, busy: bool) -> None:
            for b in (self.run_btn, self.run_model_btn, self.run_tests_btn):
                b.setEnabled(not busy)
            self.stop_btn.setEnabled(bool(busy))
            if busy:
                self.run_progress.setRange(0, 0)
            else:
                self.run_progress.setRange(0, 100)
                self.run_progress.setValue(0)

        def _set_page(self, key: str) -> None:
            key_norm = str(key or "").strip().lower()
            if key_norm not in self.page_keys:
                key_norm = "overview"
            idx = self.page_keys.index(key_norm)
            self.side_nav.setCurrentRow(idx)

        def _selected_steps(self) -> List[str]:
            out: List[str] = []
            for item in self.steps_list.selectedItems():
                out.append(item.text().replace(" (optional)", "").strip())
            return out

        def _select_steps_by_names(self, names: Iterable[str]) -> None:
            wanted = {str(n).strip() for n in names if str(n).strip()}
            for i in range(self.steps_list.count()):
                item = self.steps_list.item(i)
                nm = item.text().replace(" (optional)", "").strip()
                item.setSelected(nm in wanted)
            self._update_command_preview()

        def _select_default_steps(self) -> None:
            self._select_steps_by_names([name for name, _ in DEFAULT_STEPS])

        def _select_all_steps(self) -> None:
            for i in range(self.steps_list.count()):
                self.steps_list.item(i).setSelected(True)
            self._update_command_preview()

        def _clear_steps_selection(self) -> None:
            for i in range(self.steps_list.count()):
                self.steps_list.item(i).setSelected(False)
            self._update_command_preview()

        def _update_command_preview(self) -> None:
            try:
                cmd = build_run_weekly_command(
                    asof=self.asof_edit.text().strip(),
                    config_path=self.config_edit.text().strip() or r"config\config.yaml",
                    run_dir=self.run_dir_edit.text().strip() or None,
                    dry_run=self.dry_run_box.isChecked(),
                    steps=self._selected_steps(),
                )
                preview = subprocess.list2cmdline(cmd)
            except Exception as exc:
                preview = f"(invalid command preview) {exc}"
            self.command_preview_label.setText(preview)

        def _pick_config(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select config file",
                str(Path.cwd()),
                "YAML Files (*.yaml *.yml);;All Files (*.*)",
            )
            if path:
                self.config_edit.setText(path)

        def _pick_run_dir(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select run directory", str(Path.cwd()))
            if path:
                self.run_dir_edit.setText(path)

        def _append_log(self, text: str) -> None:
            if not text:
                return
            self._log_queue.append(str(text))
            if not self._log_timer.isActive():
                self._log_timer.start()

        def _color_for_log_line(self, line: str) -> QtGui.QColor:
            lo = line.lower()
            if line.startswith("$ "):
                return QtGui.QColor(self.tokens["primary"])
            if "[error]" in lo or " fatal:" in lo or "traceback" in lo or " failed" in lo:
                return QtGui.QColor(self.tokens["error"])
            if "warning" in lo or " warn" in lo:
                return QtGui.QColor(self.tokens["warn"])
            if "step ok" in lo or "pipeline ok" in lo or " passed" in lo or "model finished" in lo:
                return QtGui.QColor(self.tokens["success"])
            return QtGui.QColor(self.tokens["log_text"])

        def _flush_log_queue(self) -> None:
            if not self._log_queue:
                self._log_timer.stop()
                return
            cursor = self.log_edit.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            merged: List[str] = []
            for _ in range(min(160, len(self._log_queue))):
                merged.append(self._log_queue.popleft())
            for line in "".join(merged).splitlines(True):
                fmt = QtGui.QTextCharFormat()
                fmt.setForeground(self._color_for_log_line(line))
                cursor.insertText(line, fmt)
            self.log_edit.setTextCursor(cursor)
            self.log_edit.ensureCursorVisible()
            plain = self.log_edit.toPlainText()
            if len(plain) > LOG_MAX_CHARS:
                self.log_edit.setPlainText(plain[-LOG_MAX_CHARS:])
                cur = self.log_edit.textCursor()
                cur.movePosition(QtGui.QTextCursor.End)
                self.log_edit.setTextCursor(cur)

        def _apply_flag_highlights(self, editor: QtWidgets.QTextEdit, text: str) -> None:
            selections: List[QtWidgets.QTextEdit.ExtraSelection] = []
            for start, end, tag in find_result_highlight_spans(text):
                sel = QtWidgets.QTextEdit.ExtraSelection()
                cur = editor.textCursor()
                cur.setPosition(start)
                cur.setPosition(end, QtGui.QTextCursor.KeepAnchor)
                fmt = QtGui.QTextCharFormat()
                fmt.setForeground(QtGui.QColor(self.tokens["error"] if tag == "flag_red" else self.tokens["success"]))
                sel.cursor = cur
                sel.format = fmt
                selections.append(sel)
            editor.setExtraSelections(selections)

        def _set_preview_text(self, key: str, text: str, path_text: str = "") -> None:
            editor = self.section_previews.get(key)
            label = self.section_path_labels.get(key)
            if editor is None:
                return
            preview_text = str(text or "")
            if len(preview_text) > PREVIEW_SECTION_MAX_CHARS:
                preview_text = preview_text[:PREVIEW_SECTION_MAX_CHARS] + "\n\n... [truncated]"
            editor.setPlainText(preview_text)
            self._apply_flag_highlights(editor, preview_text)
            cur = editor.textCursor()
            cur.movePosition(QtGui.QTextCursor.Start)
            editor.setTextCursor(cur)
            if label is not None:
                label.setText(path_text or "No file selected")
        def _start_worker(self, cmd: List[str], task: str, report_out_dir: Optional[Path] = None) -> None:
            if self.worker is not None and self.worker.isRunning():
                self._set_status("Another process is already running", tone="warn")
                return
            self._append_log(f"$ {subprocess.list2cmdline(cmd)}\n")
            if task == "tests":
                self._set_status("Running tests...", tone="running")
            elif task == "model":
                self._set_status("Running model...", tone="running")
            else:
                self._set_status("Running pipeline...", tone="running")
            self._set_busy(True)
            self.worker = ProcessWorker(cmd=cmd, task=task, report_out_dir=report_out_dir, parent=self)
            self.worker.output_line.connect(self._append_log)
            self.worker.finished_meta.connect(self._on_worker_finished)
            self.worker.start()

        def _validate_run_inputs(self) -> Tuple[bool, Optional[Path]]:
            self._set_field_error("asof", "")
            self._set_field_error("config", "")
            self._set_field_error("run_dir", "")

            has_error = False
            try:
                validate_asof(self.asof_edit.text().strip())
            except ValueError:
                self._set_field_error("asof", "Bruk format YYYY-MM-DD.")
                has_error = True

            config_path = self.config_edit.text().strip()
            if not config_path:
                self._set_field_error("config", "Config er paakrevd.")
                has_error = True
            elif not Path(config_path).exists():
                self._set_field_error("config", "Fant ikke config-filen.")
                has_error = True

            run_dir_raw = self.run_dir_edit.text().strip()
            run_dir_path: Optional[Path] = None
            if run_dir_raw:
                run_dir_path = Path(run_dir_raw)
                if not run_dir_path.exists():
                    self._set_field_error("run_dir", "Run-dir finnes ikke. Velg Browse.")
                    has_error = True
            if has_error:
                return False, None
            return True, run_dir_path

        def _ensure_selected_run_dir_exists(self, silent: bool = False) -> bool:
            raw = self.run_dir_edit.text().strip()
            if not raw:
                self._set_field_error("run_dir", "")
                return True
            p = Path(raw)
            if p.exists():
                self._set_field_error("run_dir", "")
                return True
            msg = "Selected run directory does not exist. Use Browse to select a valid folder."
            self._set_field_error("run_dir", "Run-dir finnes ikke. Velg Browse.")
            self._append_log(f"[error] {msg}\n")
            self._set_status("Invalid run_dir", tone="error")
            if not silent:
                QtWidgets.QMessageBox.critical(self, "Deecon GUI", msg)
            return False

        def _run_pipeline(self) -> None:
            try:
                ok, _run_dir_path = self._validate_run_inputs()
                if not ok:
                    self._show_error("Valideringsfeil i run setup. Sjekk feltene over.")
                    return
                config_path = self.config_edit.text().strip()
                cmd = build_run_weekly_command(
                    asof=self.asof_edit.text().strip(),
                    config_path=config_path,
                    run_dir=self.run_dir_edit.text().strip() or None,
                    dry_run=self.dry_run_box.isChecked(),
                    steps=self._selected_steps(),
                )
                self._start_worker(cmd, task="pipeline")
            except Exception as exc:
                self._handle_exception("Could not start pipeline", exc)

        def _run_model(self) -> None:
            try:
                ok, _run_dir_path = self._validate_run_inputs()
                if not ok:
                    self._show_error("Valideringsfeil i run setup. Sjekk feltene over.")
                    return
                config_path = self.config_edit.text().strip()
                cmd = build_run_weekly_command(
                    asof=self.asof_edit.text().strip(),
                    config_path=config_path,
                    run_dir=self.run_dir_edit.text().strip() or None,
                    dry_run=False,
                    steps=["valuation", "decision"],
                )
                self._start_worker(cmd, task="model")
            except Exception as exc:
                self._handle_exception("Could not start model run", exc)

        def _run_tests(self) -> None:
            try:
                if not self._ensure_selected_run_dir_exists(silent=False):
                    return
                cmd = [sys.executable, "-m", "pytest", "-q"]
                selected_run_dir = self.run_dir_edit.text().strip()
                report_out_dir = Path(selected_run_dir) if selected_run_dir else (Path("runs") / "gui_reports")
                self._start_worker(cmd, task="tests", report_out_dir=report_out_dir)
            except Exception as exc:
                self._handle_exception("Could not start tests", exc)

        def _stop_pipeline(self) -> None:
            if self.worker is not None and self.worker.isRunning():
                self.worker.stop()
                self._set_status("Stopping...", tone="warn")

        def _on_worker_finished(self, meta: dict) -> None:
            try:
                self._set_busy(False)
                self.worker = None
                task = str(meta.get("task", ""))
                rc = int(meta.get("rc", 1))
                stopped = bool(meta.get("stopped", False))
                self._refresh_results()

                if stopped:
                    self._set_status("Ready", tone="ok")
                    self._append_log("[control] process stopped by user.\n")
                    return

                if task == "tests":
                    report_path_raw = str(meta.get("report_path", "")).strip()
                    if report_path_raw:
                        self.last_test_report = Path(report_path_raw)
                    self._set_status("Tests finished" if rc == 0 else "Tests failed", tone=("ok" if rc == 0 else "error"))
                    if self.last_test_report and self.last_test_report.exists():
                        self._set_section_preview("tests", self.last_test_report)
                        self._set_page("tests")
                elif task == "model":
                    self._set_status("Model finished" if rc == 0 else "Model failed", tone=("ok" if rc == 0 else "error"))
                    if rc == 0:
                        self._show_latest_decision_in_gui(silent=True)
                else:
                    self._set_status("Finished" if rc == 0 else "Failed", tone=("ok" if rc == 0 else "error"))
                    if rc == 0:
                        self._show_latest_decision_in_gui(silent=True)
            except Exception as exc:
                self._handle_exception("Could not finalize process result", exc)

        def _show_error(self, text: str) -> None:
            self._set_status("Error", tone="error")
            self._append_log(f"[error] {text}\n")
            QtWidgets.QMessageBox.critical(self, "Deecon GUI", text)

        def _handle_exception(self, context: str, exc: BaseException) -> None:
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._set_status("Error", tone="error")
            self._append_log(f"[error] {context}: {exc}\n{trace}\n")
            QtWidgets.QMessageBox.critical(self, "Deecon GUI", f"{context}\n\n{exc}")

        def _handle_unhandled_exception(self, exc_type, exc_value, exc_traceback) -> None:  # pragma: no cover
            if issubclass(exc_type, KeyboardInterrupt):
                if self._previous_excepthook is not None:
                    self._previous_excepthook(exc_type, exc_value, exc_traceback)
                return
            trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self._set_status("Error", tone="error")
            self._append_log(f"[error] Unhandled exception: {exc_value}\n{trace}\n")
            QtWidgets.QMessageBox.critical(self, "Deecon GUI", f"Unhandled exception:\n\n{exc_value}")

        def _result_search_roots(self) -> List[Path]:
            roots: List[Path] = []
            seen: set[str] = set()
            selected_run_dir = self.run_dir_edit.text().strip()
            if selected_run_dir:
                selected = Path(selected_run_dir)
                if selected.exists():
                    key = str(selected.resolve())
                    if key not in seen:
                        seen.add(key)
                        roots.append(selected)
                    self._set_field_error("run_dir", "")
                else:
                    self._set_field_error("run_dir", "Run-dir finnes ikke. Velg Browse.")
            default_runs = Path("runs")
            if default_runs.exists():
                key = str(default_runs.resolve())
                if key not in seen:
                    seen.add(key)
                    roots.append(default_runs)
            return roots

        def _refresh_results(self) -> None:
            try:
                roots = self._result_search_roots()
                filter_txt = self.result_filter_edit.text().strip().lower()
                hits: List[Path] = []
                seen: set[str] = set()
                for root in roots:
                    for path in list_recent_result_files(root, limit=120):
                        key = str(path.resolve())
                        if key not in seen:
                            seen.add(key)
                            hits.append(path)
                hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if filter_txt:
                    hits = [p for p in hits if filter_txt in str(p).lower() or filter_txt in p.name.lower()]
                hits = hits[:120]

                current = self.results_combo.currentText().strip()
                self.result_choices = {}
                labels: List[str] = []
                for path in hits:
                    label = format_result_label(path, Path.cwd())
                    self.result_choices[label] = path
                    labels.append(label)
                self.result_stats_label.setText(f"{len(labels)} files")

                self.results_combo.blockSignals(True)
                self.results_combo.clear()
                self.results_combo.addItems(labels)
                if labels:
                    if current in self.result_choices:
                        self.results_combo.setCurrentText(current)
                    else:
                        self.results_combo.setCurrentIndex(0)
                self.results_combo.blockSignals(False)

                if labels:
                    self._preview_selected_result()
                    return
                if self.run_dir_edit.text().strip() and not Path(self.run_dir_edit.text().strip()).exists():
                    self._set_preview_text(
                        "overview",
                        "Selected run_dir does not exist.\nUse Browse to choose an existing folder, then refresh.",
                        "No result selected",
                    )
                else:
                    self._set_preview_text("overview", "No prior result files found.", "No result selected")
            except Exception as exc:
                self._handle_exception("Could not refresh results", exc)

        def _preview_selected_result(self) -> None:
            try:
                label = self.results_combo.currentText().strip()
                path = self.result_choices.get(label)
                if path is None:
                    self._set_preview_text("overview", "No selected result file.", "No result selected")
                    return
                txt = build_result_preview(path)
                self._set_preview_text("overview", txt, str(path))
                name = path.name.lower()
                if name in {"decision.md", "decision_report.html"}:
                    self._load_decision_sections(path)
            except Exception as exc:
                self._handle_exception("Could not preview selected result", exc)

        def _select_result_path(self, path: Path) -> bool:
            target = path.resolve() if path.exists() else path
            for label, candidate in self.result_choices.items():
                cand_resolved = candidate.resolve() if candidate.exists() else candidate
                if cand_resolved == target:
                    self.results_combo.setCurrentText(label)
                    self._preview_selected_result()
                    return True
            return False

        def _set_section_preview(self, key: str, path: Path) -> None:
            key_norm = str(key).strip().lower()
            txt = build_result_preview(path)
            self._set_preview_text(key_norm, txt, str(path))

        def _resolve_decision_markdown_path(self, artifact_path: Path) -> Path:
            p = Path(artifact_path)
            if p.name.lower() == "decision.md":
                return p
            sibling_md = p.with_name("decision.md")
            if sibling_md.exists():
                return sibling_md
            return p

        def _load_decision_sections(self, artifact_path: Path) -> None:
            src = self._resolve_decision_markdown_path(artifact_path)
            raw = ""
            try:
                if src.exists():
                    raw, _ = _read_preview_text(src, max_bytes=2_000_000)
            except Exception as exc:
                self._handle_exception("Could not read decision source", exc)
                raw = ""

            if src.suffix.lower() == ".md" and raw.strip():
                self.decision_sections = extract_decision_sections_from_markdown(raw)
            else:
                fallback = build_result_preview(artifact_path)
                self.decision_sections = {"all": fallback, "overview": fallback}
            self.decision_source_path = str(src if src.exists() else artifact_path)
            self._show_decision_subsection("overview", silent=True)

        def _show_decision_subsection(self, key: str, silent: bool = False) -> None:
            key_norm = str(key).strip().lower()
            valid = {"overview", "fundamental", "stock", "products", "media", "quality", "schema", "all"}
            if key_norm not in valid:
                key_norm = "overview"
            for k, btn in self.decision_section_buttons.items():
                btn.setProperty("active", "true" if k == key_norm else "false")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            if not self.decision_sections:
                if not silent:
                    self._set_preview_text("decision", "Ingen decision-data lastet enda.", f"{self.decision_source_path} | section={key_norm}")
                return
            text = self.decision_sections.get(key_norm, "").strip()
            if not text:
                text = self.decision_sections.get("overview", "").strip() or self.decision_sections.get("all", "").strip() or "(Tom seksjon)"
            self._set_preview_text("decision", text, f"{self.decision_source_path} | section={key_norm}")

        def _pick_latest(self, finder: Callable[[Path], Optional[Path]]) -> Optional[Path]:
            latest: Optional[Path] = None
            for root in self._result_search_roots():
                cand = finder(root)
                if cand is not None and (latest is None or cand.stat().st_mtime > latest.stat().st_mtime):
                    latest = cand
            return latest

        def _show_latest_decision_in_gui(self, silent: bool = False) -> None:
            try:
                if not self._ensure_selected_run_dir_exists(silent=silent):
                    return
                latest = self._pick_latest(find_latest_decision_artifact)
                if latest is None:
                    if not silent:
                        self._show_error("Could not find decision_report.html/decision.md in selected run-dir or runs/.")
                    return
                self._refresh_results()
                self._select_result_path(latest)
                self._load_decision_sections(latest)
                self._show_decision_subsection("overview", silent=True)
                self._set_page("decision")
                self._set_status(f"Previewing: {latest}", tone="ok")
            except Exception as exc:
                self._handle_exception("Could not open latest decision", exc)

        def _show_latest_test_report_in_gui(self, silent: bool = False) -> None:
            try:
                if not self._ensure_selected_run_dir_exists(silent=silent):
                    return
                latest = self._pick_latest(find_latest_test_report)
                if latest is None:
                    if not silent:
                        self._show_error("Could not find test report in selected run-dir or runs/.")
                    return
                self._refresh_results()
                self._select_result_path(latest)
                self._set_section_preview("tests", latest)
                self._set_page("tests")
                self._set_status(f"Previewing: {latest}", tone="ok")
            except Exception as exc:
                self._handle_exception("Could not open latest test report", exc)

        def _show_latest_dq_report_in_gui(self, silent: bool = False) -> None:
            try:
                if not self._ensure_selected_run_dir_exists(silent=silent):
                    return
                latest = self._pick_latest(find_latest_data_quality_report)
                if latest is None:
                    if not silent:
                        self._show_error("Could not find data quality report in selected run-dir or runs/.")
                    return
                self._refresh_results()
                self._select_result_path(latest)
                self._set_section_preview("dq", latest)
                self._set_page("dq")
                self._set_status(f"Previewing: {latest}", tone="ok")
            except Exception as exc:
                self._handle_exception("Could not open latest DQ report", exc)

        def _show_latest_result_in_gui(self, silent: bool = False) -> None:
            try:
                if not self._ensure_selected_run_dir_exists(silent=silent):
                    return
                latest = self._pick_latest(find_latest_result_file)
                if latest is None:
                    if not silent:
                        self._show_error("Could not find prior result file.")
                    return
                self._refresh_results()
                self._select_result_path(latest)
                self._set_page("overview")
                self._set_status(f"Previewing: {latest}", tone="ok")
            except Exception as exc:
                self._handle_exception("Could not open latest result", exc)


def main() -> int:
    if QtWidgets is None or QtCore is None or QtGui is None:
        print("PySide6 is not available in this Python environment. Install requirements and try again.", file=sys.stderr)
        return 1
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(build_light_qss(build_ui_tokens()))
    win = DeeconGui()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
