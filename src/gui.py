from __future__ import annotations

import datetime as _dt
import html
import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd
import yaml
from src.run_weekly import DEFAULT_STEPS, OPTIONAL_STEPS

RUN_WEEKLY_SUBPROCESS_FLAG = "--run-weekly-subprocess"


def is_frozen_bundle() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_base_dir() -> Path:
    if is_frozen_bundle():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def ensure_runtime_working_directory() -> None:
    if is_frozen_bundle():
        os.chdir(bundle_base_dir())

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - dependency/runtime specific
    BeautifulSoup = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover - environment-specific
    tk = None
    filedialog = None
    messagebox = None
    ttk = None


def validate_asof(value: str) -> str:
    raw = (value or "").strip()
    try:
        _dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError("asof must be YYYY-MM-DD") from exc
    return raw


def build_module_command(module_name: str, module_args: Iterable[str]) -> List[str]:
    args = [str(a) for a in module_args]
    if bool(getattr(sys, "frozen", False)):
        return [sys.executable, "--run-module", str(module_name), *args]
    return [sys.executable, "-m", str(module_name), *args]


def build_run_weekly_command(
    asof: str,
    config_path: str,
    run_dir: str | None,
    dry_run: bool,
    steps: Iterable[str],
) -> List[str]:
    asof_valid = validate_asof(asof)
    if is_frozen_bundle():
        cmd = [sys.executable, RUN_WEEKLY_SUBPROCESS_FLAG, "--asof", asof_valid, "--config", config_path]
    else:
        cmd = build_module_command(
            "src.run_weekly",
            ["--asof", asof_valid, "--config", config_path],
        )
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
    "kjøp",
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


def find_latest_model_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    latest_run: Path | None = None
    latest_ts = -1.0
    for decision_csv in base_dir.rglob("decision.csv"):
        if not decision_csv.is_file():
            continue
        try:
            ts = float(decision_csv.stat().st_mtime)
        except Exception:
            continue
        if ts > latest_ts:
            latest_ts = ts
            latest_run = decision_csv.parent
    return latest_run


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


def build_result_preview(path: Path, max_chars: int = 120_000) -> str:
    if not path.exists():
        return f"File not found:\n{path}"
    if not is_previewable_result(path):
        return (
            f"This file type is not previewed in-app:\n{path.name}\n\n"
            "Use a previewable file (for example decision_report.html, decision.md or result.txt)."
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1", errors="replace")
    except Exception as exc:
        return f"Could not read file:\n{path}\n\nError: {exc}"

    if path.suffix.lower() == ".html":
        rendered = _html_to_preview_text(raw)
    else:
        rendered = raw

    txt = rendered if len(rendered) <= max_chars else rendered[:max_chars] + "\n\n... [truncated]"
    mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return f"File: {path}\nModified: {mtime}\n\n{txt}"


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
    status_color = "#0b6b0b" if status == "PASS" else "#8b0000"
    cmd_text = " ".join(command)
    duration_sec = max(0.0, (finished_at - started_at).total_seconds())

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deecon Test Report</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #111; background: #fafafa; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
    .status {{ font-weight: 700; color: {status_color}; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #111; color: #eee; border-radius: 8px; padding: 12px; }}
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


@dataclass
class DataBlockMeta:
    source_files: List[str]
    asof: str
    last_updated: Dict[str, str]
    coverage: Dict[str, float]
    qc_status: str
    notes: str
    trust_score: int


@dataclass
class LoadedRunArtifacts:
    run_dir: Optional[Path]
    run_id: str
    asof: str
    config_path: str
    manifest: Dict[str, Any]
    dataframes: Dict[str, pd.DataFrame]
    texts: Dict[str, str]
    source_paths: Dict[str, Path]
    load_status: Dict[str, str]
    error: str = ""


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            if v != v:
                return None
            return v
        except Exception:
            return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        v = float(raw)
        if v != v:
            return None
        return v
    except Exception:
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes", "y"}:
        return True
    if raw in {"false", "0", "no", "n"}:
        return False
    return None


def _fmt_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "IKKE FUNNET"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        try:
            v = float(value)
        except Exception:
            return str(value)
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    txt = str(value).strip()
    return txt if txt else "IKKE FUNNET"


def _fmt_percent(value: Any, digits: int = 1) -> str:
    v = _to_float(value)
    if v is None:
        return "IKKE FUNNET"
    return f"{(v * 100.0):.{digits}f}%"


def _fmt_timestamp(ts: float) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "IKKE FUNNET"


def _path_mtime(path: Path) -> str:
    try:
        return _fmt_timestamp(path.stat().st_mtime)
    except Exception:
        return "IKKE FUNNET"


def load_decision_thresholds(config_path: Path | str | None) -> Dict[str, Optional[float]]:
    defaults: Dict[str, Optional[float]] = {
        "candidate_min_required_ratio": 0.75,
        "max_price_age_days": 7.0,
        "min_fresh_price_coverage": 0.50,
        "mad_min": None,
    }
    if not config_path:
        return defaults
    path = Path(config_path)
    if not path.exists():
        return defaults
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return defaults
    decision_cfg = payload.get("decision", {}) if isinstance(payload, dict) else {}
    if not isinstance(decision_cfg, dict):
        return defaults
    out = dict(defaults)
    for key in ("candidate_min_required_ratio", "max_price_age_days", "min_fresh_price_coverage", "mad_min"):
        if key in decision_cfg:
            out[key] = _to_float(decision_cfg.get(key))
    return out


def resolve_active_run_dir(
    selected_run_dir: Path | str | None,
    default_runs_dir: Path | str = Path("runs"),
) -> Path | None:
    def _latest_child_dir(path: Path) -> Path | None:
        dirs = [p for p in path.iterdir() if p.is_dir()]
        if not dirs:
            return None
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0]

    def _resolve_from_root(path: Path) -> Path | None:
        if not path.exists():
            return None
        if path.is_file():
            return path.parent
        artifact = find_latest_decision_artifact(path)
        if artifact is not None:
            return artifact.parent
        if any((path / name).exists() for name in ("manifest.json", "decision.csv", "decision.md", "decision_report.html")):
            return path
        return _latest_child_dir(path)

    if selected_run_dir:
        candidate = _resolve_from_root(Path(selected_run_dir))
        if candidate is not None:
            return candidate
    default_candidate = _resolve_from_root(Path(default_runs_dir))
    return default_candidate


def calculate_trust_score(
    record: Mapping[str, Any] | None,
    thresholds: Mapping[str, Any],
    has_decision_files: bool,
) -> Tuple[int, str, List[str], bool]:
    row = record or {}
    notes: List[str] = []
    if not has_decision_files:
        return 0, "FAIL", ["missing_decision_files"], True

    candidate_data_ok = _to_bool(row.get("candidate_data_ok"))
    if candidate_data_ok is False:
        return 0, "FAIL", ["candidate_data_ok=False"], True

    score = 100
    min_required_ratio = _to_float(thresholds.get("candidate_min_required_ratio")) or 0.75
    max_price_age_days = _to_float(thresholds.get("max_price_age_days")) or 7.0
    min_fresh_coverage = _to_float(thresholds.get("min_fresh_price_coverage")) or 0.50

    candidate_ratio = _to_float(row.get("candidate_data_coverage_ratio"))
    if candidate_ratio is not None and candidate_ratio < min_required_ratio:
        score -= 30
        notes.append(
            f"candidate_data_coverage_ratio={candidate_ratio:.3f} < candidate_min_required_ratio={min_required_ratio:.3f}"
        )

    stock_price_age_days = _to_float(row.get("stock_price_age_days"))
    if stock_price_age_days is not None and stock_price_age_days > max_price_age_days:
        score -= 20
        notes.append(
            f"stock_price_age_days={stock_price_age_days:.1f} > max_price_age_days={max_price_age_days:.1f}"
        )

    fresh_stock_coverage = _to_float(row.get("fresh_stock_price_coverage"))
    if fresh_stock_coverage is not None and fresh_stock_coverage < min_fresh_coverage:
        score -= 20
        notes.append(
            f"fresh_stock_price_coverage={fresh_stock_coverage:.3f} < min_fresh_price_coverage={min_fresh_coverage:.3f}"
        )

    unresolved_alerts = _to_float(row.get("value_qc_unresolved_alert_count"))
    if unresolved_alerts is not None and unresolved_alerts > 0:
        score -= 15
        notes.append(f"value_qc_unresolved_alert_count={int(unresolved_alerts)} > 0")

    score = max(0, min(100, int(round(score))))
    if score >= 80:
        status = "PASS"
    elif score >= 50:
        status = "WARN"
    else:
        status = "FAIL"
    return score, status, notes, False


def calculate_timing_label(
    fundamental_ok: Any,
    technical_ok: Any,
    data_blocked: bool = False,
) -> str:
    if data_blocked:
        return "NEI"
    fundamental = _to_bool(fundamental_ok)
    technical = _to_bool(technical_ok)
    if fundamental is not True:
        return "NEI"
    if technical is True:
        return "KJØP OK"
    return "VENT"


class DeeconGui:
    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.root.title("Deecon Pipeline GUI")
        self.root.geometry("1320x860")
        self.root.minsize(1120, 740)
        self.proc: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[object] = queue.Queue()
        self.last_test_report: Path | None = None
        self.result_choices: Dict[str, Path] = {}

        self.asof_var = tk.StringVar(value=_dt.date.today().isoformat())
        self.config_var = tk.StringVar(value=r"config\config.yaml")
        self.run_dir_var = tk.StringVar(value="")
        self.dry_run_var = tk.BooleanVar(value=False)
        self.result_var = tk.StringVar(value="")
        self.result_filter_var = tk.StringVar(value="")
        self.preview_path_var = tk.StringVar(value="No result selected")
        self.result_stats_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.command_preview_var = tk.StringVar(value="")
        self.selected_ticker_var = tk.StringVar(value="")
        self.ticker_query_var = tk.StringVar(value="")
        self.ticker_feedback_var = tk.StringVar(value="Skriv ticker og trykk 'Vurder ticker'.")
        self.active_run_dir_var = tk.StringVar(value="IKKE FUNNET")
        self.active_run_id_var = tk.StringVar(value="IKKE FUNNET")
        self.active_asof_var = tk.StringVar(value="IKKE FUNNET")
        self.active_config_path_var = tk.StringVar(value="IKKE FUNNET")
        self.active_company_var = tk.StringVar(value="IKKE FUNNET")
        self.active_trust_var = tk.StringVar(value="0")
        self.active_qc_var = tk.StringVar(value="FAIL")
        self.last_preview_text = ""
        self.decision_sections: Dict[str, str] = {}
        self.loaded: LoadedRunArtifacts = LoadedRunArtifacts(
            run_dir=None,
            run_id="IKKE FUNNET",
            asof="IKKE FUNNET",
            config_path="IKKE FUNNET",
            manifest={},
            dataframes={},
            texts={},
            source_paths={},
            load_status={},
            error="",
        )
        self.thresholds: Dict[str, Optional[float]] = load_decision_thresholds(Path(self.config_var.get()))
        self.join_key_error: str = ""
        self.ticker_items: Dict[str, Dict[str, str]] = {}
        self.tab_header_vars: Dict[str, Dict[str, "tk.StringVar"]] = {}
        self.tab_banner_vars: Dict[str, "tk.StringVar"] = {}
        self.tab_cards: Dict[str, Dict[str, "tk.StringVar"]] = {}
        self.tab_tables: Dict[str, "ttk.Treeview"] = {}
        self.tab_table_columns: Dict[str, List[str]] = {}
        self.tab_frames: Dict[str, "ttk.Frame"] = {}
        self.ticker_selectors: List["ttk.Combobox"] = []
        self.result_notebook: "ttk.Notebook | None" = None
        self.results_combo: "ttk.Combobox | None" = None
        self.preview_box: "tk.Text | None" = None
        self.command_preview_label: "ttk.Label | None" = None
        self.ticker_feedback_label: "ttk.Label | None" = None
        self.results_tab_index = 0
        self.agents_notebook: "ttk.Notebook | None" = None
        self.strategy_a_text: "tk.Text | None" = None

        self._configure_styles()
        self._build_layout()
        self._wire_reactive_updates()
        self._set_status("Ready", tone="ok")
        self._load_latest_data_on_startup()
        self._poll_logs()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        available = set(style.theme_names())
        for theme in ("vista", "xpnative", "clam", "alt", "default"):
            if theme in available:
                style.theme_use(theme)
                break

        self.root.option_add("*Font", "{Segoe UI} 11")
        self.root.option_add("*TCombobox*Listbox.Font", "{Segoe UI} 11")
        self.root.option_add("*TCombobox*Listbox.Background", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.Foreground", "#0f172a")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#dbeafe")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#0f172a")
        self.root.configure(background="#f2f5fa")

        style.configure("App.TFrame", background="#f2f5fa")
        style.configure("Hero.TFrame", background="#ffffff", relief="flat")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Section.TLabelframe", background="#ffffff", bordercolor="#d8e2f0", borderwidth=1, relief="solid")
        style.configure("Section.TLabelframe.Label", background="#ffffff", foreground="#162235", font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background="#ffffff", foreground="#30465f")
        style.configure("Info.TLabel", background="#ffffff", foreground="#102a43")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#2f435a")
        style.configure("Chip.TLabel", background="#eef2ff", foreground="#1f3248", padding=(8, 4), font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 7))
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Nav.TButton", font=("Segoe UI", 10, "bold"), anchor="w", padding=(10, 6))
        style.configure("NavActive.TButton", font=("Segoe UI", 10, "bold"), anchor="w", padding=(10, 6))
        style.map("NavActive.TButton", background=[("!disabled", "#dbeafe")], foreground=[("!disabled", "#1d4ed8")])
        style.configure("DangerBanner.TLabel", background="#fee2e2", foreground="#7f1d1d", padding=(10, 7), font=("Segoe UI", 10, "bold"))
        style.configure("WarnBanner.TLabel", background="#fef3c7", foreground="#7a4300", padding=(10, 7), font=("Segoe UI", 10, "bold"))
        style.configure("OkBanner.TLabel", background="#dcfce7", foreground="#14532d", padding=(10, 7), font=("Segoe UI", 10, "bold"))
        style.configure("CardValue.TLabel", background="#ffffff", foreground="#0f172a", justify="left", font=("Segoe UI", 10))
        style.configure("StatusNeutral.TLabel", background="#e7edf7", foreground="#1f2937", padding=(10, 5), font=("Segoe UI", 10, "bold"))
        style.configure("StatusRun.TLabel", background="#dbeafe", foreground="#1d4ed8", padding=(10, 5), font=("Segoe UI", 10, "bold"))
        style.configure("StatusOk.TLabel", background="#dcfce7", foreground="#166534", padding=(10, 5), font=("Segoe UI", 10, "bold"))
        style.configure("StatusWarn.TLabel", background="#fef3c7", foreground="#92400e", padding=(10, 5), font=("Segoe UI", 10, "bold"))
        style.configure("StatusBad.TLabel", background="#fee2e2", foreground="#991b1b", padding=(10, 5), font=("Segoe UI", 10, "bold"))
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#e5e7eb", bordercolor="#e5e7eb", background="#2563eb", lightcolor="#2563eb", darkcolor="#2563eb")
        style.configure("TNotebook", background="#f2f5fa", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 10), font=("Segoe UI", 10, "bold"))
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("!selected", "#e6eef8")],
            foreground=[("selected", "#0f172a"), ("!selected", "#334155")],
        )
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10), background="#ffffff", fieldbackground="#ffffff", foreground="#0f172a")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#e7edf7", foreground="#0f172a")
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#0f172a")])
        style.configure("TEntry", padding=(6, 5))
        style.configure("TCombobox", padding=(6, 5))

    def _set_status(self, text: str, tone: str = "neutral") -> None:
        self.status_var.set(text)
        if not hasattr(self, "status_chip"):
            return
        style = "StatusNeutral.TLabel"
        t = str(tone).strip().lower()
        if t in {"running", "run", "info"}:
            style = "StatusRun.TLabel"
        elif t in {"ok", "success", "done"}:
            style = "StatusOk.TLabel"
        elif t in {"warn", "warning"}:
            style = "StatusWarn.TLabel"
        elif t in {"bad", "error", "fail", "failed"}:
            style = "StatusBad.TLabel"
        self.status_chip.configure(style=style)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        if hasattr(self, "run_btn"):
            self.run_btn.configure(state=state)
        if hasattr(self, "run_model_btn"):
            self.run_model_btn.configure(state=state)
        if hasattr(self, "run_tests_btn"):
            self.run_tests_btn.configure(state=state)
        if hasattr(self, "stop_btn"):
            self.stop_btn.configure(state="normal" if busy else "disabled")
        if hasattr(self, "busy_bar"):
            if busy:
                self.busy_bar.start(12)
            else:
                self.busy_bar.stop()

    def _wire_reactive_updates(self) -> None:
        self.asof_var.trace_add("write", lambda *_: self._update_command_preview())
        self.config_var.trace_add("write", lambda *_: self._update_command_preview())
        self.run_dir_var.trace_add("write", lambda *_: self._update_command_preview())
        self.dry_run_var.trace_add("write", lambda *_: self._update_command_preview())
        self.result_filter_var.trace_add("write", lambda *_: self._refresh_results())
        self.selected_ticker_var.trace_add("write", lambda *_: self._render_results_for_ticker())
        if hasattr(self, "steps_list"):
            self.steps_list.bind("<<ListboxSelect>>", lambda _e: self._update_command_preview())
        self.root.bind("<F5>", lambda _e: self._run_model())
        self.root.bind("<Control-r>", lambda _e: self._run_pipeline())
        self.root.bind("<Control-t>", lambda _e: self._run_tests())
        self.root.bind("<Configure>", self._on_root_resize, add="+")
        self._update_command_preview()

    def _on_root_resize(self, _event=None) -> None:
        width = max(900, int(self.root.winfo_width()))
        if self.command_preview_label is not None:
            self.command_preview_label.configure(wraplength=max(460, width - 880))
        if self.ticker_feedback_label is not None:
            self.ticker_feedback_label.configure(wraplength=max(520, width - 500))

    def _select_steps_by_names(self, names: Iterable[str]) -> None:
        wanted = {str(n).strip() for n in names if str(n).strip()}
        self.steps_list.selection_clear(0, tk.END)
        for idx in range(self.steps_list.size()):
            name = self.steps_list.get(idx).replace(" (optional)", "").strip()
            if name in wanted:
                self.steps_list.selection_set(idx)
        self._update_command_preview()

    def _select_default_steps(self) -> None:
        self._select_steps_by_names([name for name, _ in DEFAULT_STEPS])

    def _select_all_steps(self) -> None:
        self.steps_list.selection_set(0, tk.END)
        self._update_command_preview()

    def _clear_steps_selection(self) -> None:
        self.steps_list.selection_clear(0, tk.END)
        self._update_command_preview()

    def _update_command_preview(self) -> None:
        if not hasattr(self, "steps_list"):
            return
        try:
            cmd = build_run_weekly_command(
                asof=self.asof_var.get(),
                config_path=self.config_var.get().strip() or r"config\config.yaml",
                run_dir=self.run_dir_var.get().strip() or None,
                dry_run=bool(self.dry_run_var.get()),
                steps=self._selected_steps(),
            )
            preview = subprocess.list2cmdline(cmd)
        except Exception as exc:
            preview = f"(invalid command preview) {exc}"
        self.command_preview_var.set(preview)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12, style="App.TFrame")
        root_frame.pack(fill="both", expand=True)

        hero = ttk.Frame(root_frame, padding=(16, 12), style="Hero.TFrame")
        hero.pack(fill="x", pady=(0, 10))
        hero.columnconfigure(0, weight=1)
        ttk.Label(hero, text="Deecon Control Center", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            hero,
            text="Kjor modell, pipeline og tester, og les alle resultater direkte i GUI.",
            style="SubHeader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        chips = ttk.Frame(hero, style="Hero.TFrame")
        chips.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(chips, text=f"Python {sys.version_info.major}.{sys.version_info.minor}", style="Chip.TLabel").pack(side="left", padx=(0, 6))
        ttk.Label(chips, text=f"Today {_dt.date.today().isoformat()}", style="Chip.TLabel").pack(side="left")

        self.tabs = ttk.Notebook(root_frame)
        self.tabs.pack(fill="both", expand=True)

        run_tab = ttk.Frame(self.tabs, padding=10, style="App.TFrame")
        results_tab = ttk.Frame(self.tabs, padding=10, style="App.TFrame")
        agents_tab = ttk.Frame(self.tabs, padding=10, style="App.TFrame")
        self.tabs.add(results_tab, text="Resultater")
        self.tabs.add(run_tab, text="Kjoring")
        self.tabs.add(agents_tab, text="Agenter")
        self._build_results_tab(results_tab)
        self._build_run_tab(run_tab)
        self._build_agents_tab(agents_tab)
        self._refresh_results()

    def _select_results_main_tab(self) -> None:
        try:
            self.tabs.select(self.results_tab_index)
        except Exception:
            pass

    def _load_latest_data_on_startup(self) -> None:
        self._select_results_main_tab()
        if not self._show_latest_model_run_in_gui(silent=True, refresh=False):
            self._show_latest_decision_in_gui(silent=True, refresh=False)

    def _build_run_tab(self, run_tab: "ttk.Frame") -> None:
        splitter = ttk.Panedwindow(run_tab, orient="horizontal")
        splitter.pack(fill="both", expand=True)

        control_col = ttk.Frame(splitter, padding=6, style="App.TFrame")
        log_col = ttk.Frame(splitter, padding=6, style="App.TFrame")
        splitter.add(control_col, weight=3)
        splitter.add(log_col, weight=5)

        setup_box = ttk.LabelFrame(control_col, text="Run Setup", padding=10, style="Section.TLabelframe")
        setup_box.pack(fill="x")
        setup_box.columnconfigure(1, weight=1)
        ttk.Label(setup_box, text="Asof (YYYY-MM-DD)", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(setup_box, textvariable=self.asof_var, width=20).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(setup_box, text="Config", style="Info.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(setup_box, textvariable=self.config_var).grid(row=1, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        ttk.Button(setup_box, text="Browse", style="Secondary.TButton", command=self._pick_config).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(setup_box, text="Run dir (optional)", style="Info.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(setup_box, textvariable=self.run_dir_var).grid(row=2, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        ttk.Button(setup_box, text="Browse", style="Secondary.TButton", command=self._pick_run_dir).grid(row=2, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(setup_box, text="Dry run", variable=self.dry_run_var).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        steps_box = ttk.LabelFrame(control_col, text="Steps", padding=10, style="Section.TLabelframe")
        steps_box.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(steps_box, text="Velg steg for pipeline-kjoring:", style="Muted.TLabel").pack(anchor="w")
        preset_row = ttk.Frame(steps_box, style="Card.TFrame")
        preset_row.pack(fill="x", pady=(6, 0))
        ttk.Button(preset_row, text="Default", style="Secondary.TButton", command=self._select_default_steps).pack(side="left")
        ttk.Button(preset_row, text="All", style="Secondary.TButton", command=self._select_all_steps).pack(side="left", padx=(6, 0))
        ttk.Button(preset_row, text="Clear", style="Secondary.TButton", command=self._clear_steps_selection).pack(side="left", padx=(6, 0))
        self.steps_list = tk.Listbox(
            steps_box,
            selectmode="multiple",
            exportselection=False,
            height=11,
            bg="#f8fafc",
            fg="#0f172a",
            selectbackground="#2563eb",
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            relief="flat",
            font=("Segoe UI", 11),
        )
        self.steps_list.pack(fill="both", expand=True, pady=(6, 0))
        for name, _ in DEFAULT_STEPS:
            self.steps_list.insert(tk.END, name)
        for name, _ in OPTIONAL_STEPS:
            self.steps_list.insert(tk.END, f"{name} (optional)")
        self.steps_list.select_set(0, len(DEFAULT_STEPS) - 1)

        cmd_box = ttk.LabelFrame(control_col, text="Command Preview", padding=10, style="Section.TLabelframe")
        cmd_box.pack(fill="x", pady=(10, 0))
        self.command_preview_label = ttk.Label(
            cmd_box,
            textvariable=self.command_preview_var,
            style="Info.TLabel",
            wraplength=520,
            justify="left",
        )
        self.command_preview_label.pack(anchor="w")

        actions_box = ttk.LabelFrame(control_col, text="Actions", padding=10, style="Section.TLabelframe")
        actions_box.pack(fill="x", pady=(10, 0))
        self.run_btn = ttk.Button(actions_box, text="Run pipeline", style="Primary.TButton", command=self._run_pipeline)
        self.run_btn.grid(row=0, column=0, sticky="we")
        self.run_model_btn = ttk.Button(actions_box, text="Run model", style="Primary.TButton", command=self._run_model)
        self.run_model_btn.grid(row=0, column=1, sticky="we", padx=(8, 0))
        self.run_tests_btn = ttk.Button(actions_box, text="Run tests", style="Secondary.TButton", command=self._run_tests)
        self.run_tests_btn.grid(row=1, column=0, sticky="we", pady=(8, 0))
        self.stop_btn = ttk.Button(actions_box, text="Stop", style="Secondary.TButton", command=self._stop_pipeline, state="disabled")
        self.stop_btn.grid(row=1, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        ttk.Button(actions_box, text="Show latest model run", style="Secondary.TButton", command=self._show_latest_model_run_in_gui).grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Button(actions_box, text="Show latest decision", style="Secondary.TButton", command=self._show_latest_decision_in_gui).grid(row=2, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        ttk.Button(actions_box, text="Show latest test report", style="Secondary.TButton", command=self._show_latest_test_report_in_gui).grid(row=3, column=0, columnspan=2, sticky="we", pady=(8, 0))
        ttk.Button(actions_box, text="Clear log", style="Secondary.TButton", command=self._clear_log).grid(row=4, column=0, columnspan=2, sticky="we", pady=(8, 0))
        actions_box.columnconfigure(0, weight=1)
        actions_box.columnconfigure(1, weight=1)

        status_box = ttk.Frame(control_col, padding=(10, 8), style="Card.TFrame")
        status_box.pack(fill="x", pady=(10, 0))
        ttk.Label(status_box, text="Status", style="Muted.TLabel").pack(anchor="w")
        self.status_chip = ttk.Label(status_box, textvariable=self.status_var, style="StatusNeutral.TLabel")
        self.status_chip.pack(anchor="w", pady=(4, 0))
        self.busy_bar = ttk.Progressbar(status_box, mode="indeterminate", style="Accent.Horizontal.TProgressbar")
        self.busy_bar.pack(fill="x", pady=(8, 0))

        log_box_wrap = ttk.LabelFrame(log_col, text="Live Log", padding=10, style="Section.TLabelframe")
        log_box_wrap.pack(fill="both", expand=True)
        self.log_box = tk.Text(
            log_box_wrap,
            wrap="word",
            bg="#0b1220",
            fg="#d9e4ff",
            insertbackground="#d9e4ff",
            relief="flat",
            borderwidth=0,
            font=("Consolas", 11),
            padx=8,
            pady=8,
            spacing1=1,
            spacing3=1,
        )
        self.log_box.tag_configure("log_cmd", foreground="#bfdbfe")
        self.log_box.tag_configure("log_info", foreground="#cbd5e1")
        self.log_box.tag_configure("log_warn", foreground="#f59e0b")
        self.log_box.tag_configure("log_error", foreground="#fca5a5")
        self.log_box.tag_configure("log_ok", foreground="#86efac")
        self.log_box.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(log_box_wrap, orient="vertical", command=self.log_box.yview)
        yscroll.pack(side="right", fill="y")
        self.log_box["yscrollcommand"] = yscroll.set
        self.log_box.configure(state="disabled")

    def _build_agents_tab(self, tab: "ttk.Frame") -> None:
        wrap = ttk.Frame(tab, style="App.TFrame")
        wrap.pack(fill="both", expand=True)

        toolbar = ttk.LabelFrame(wrap, text="Agent Toolbar", padding=10, style="Section.TLabelframe")
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="Refresh Agents", style="Secondary.TButton", command=self._refresh_agents).pack(side="left")

        self.agents_notebook = ttk.Notebook(wrap)
        self.agents_notebook.pack(fill="both", expand=True)

        # Strategy A tab
        strategy_a_tab = ttk.Frame(self.agents_notebook, padding=10, style="App.TFrame")
        self.agents_notebook.add(strategy_a_tab, text="Strategy A")
        self._build_strategy_a_tab(strategy_a_tab)

    def _build_strategy_a_tab(self, tab: "ttk.Frame") -> None:
        self.strategy_a_text = tk.Text(
            tab,
            wrap="word",
            bg="#f8fafc",
            fg="#0f172a",
            relief="flat",
            borderwidth=0,
            font=("Segoe UI", 11),
            padx=8,
            pady=8,
        )
        self.strategy_a_text.pack(fill="both", expand=True)
        yscroll = ttk.Scrollbar(tab, orient="vertical", command=self.strategy_a_text.yview)
        yscroll.pack(side="right", fill="y")
        self.strategy_a_text["yscrollcommand"] = yscroll.set
        self._load_strategy_a_recommendation()

    def _load_strategy_a_recommendation(self) -> None:
        path = Path("experiments/strategyA_recommendation.md")
        if path.exists():
            text = path.read_text(encoding="utf-8")
        else:
            text = "Strategy A recommendation not found. Run agent A first."
        self._set_text_widget_text(self.strategy_a_text, text)

    def _refresh_agents(self) -> None:
        self._load_strategy_a_recommendation()

    def _set_text_widget_text(self, widget: "tk.Text", text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        for start, end, tag in find_result_highlight_spans(text):
            i1 = f"1.0+{start}c"
            i2 = f"1.0+{end}c"
            widget.tag_add(tag, i1, i2)
        widget.see("1.0")
        widget.configure(state="disabled")

    def _build_results_tab(self, tab: "ttk.Frame") -> None:
        wrap = ttk.Frame(tab, style="App.TFrame")
        wrap.pack(fill="both", expand=True)

        toolbar = ttk.LabelFrame(wrap, text="Result Toolbar", padding=10, style="Section.TLabelframe")
        toolbar.pack(fill="x", pady=(0, 8))
        toolbar.columnconfigure(1, weight=1)
        toolbar.columnconfigure(4, weight=1)
        ttk.Label(toolbar, text="Filter", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar, textvariable=self.result_filter_var).grid(row=0, column=1, sticky="we", padx=(6, 8))
        ttk.Label(toolbar, textvariable=self.result_stats_var, style="Info.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Label(toolbar, text="Resultatfil", style="Info.TLabel").grid(row=0, column=3, sticky="w")
        self.results_combo = ttk.Combobox(toolbar, textvariable=self.result_var, state="readonly")
        self.results_combo.grid(row=0, column=4, sticky="we", padx=(6, 8))
        self.results_combo.bind("<<ComboboxSelected>>", lambda _e: self._preview_selected_result())
        ttk.Button(toolbar, text="Refresh", style="Secondary.TButton", command=self._refresh_results).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(toolbar, text="Latest result", style="Secondary.TButton", command=self._show_latest_result_in_gui).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(toolbar, text="Latest model run", style="Secondary.TButton", command=self._show_latest_model_run_in_gui).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(toolbar, text="Latest decision", style="Secondary.TButton", command=self._show_latest_decision_in_gui).grid(row=0, column=8, padx=(0, 6))
        ttk.Button(toolbar, text="Latest test", style="Secondary.TButton", command=self._show_latest_test_report_in_gui).grid(row=0, column=9, padx=(0, 6))
        ttk.Button(toolbar, text="Latest DQ", style="Secondary.TButton", command=self._show_latest_dq_report_in_gui).grid(row=0, column=10)
        ttk.Button(toolbar, text="Run Model", style="Primary.TButton", command=self._run_model).grid(row=0, column=11, padx=(0, 6))
        ttk.Label(toolbar, text="Ticker", style="Info.TLabel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ticker_entry = ttk.Entry(toolbar, textvariable=self.ticker_query_var)
        ticker_entry.grid(row=1, column=1, columnspan=3, sticky="we", padx=(6, 8), pady=(8, 0))
        ticker_entry.bind("<Return>", lambda _e: self._evaluate_ticker_query())
        ttk.Button(toolbar, text="Vurder ticker", style="Secondary.TButton", command=self._evaluate_ticker_query).grid(
            row=1, column=4, sticky="w", padx=(0, 8), pady=(8, 0)
        )
        self.ticker_feedback_label = ttk.Label(
            toolbar,
            textvariable=self.ticker_feedback_var,
            style="Info.TLabel",
            wraplength=900,
            justify="left",
        )
        self.ticker_feedback_label.grid(
            row=1, column=5, columnspan=6, sticky="w", pady=(8, 0)
        )

        self.result_notebook = ttk.Notebook(wrap)
        self.result_notebook.pack(fill="both", expand=True)

        tab_specs: List[Tuple[str, str, List[str], List[str], str]] = [
            (
                "snapshot",
                "Snapshot",
                [
                    "Decision Candidate",
                    "Fundamental Gate",
                    "Technical Gate",
                    "Data Sufficiency",
                    "Margin of Safety",
                    "Price Freshness",
                    "Quality Score",
                    "Risk Flags",
                ],
                ["field", "value", "source", "status"],
                "Snapshot Drilldown",
            ),
            (
                "fundamentals",
                "Fundamentals & Valuation",
                [
                    "Intrinsic Value",
                    "Market Cap",
                    "Margin of Safety",
                    "WACC / COE",
                    "ROIC-WACC Spread",
                    "Value Creation",
                    "Value QC Alerts",
                    "Sensitivity Coverage",
                ],
                ["metric", "value", "source", "asof"],
                "Valuation Drilldown",
            ),
            (
                "dq",
                "Data Quality & Provenance",
                [
                    "Trust Score",
                    "QC Status",
                    "DQ Fail Count",
                    "DQ Warn Count",
                    "Candidate Coverage",
                    "Fresh Price Coverage",
                    "Missing Fields",
                    "Provenance Notes",
                ],
                ["rule", "severity", "field", "detail"],
                "Data Quality Drilldown",
            ),
            (
                "timing",
                "Technical Timing",
                [
                    "Timing Conclusion",
                    "Fundamental OK",
                    "Technical OK",
                    "Stock Above MA200",
                    "MAD",
                    "Index MA200 OK",
                    "Price Age Days",
                    "Falling Knife",
                ],
                ["signal", "value", "threshold", "status"],
                "Technical Drilldown",
            ),
        ]

        for key, label, cards, columns, table_title in tab_specs:
            frame = ttk.Frame(self.result_notebook, padding=8, style="Card.TFrame")
            self.result_notebook.add(frame, text=label)
            self.tab_frames[key] = frame
            self._build_result_detail_tab(
                parent=frame,
                tab_key=key,
                card_titles=cards,
                table_columns=columns,
                table_title=table_title,
                include_preview=(key == "snapshot"),
            )

    def _build_result_detail_tab(
        self,
        parent: "ttk.Frame",
        tab_key: str,
        card_titles: List[str],
        table_columns: List[str],
        table_title: str,
        include_preview: bool = False,
    ) -> None:
        top = ttk.LabelFrame(parent, text="Run / Sources", padding=8, style="Section.TLabelframe")
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        header_vars = {
            "company": tk.StringVar(value="IKKE FUNNET"),
            "run_meta": tk.StringVar(value="asof=IKKE FUNNET | run_id=IKKE FUNNET | run_dir=IKKE FUNNET"),
            "config": tk.StringVar(value="config_path=IKKE FUNNET"),
            "sources": tk.StringVar(value="kilder: IKKE FUNNET"),
            "qc": tk.StringVar(value="qc_status=FAIL | trust_score=0"),
        }
        self.tab_header_vars[tab_key] = header_vars

        ttk.Label(top, text="Ticker", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(top, textvariable=self.selected_ticker_var, state="readonly", width=18)
        combo.grid(row=0, column=1, sticky="w")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._render_results_for_ticker())
        self.ticker_selectors.append(combo)
        ttk.Label(top, text="Company", style="Info.TLabel").grid(row=0, column=2, sticky="e", padx=(12, 6))
        ttk.Label(top, textvariable=header_vars["company"], style="Info.TLabel").grid(row=0, column=3, sticky="w")

        ttk.Label(top, textvariable=header_vars["run_meta"], style="Info.TLabel").grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(top, textvariable=header_vars["config"], style="Info.TLabel").grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))
        ttk.Label(top, textvariable=header_vars["sources"], style="Info.TLabel", wraplength=1080, justify="left").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(2, 0)
        )
        ttk.Label(top, textvariable=header_vars["qc"], style="Info.TLabel").grid(row=4, column=0, columnspan=4, sticky="w", pady=(2, 0))

        banner_var = tk.StringVar(value="")
        self.tab_banner_vars[tab_key] = banner_var
        banner = ttk.Label(parent, textvariable=banner_var, style="DangerBanner.TLabel")
        banner.pack(fill="x", pady=(6, 6))
        banner.pack_forget()
        banner._deecon_visible = False  # type: ignore[attr-defined]
        self.tab_frames[f"{tab_key}__banner"] = banner  # type: ignore[assignment]

        cards_wrap = ttk.Frame(parent, style="Card.TFrame")
        cards_wrap.pack(fill="x", pady=(0, 6))
        self.tab_cards[tab_key] = {}
        for idx, title in enumerate(card_titles):
            row = idx // 4
            col = idx % 4
            cards_wrap.columnconfigure(col, weight=1)
            card = ttk.LabelFrame(cards_wrap, text=title, padding=8, style="Section.TLabelframe")
            card.grid(row=row, column=col, sticky="nsew", padx=(0, 6), pady=(0, 6))
            value_var = tk.StringVar(value="IKKE FUNNET")
            self.tab_cards[tab_key][title] = value_var
            ttk.Label(card, textvariable=value_var, style="CardValue.TLabel", wraplength=260, justify="left").pack(anchor="w")

        table_box = ttk.LabelFrame(parent, text=table_title, padding=8, style="Section.TLabelframe")
        table_box.pack(fill="both", expand=True, pady=(0, 6))
        tree_area = ttk.Frame(table_box, style="Card.TFrame")
        tree_area.pack(fill="both", expand=True)
        tree = ttk.Treeview(tree_area, columns=table_columns, show="headings", height=10)
        for col in table_columns:
            tree.heading(col, text=col)
            tree.column(col, width=180, anchor="w", stretch=True)
        tree.column(table_columns[-1], width=240, anchor="w", stretch=True)
        tree.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(tree_area, orient="vertical", command=tree.yview)
        yscroll.pack(side="right", fill="y")
        xscroll = ttk.Scrollbar(table_box, orient="horizontal", command=tree.xview)
        xscroll.pack(fill="x", pady=(6, 0))
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.tag_configure("row_even", background="#ffffff")
        tree.tag_configure("row_odd", background="#f6f9ff")
        self.tab_tables[tab_key] = tree
        self.tab_table_columns[tab_key] = table_columns

        if include_preview:
            prev = ttk.LabelFrame(parent, text="Preview / Fallback", padding=8, style="Section.TLabelframe")
            prev.pack(fill="both", expand=True)
            ttk.Label(prev, textvariable=self.preview_path_var, style="Info.TLabel").pack(anchor="w")
            txt = tk.Text(
                prev,
                wrap="word",
                height=10,
                bg="#0f172a",
                fg="#e2e8f0",
                insertbackground="#e2e8f0",
                relief="flat",
                borderwidth=0,
                font=("Consolas", 11),
                padx=8,
                pady=8,
                spacing1=1,
                spacing3=1,
            )
            txt.pack(side="left", fill="both", expand=True, pady=(6, 0))
            txt.tag_configure("flag_red", foreground="#fecaca", background="#7f1d1d")
            txt.tag_configure("flag_green", foreground="#dcfce7", background="#14532d")
            txt.configure(state="disabled")
            scrollbar = ttk.Scrollbar(prev, orient="vertical", command=txt.yview)
            scrollbar.pack(side="right", fill="y", pady=(6, 0))
            txt.configure(yscrollcommand=scrollbar.set)
            self.preview_box = txt

    def _set_banner(self, tab_key: str, text: str, tone: str = "danger") -> None:
        banner = self.tab_frames.get(f"{tab_key}__banner")
        if not isinstance(banner, ttk.Label):
            return
        txt = str(text or "").strip()
        if not txt:
            if getattr(banner, "_deecon_visible", False):
                banner.pack_forget()
                banner._deecon_visible = False  # type: ignore[attr-defined]
            return
        style = "DangerBanner.TLabel"
        if tone == "warn":
            style = "WarnBanner.TLabel"
        elif tone == "ok":
            style = "OkBanner.TLabel"
        banner.configure(style=style)
        if not getattr(banner, "_deecon_visible", False):
            banner.pack(fill="x", pady=(6, 6))
            banner._deecon_visible = True  # type: ignore[attr-defined]
        self.tab_banner_vars[tab_key].set(txt)

    def _populate_table(self, tab_key: str, rows: List[Mapping[str, Any]]) -> None:
        tree = self.tab_tables.get(tab_key)
        columns = self.tab_table_columns.get(tab_key, [])
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        for idx, row in enumerate(rows[:200]):
            values = [str(row.get(col, "")) for col in columns]
            stripe_tag = "row_even" if idx % 2 == 0 else "row_odd"
            tree.insert("", "end", values=values, tags=(stripe_tag,))

    def _pick_config(self) -> None:
        if filedialog is None:
            return
        picked = filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if picked:
            self.config_var.set(picked)

    def _pick_run_dir(self) -> None:
        if filedialog is None:
            return
        picked = filedialog.askdirectory(title="Select run directory")
        if picked:
            self.run_dir_var.set(picked)

    def _selected_steps(self) -> List[str]:
        steps: List[str] = []
        for idx in self.steps_list.curselection():
            name = self.steps_list.get(idx)
            steps.append(name.replace(" (optional)", ""))
        return steps

    def _run_pipeline(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._set_status("Pipeline already running", tone="warn")
            return

        config_path = self.config_var.get().strip()
        if not config_path:
            self._show_error("Config is required.")
            return
        if not Path(config_path).exists():
            self._show_error(f"Config not found: {config_path}")
            return

        try:
            cmd = build_run_weekly_command(
                asof=self.asof_var.get(),
                config_path=config_path,
                run_dir=self.run_dir_var.get().strip() or None,
                dry_run=bool(self.dry_run_var.get()),
                steps=self._selected_steps(),
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self._append_log(f"$ {' '.join(cmd)}\n")
        self._set_status("Running pipeline...", tone="running")
        self._set_busy(True)

        def worker() -> None:
            started_at = _dt.datetime.now()
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            rc = self.proc.wait()
            self.log_queue.put(f"\n[exit code: {rc}]\n")
            self.log_queue.put(
                {
                    "type": "done",
                    "task": "pipeline",
                    "rc": int(rc),
                    "started_at": started_at,
                    "finished_at": _dt.datetime.now(),
                }
            )

        threading.Thread(target=worker, daemon=True).start()

    def _run_tests(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._set_status("Another process is already running", tone="warn")
            return

        cmd = [sys.executable, "-m", "pytest", "-q"]
        selected_run_dir = self.run_dir_var.get().strip()
        report_out_dir = Path(selected_run_dir) if selected_run_dir else (Path("runs") / "gui_reports")

        self._append_log(f"$ {' '.join(cmd)}\n")
        self._set_status("Running tests...", tone="running")
        self._set_busy(True)

        def worker() -> None:
            started_at = _dt.datetime.now()
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert self.proc.stdout is not None
            out_parts: list[str] = []
            for line in self.proc.stdout:
                out_parts.append(line)
                self.log_queue.put(line)
            rc = self.proc.wait()
            output = "".join(out_parts)
            finished_at = _dt.datetime.now()
            report_path = write_test_result_page(
                out_dir=report_out_dir,
                command=cmd,
                exit_code=int(rc),
                output=output,
                started_at=started_at,
                finished_at=finished_at,
            )
            self.log_queue.put(f"\n[exit code: {rc}]\n")
            self.log_queue.put(f"[test report] {report_path}\n")
            self.log_queue.put(
                {
                    "type": "done",
                    "task": "tests",
                    "rc": int(rc),
                    "report_path": str(report_path),
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
            )

        threading.Thread(target=worker, daemon=True).start()

    def _run_model(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._set_status("Another process is already running", tone="warn")
            return

        config_path = self.config_var.get().strip()
        if not config_path:
            self._show_error("Config is required.")
            return
        if not Path(config_path).exists():
            self._show_error(f"Config not found: {config_path}")
            return

        try:
            cmd = build_run_weekly_command(
                asof=self.asof_var.get(),
                config_path=config_path,
                run_dir=self.run_dir_var.get().strip() or None,
                dry_run=False,
                steps=["valuation", "decision"],
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self._append_log(f"$ {' '.join(cmd)}\n")
        self._set_status("Running model...", tone="running")
        self._set_busy(True)

        def worker() -> None:
            started_at = _dt.datetime.now()
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.log_queue.put(line)
            rc = self.proc.wait()
            self.log_queue.put(f"\n[exit code: {rc}]\n")
            self.log_queue.put(
                {
                    "type": "done",
                    "task": "model",
                    "rc": int(rc),
                    "started_at": started_at,
                    "finished_at": _dt.datetime.now(),
                }
            )

        threading.Thread(target=worker, daemon=True).start()

    def _stop_pipeline(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self._set_status("Stopping...", tone="warn")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")

    def _result_search_roots(self) -> List[Path]:
        roots: List[Path] = []
        seen: set[str] = set()
        selected_run_dir = self.run_dir_var.get().strip()
        if selected_run_dir:
            p = Path(selected_run_dir)
            if p.exists():
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    roots.append(p)
        runs_root = Path("runs")
        if runs_root.exists():
            key = str(runs_root.resolve())
            if key not in seen:
                seen.add(key)
                roots.append(runs_root)
        if not roots:
            roots.append(runs_root)
        return roots

    def _read_json_safe(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _read_text_safe(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return path.read_text(encoding="latin-1", errors="replace")
            except Exception:
                return ""
        except Exception:
            return ""

    def _read_csv_safe(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return pd.DataFrame()

    def _read_parquet_safe(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    def _load_run_artifacts(self, run_dir: Path | None) -> LoadedRunArtifacts:
        if run_dir is None or not run_dir.exists():
            return LoadedRunArtifacts(
                run_dir=run_dir,
                run_id="IKKE FUNNET",
                asof="IKKE FUNNET",
                config_path=self.config_var.get().strip() or "IKKE FUNNET",
                manifest={},
                dataframes={},
                texts={},
                source_paths={},
                load_status={},
                error="run_dir_not_found",
            )

        file_map = {
            "manifest": run_dir / "manifest.json",
            "decision_csv": run_dir / "decision.csv",
            "screen_basic_csv": run_dir / "screen_basic.csv",
            "valuation_csv": run_dir / "valuation.csv",
            "valuation_sensitivity_csv": run_dir / "valuation_sensitivity.csv",
            "dq_audit_csv": run_dir / "data_quality_audit.csv",
            "dq_by_ticker_csv": run_dir / "data_quality_by_ticker.csv",
            "decision_md": run_dir / "decision.md",
            "decision_report_html": run_dir / "decision_report.html",
            "dq_report_md": run_dir / "data_quality_report.md",
        }
        prices_path = Path("data") / "processed" / "prices.parquet"
        file_map["prices_parquet"] = prices_path

        source_paths: Dict[str, Path] = {}
        load_status: Dict[str, str] = {}
        for key, path in file_map.items():
            if path.exists():
                source_paths[key] = path
                load_status[key] = _path_mtime(path)
            else:
                load_status[key] = "IKKE FUNNET"

        manifest = self._read_json_safe(file_map["manifest"]) if file_map["manifest"].exists() else {}
        run_id = str(manifest.get("run_id") or run_dir.name)
        asof = str(manifest.get("asof") or "IKKE FUNNET")
        config_path = str(manifest.get("config_path") or self.config_var.get().strip() or "IKKE FUNNET")

        dataframes = {
            "decision": self._read_csv_safe(file_map["decision_csv"]) if file_map["decision_csv"].exists() else pd.DataFrame(),
            "screen_basic": self._read_csv_safe(file_map["screen_basic_csv"]) if file_map["screen_basic_csv"].exists() else pd.DataFrame(),
            "valuation": self._read_csv_safe(file_map["valuation_csv"]) if file_map["valuation_csv"].exists() else pd.DataFrame(),
            "valuation_sensitivity": self._read_csv_safe(file_map["valuation_sensitivity_csv"]) if file_map["valuation_sensitivity_csv"].exists() else pd.DataFrame(),
            "dq_audit": self._read_csv_safe(file_map["dq_audit_csv"]) if file_map["dq_audit_csv"].exists() else pd.DataFrame(),
            "dq_by_ticker": self._read_csv_safe(file_map["dq_by_ticker_csv"]) if file_map["dq_by_ticker_csv"].exists() else pd.DataFrame(),
            "prices": self._read_parquet_safe(file_map["prices_parquet"]) if file_map["prices_parquet"].exists() else pd.DataFrame(),
        }

        texts = {
            "decision_md": self._read_text_safe(file_map["decision_md"]) if file_map["decision_md"].exists() else "",
            "decision_report_html": self._read_text_safe(file_map["decision_report_html"]) if file_map["decision_report_html"].exists() else "",
            "dq_report_md": self._read_text_safe(file_map["dq_report_md"]) if file_map["dq_report_md"].exists() else "",
        }

        loaded = LoadedRunArtifacts(
            run_dir=run_dir,
            run_id=run_id,
            asof=asof,
            config_path=config_path,
            manifest=manifest,
            dataframes=dataframes,
            texts=texts,
            source_paths=source_paths,
            load_status=load_status,
            error="",
        )
        self._prepare_join_keys(loaded)
        return loaded

    def _prepare_join_keys(self, loaded: LoadedRunArtifacts) -> None:
        valuation = loaded.dataframes.get("valuation", pd.DataFrame())
        ticker_to_yahoo: Dict[str, str] = {}
        if not valuation.empty and "ticker" in valuation.columns and "yahoo_ticker" in valuation.columns:
            for _, row in valuation.iterrows():
                ticker = str(row.get("ticker", "")).strip()
                ytick = str(row.get("yahoo_ticker", "")).strip()
                if ticker and ytick and ticker not in ticker_to_yahoo:
                    ticker_to_yahoo[ticker] = ytick

        def _normalize(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            out = df.copy()
            if "ticker" in out.columns:
                out["__ticker"] = out["ticker"].map(lambda x: str(x).strip())
            elif "yahoo_ticker" in out.columns:
                out["__ticker"] = out["yahoo_ticker"].map(lambda x: str(x).strip())
            else:
                out["__ticker"] = ""

            if "yahoo_ticker" in out.columns:
                out["__yahoo_ticker"] = out["yahoo_ticker"].map(lambda x: str(x).strip())
            else:
                out["__yahoo_ticker"] = out["__ticker"].map(lambda x: ticker_to_yahoo.get(x, x))
            out["__yahoo_ticker"] = out["__yahoo_ticker"].fillna("").map(lambda x: str(x).strip())
            return out

        for key in ("decision", "screen_basic", "valuation", "valuation_sensitivity", "dq_by_ticker"):
            loaded.dataframes[key] = _normalize(loaded.dataframes.get(key, pd.DataFrame()))

        dq_audit = loaded.dataframes.get("dq_audit", pd.DataFrame())
        if not dq_audit.empty:
            audit = dq_audit.copy()
            if "ticker" in audit.columns:
                audit["__ticker"] = audit["ticker"].map(lambda x: str(x).strip())
            else:
                audit["__ticker"] = ""
            audit["__yahoo_ticker"] = audit["__ticker"].map(lambda x: ticker_to_yahoo.get(x, x))
            loaded.dataframes["dq_audit"] = audit

        self.join_key_error = ""
        try:
            for name in ("decision", "screen_basic", "valuation", "dq_by_ticker"):
                frame = loaded.dataframes.get(name, pd.DataFrame())
                if frame.empty or "__yahoo_ticker" not in frame.columns:
                    continue
                non_empty = frame[frame["__yahoo_ticker"].map(lambda x: str(x).strip() != "")]
                dupes = non_empty[non_empty["__yahoo_ticker"].duplicated(keep=False)]
                if not dupes.empty:
                    sample = ", ".join(sorted(set(dupes["__yahoo_ticker"].astype(str).head(5).tolist())))
                    raise ValueError(f"Duplicate yahoo_ticker in {name}: {sample}")
        except Exception as exc:
            self.join_key_error = str(exc)

    def _find_row(self, frame: pd.DataFrame, yahoo_ticker: str) -> Dict[str, Any]:
        if frame.empty:
            return {}
        key = str(yahoo_ticker or "").strip()
        if not key:
            return {}
        if "__yahoo_ticker" in frame.columns:
            hits = frame[frame["__yahoo_ticker"] == key]
            if len(hits) > 1:
                raise ValueError(f"Duplicate yahoo_ticker={key}")
            if len(hits) == 1:
                return hits.iloc[0].to_dict()
        if "__ticker" in frame.columns:
            hits = frame[frame["__ticker"] == key]
            if len(hits) > 1:
                raise ValueError(f"Duplicate ticker={key}")
            if len(hits) == 1:
                return hits.iloc[0].to_dict()
        return {}

    def _build_ticker_items(self) -> Dict[str, Dict[str, str]]:
        decision = self.loaded.dataframes.get("decision", pd.DataFrame())
        screen = self.loaded.dataframes.get("screen_basic", pd.DataFrame())
        valuation = self.loaded.dataframes.get("valuation", pd.DataFrame())
        priority = decision if not decision.empty else screen
        if priority.empty:
            priority = valuation

        items: Dict[str, Dict[str, str]] = {}
        if priority.empty:
            return items

        for _, row in priority.iterrows():
            yahoo_ticker = str(row.get("__yahoo_ticker", "")).strip()
            ticker = str(row.get("ticker", "")).strip() or yahoo_ticker
            company = str(row.get("company", "")).strip()
            if not yahoo_ticker:
                continue
            label = ticker
            if company:
                label = f"{ticker} | {company}"
            items[label] = {
                "yahoo_ticker": yahoo_ticker,
                "ticker": ticker,
                "company": company,
            }
        return dict(sorted(items.items(), key=lambda kv: kv[0]))

    def _decision_text_fallback(self) -> str:
        md = (self.loaded.texts.get("decision_md", "") or "").strip()
        if md:
            self.decision_sections = extract_decision_sections_from_markdown(md)
            return self.decision_sections.get("overview", "") or self.decision_sections.get("all", "") or md
        report_path = self.loaded.source_paths.get("decision_report_html")
        if report_path is not None and report_path.exists():
            return build_result_preview(report_path)
        return "IKKE FUNNET"

    def _build_meta_for_tab(self, tab_key: str, row: Mapping[str, Any]) -> DataBlockMeta:
        source_map = {
            "snapshot": ["manifest", "decision_csv", "screen_basic_csv", "decision_md", "decision_report_html"],
            "fundamentals": ["manifest", "valuation_csv", "valuation_sensitivity_csv", "decision_csv"],
            "dq": ["manifest", "decision_csv", "dq_by_ticker_csv", "dq_audit_csv", "dq_report_md"],
            "timing": ["manifest", "decision_csv", "screen_basic_csv", "prices_parquet"],
        }
        source_files: List[str] = []
        last_updated: Dict[str, str] = {}
        for source_key in source_map.get(tab_key, []):
            p = self.loaded.source_paths.get(source_key)
            if p is None:
                source_files.append(f"{source_key}:IKKE FUNNET")
                last_updated[source_key] = "IKKE FUNNET"
            else:
                source_files.append(str(p))
                last_updated[str(p)] = _path_mtime(p)

        has_decision_files = any(
            key in self.loaded.source_paths for key in ("decision_csv", "decision_md", "decision_report_html")
        )
        score, status, notes, hard_fail = calculate_trust_score(
            record=row,
            thresholds=self.thresholds,
            has_decision_files=has_decision_files,
        )
        if self.join_key_error:
            score, status, hard_fail = 0, "FAIL", True
            notes = [self.join_key_error]
        if self.loaded.error:
            score, status, hard_fail = 0, "FAIL", True
            notes = [self.loaded.error]

        coverage = {
            "candidate_data_coverage_ratio": _to_float(row.get("candidate_data_coverage_ratio")) or 0.0,
            "fresh_stock_price_coverage": _to_float(row.get("fresh_stock_price_coverage")) or 0.0,
        }
        note_txt = "; ".join(notes) if notes else ("hard_fail" if hard_fail else "ok")
        return DataBlockMeta(
            source_files=source_files,
            asof=self.loaded.asof,
            last_updated=last_updated,
            coverage=coverage,
            qc_status=status,
            notes=note_txt,
            trust_score=score,
        )

    def _update_tab_header(
        self,
        tab_key: str,
        ticker_label: str,
        company: str,
        meta: DataBlockMeta,
    ) -> None:
        vars_map = self.tab_header_vars.get(tab_key, {})
        run_dir_txt = str(self.loaded.run_dir) if self.loaded.run_dir is not None else "IKKE FUNNET"
        source_entries = []
        for src, ts in meta.last_updated.items():
            if src.endswith("IKKE FUNNET"):
                source_entries.append(src)
            else:
                source_entries.append(f"{Path(src).name} ({ts})")
        sources_txt = "kilder: " + " | ".join(source_entries)
        if "company" in vars_map:
            vars_map["company"].set(company or "IKKE FUNNET")
        if "run_meta" in vars_map:
            vars_map["run_meta"].set(
                f"ticker={ticker_label or 'IKKE FUNNET'} | asof={self.loaded.asof} | run_id={self.loaded.run_id} | run_dir={run_dir_txt}"
            )
        if "config" in vars_map:
            vars_map["config"].set(f"config_path={self.loaded.config_path or 'IKKE FUNNET'}")
        if "sources" in vars_map:
            vars_map["sources"].set(sources_txt)
        if "qc" in vars_map:
            vars_map["qc"].set(f"qc_status={meta.qc_status} | trust_score={meta.trust_score}")

        if meta.qc_status == "FAIL":
            self._set_banner(tab_key, f"IKKE BESLUTNINGSGRUNNLAG: {meta.notes}", tone="danger")
        elif meta.qc_status == "WARN":
            self._set_banner(tab_key, f"ADVARSEL: {meta.notes}", tone="warn")
        else:
            self._set_banner(tab_key, "")

    def _set_card(
        self,
        tab_key: str,
        title: str,
        value: str,
        source: str,
        meta: DataBlockMeta,
        coverage_note: str = "",
    ) -> None:
        card_var = self.tab_cards.get(tab_key, {}).get(title)
        if card_var is None:
            return
        coverage_txt = coverage_note or (
            f"candidate={_fmt_percent(meta.coverage.get('candidate_data_coverage_ratio'))}, "
            f"fresh_price={_fmt_percent(meta.coverage.get('fresh_stock_price_coverage'))}"
        )
        card_var.set(
            f"{value}\nsource: {source}\nasof: {meta.asof}\ncoverage: {coverage_txt}\nqc_status: {meta.qc_status}"
        )

    def _compute_falling_knife(self, yahoo_ticker: str) -> Tuple[str, str]:
        prices = self.loaded.dataframes.get("prices", pd.DataFrame())
        if prices.empty:
            return "IKKE FUNNET", "historikk ikke tilgjengelig"
        frame = prices.copy()
        if "yahoo_ticker" not in frame.columns or "adj_close" not in frame.columns or "date" not in frame.columns:
            return "IKKE FUNNET", "prices.parquet mangler felt"
        scope = frame[frame["yahoo_ticker"].map(str).str.strip() == str(yahoo_ticker).strip()].copy()
        if scope.empty:
            return "IKKE FUNNET", "ingen historikk for ticker"
        scope["date"] = pd.to_datetime(scope["date"], errors="coerce")
        scope = scope.dropna(subset=["date"]).sort_values("date")
        closes = pd.to_numeric(scope["adj_close"], errors="coerce").dropna()
        if len(closes) < 22:
            return "IKKE FUNNET", "for kort historikk"
        latest = float(closes.iloc[-1])
        back_21 = float(closes.iloc[-22]) if len(closes) >= 22 else latest
        ret21 = ((latest / back_21) - 1.0) * 100.0 if back_21 > 0 else 0.0
        lookback = closes.iloc[-252:] if len(closes) >= 252 else closes
        high_52w = float(lookback.max()) if len(lookback) else latest
        drawdown = ((latest / high_52w) - 1.0) * 100.0 if high_52w > 0 else 0.0
        warn = ret21 < -10.0 or drawdown < -20.0
        if warn:
            return "FALLING KNIFE", f"21d={ret21:.1f}% | dd52w={drawdown:.1f}%"
        return "OK", f"21d={ret21:.1f}% | dd52w={drawdown:.1f}%"

    def _render_snapshot_tab(self, ticker: str, company: str, row: Mapping[str, Any], meta: DataBlockMeta) -> None:
        source = "decision.csv" if not self.loaded.dataframes.get("decision", pd.DataFrame()).empty else "screen_basic.csv"
        self._set_card("snapshot", "Decision Candidate", ticker or "IKKE FUNNET", source, meta)
        self._set_card("snapshot", "Fundamental Gate", _fmt_value(_to_bool(row.get("fundamental_ok"))), source, meta)
        self._set_card("snapshot", "Technical Gate", _fmt_value(_to_bool(row.get("technical_ok"))), source, meta)
        self._set_card("snapshot", "Data Sufficiency", _fmt_value(_to_bool(row.get("candidate_data_ok"))), source, meta)
        self._set_card("snapshot", "Margin of Safety", _fmt_percent(row.get("mos")), source, meta)
        self._set_card("snapshot", "Price Freshness", _fmt_value(row.get("stock_price_age_days")), source, meta)
        self._set_card("snapshot", "Quality Score", _fmt_value(row.get("quality_score")), source, meta)
        self._set_card(
            "snapshot",
            "Risk Flags",
            str(row.get("reason_fundamental_fail", "") or row.get("reason_technical_fail", "") or "none"),
            source,
            meta,
        )

        rows = [
            {"field": "ticker", "value": ticker, "source": source, "status": meta.qc_status},
            {"field": "company", "value": company or "IKKE FUNNET", "source": source, "status": meta.qc_status},
            {"field": "fundamental_ok", "value": _fmt_value(_to_bool(row.get("fundamental_ok"))), "source": source, "status": meta.qc_status},
            {"field": "technical_ok", "value": _fmt_value(_to_bool(row.get("technical_ok"))), "source": source, "status": meta.qc_status},
            {"field": "decision_reasons", "value": str(row.get("decision_reasons", "") or "IKKE FUNNET"), "source": source, "status": meta.qc_status},
            {"field": "dq_notes", "value": meta.notes, "source": "trust", "status": meta.qc_status},
        ]
        self._populate_table("snapshot", rows)

    def _render_fundamentals_tab(self, yahoo_ticker: str, row: Mapping[str, Any], meta: DataBlockMeta) -> None:
        valuation_row = self._find_row(self.loaded.dataframes.get("valuation", pd.DataFrame()), yahoo_ticker)
        source = "valuation.csv"
        intrinsic = valuation_row.get("intrinsic_equity", row.get("intrinsic_value"))
        self._set_card("fundamentals", "Intrinsic Value", _fmt_value(intrinsic), source, meta)
        self._set_card("fundamentals", "Market Cap", _fmt_value(row.get("market_cap")), "decision.csv", meta)
        self._set_card("fundamentals", "Margin of Safety", _fmt_percent(row.get("mos")), "decision.csv", meta)
        self._set_card(
            "fundamentals",
            "WACC / COE",
            f"WACC={_fmt_percent(valuation_row.get('wacc_used') or row.get('wacc_used'))}, "
            f"COE={_fmt_percent(valuation_row.get('coe_used') or row.get('coe_used'))}",
            source,
            meta,
        )
        self._set_card("fundamentals", "ROIC-WACC Spread", _fmt_percent(row.get("roic_wacc_spread")), "decision.csv", meta)
        self._set_card("fundamentals", "Value Creation", _fmt_value(_to_bool(row.get("value_creation_ok"))), "decision.csv", meta)
        self._set_card("fundamentals", "Value QC Alerts", _fmt_value(row.get("value_qc_unresolved_alert_count")), "decision.csv", meta)

        sensitivity = self.loaded.dataframes.get("valuation_sensitivity", pd.DataFrame())
        sens_rows: List[Dict[str, Any]] = []
        if not sensitivity.empty and "__yahoo_ticker" in sensitivity.columns:
            scope = sensitivity[sensitivity["__yahoo_ticker"] == yahoo_ticker].head(50)
            for _, r in scope.iterrows():
                sens_rows.append(
                    {
                        "metric": str(r.get("scenario", "")),
                        "value": _fmt_value(r.get("intrinsic_equity")),
                        "source": "valuation_sensitivity.csv",
                        "asof": self.loaded.asof,
                    }
                )
        if sens_rows:
            self._set_card("fundamentals", "Sensitivity Coverage", f"{len(sens_rows)} scenarios", "valuation_sensitivity.csv", meta)
            self._populate_table("fundamentals", sens_rows)
        else:
            self._set_card("fundamentals", "Sensitivity Coverage", "IKKE FUNNET", "valuation_sensitivity.csv", meta)
            self._populate_table(
                "fundamentals",
                [
                    {"metric": "intrinsic_value", "value": _fmt_value(intrinsic), "source": "valuation.csv", "asof": self.loaded.asof},
                    {"metric": "fcf_used_millions", "value": _fmt_value(valuation_row.get("fcf_used_millions")), "source": "valuation.csv", "asof": self.loaded.asof},
                    {"metric": "quarterly_data_ok", "value": _fmt_value(_to_bool(valuation_row.get("quarterly_data_ok"))), "source": "valuation.csv", "asof": self.loaded.asof},
                ],
            )

    def _render_dq_tab(self, yahoo_ticker: str, row: Mapping[str, Any], meta: DataBlockMeta) -> None:
        self._set_card("dq", "Trust Score", str(meta.trust_score), "trust", meta)
        self._set_card("dq", "QC Status", meta.qc_status, "trust", meta)
        self._set_card("dq", "DQ Fail Count", _fmt_value(row.get("dq_fail_count")), "decision.csv", meta)
        self._set_card("dq", "DQ Warn Count", _fmt_value(row.get("dq_warn_count")), "decision.csv", meta)
        self._set_card("dq", "Candidate Coverage", _fmt_percent(row.get("candidate_data_coverage_ratio")), "decision.csv", meta)
        self._set_card("dq", "Fresh Price Coverage", _fmt_percent(row.get("fresh_stock_price_coverage")), "decision.csv", meta)
        self._set_card("dq", "Missing Fields", str(row.get("candidate_data_missing_fields", "") or "none"), "decision.csv", meta)
        self._set_card("dq", "Provenance Notes", meta.notes, "trust", meta)

        dq_rows: List[Dict[str, Any]] = []
        audit = self.loaded.dataframes.get("dq_audit", pd.DataFrame())
        if not audit.empty and "__yahoo_ticker" in audit.columns:
            scope = audit[audit["__yahoo_ticker"] == yahoo_ticker].head(120)
            for _, r in scope.iterrows():
                dq_rows.append(
                    {
                        "rule": str(r.get("rule_id", "")),
                        "severity": str(r.get("severity", "")),
                        "field": str(r.get("field", "")),
                        "detail": str(r.get("detail", "")),
                    }
                )
        if not dq_rows:
            dq_rows.append(
                {
                    "rule": "summary",
                    "severity": meta.qc_status,
                    "field": "notes",
                    "detail": str(row.get("dq_fail_reasons", "") or row.get("dq_warn_reasons", "") or meta.notes),
                }
            )
        self._populate_table("dq", dq_rows)

    def _render_timing_tab(self, yahoo_ticker: str, row: Mapping[str, Any], meta: DataBlockMeta) -> None:
        data_blocked = meta.qc_status == "FAIL" or _to_bool(row.get("dq_blocked")) is True
        conclusion = calculate_timing_label(
            fundamental_ok=row.get("fundamental_ok"),
            technical_ok=row.get("technical_ok"),
            data_blocked=data_blocked,
        )
        if data_blocked:
            conclusion = "IKKE BESLUTNINGSGRUNNLAG"

        self._set_card("timing", "Timing Conclusion", conclusion, "decision.csv", meta)
        self._set_card("timing", "Fundamental OK", _fmt_value(_to_bool(row.get("fundamental_ok"))), "decision.csv", meta)
        self._set_card("timing", "Technical OK", _fmt_value(_to_bool(row.get("technical_ok"))), "decision.csv", meta)
        self._set_card("timing", "Stock Above MA200", _fmt_value(_to_bool(row.get("above_ma200") or row.get("stock_ma200_ok"))), "decision.csv", meta)
        self._set_card("timing", "MAD", _fmt_value(row.get("mad")), "decision.csv", meta)
        self._set_card("timing", "Index MA200 OK", _fmt_value(_to_bool(row.get("index_ma200_ok"))), "decision.csv", meta)
        self._set_card("timing", "Price Age Days", _fmt_value(row.get("stock_price_age_days")), "decision.csv", meta)
        knife_label, knife_detail = self._compute_falling_knife(yahoo_ticker)
        self._set_card("timing", "Falling Knife", f"{knife_label} ({knife_detail})", "prices.parquet", meta)

        max_price_age_days = _to_float(self.thresholds.get("max_price_age_days")) or 7.0
        mad_min = _to_float(row.get("mad_min"))
        if mad_min is None:
            mad_min = _to_float(self.thresholds.get("mad_min"))
        rows = [
            {"signal": "conclusion", "value": conclusion, "threshold": "-", "status": meta.qc_status},
            {
                "signal": "max_price_age_days",
                "value": _fmt_value(row.get("stock_price_age_days")),
                "threshold": f"<= {max_price_age_days:.1f}",
                "status": "OK" if (_to_float(row.get("stock_price_age_days")) or 0.0) <= max_price_age_days else "FAIL",
            },
            {
                "signal": "mad_min",
                "value": _fmt_value(row.get("mad")),
                "threshold": f">= {_fmt_value(mad_min)}",
                "status": _fmt_value(_to_bool(row.get("stock_mad_ok"))),
            },
            {"signal": "index_ma200_ok", "value": _fmt_value(_to_bool(row.get("index_ma200_ok"))), "threshold": "True", "status": _fmt_value(_to_bool(row.get("index_ma200_ok")))},
            {"signal": "index_data_ok", "value": _fmt_value(_to_bool(row.get("index_data_ok"))), "threshold": "True", "status": _fmt_value(_to_bool(row.get("index_data_ok")))},
            {"signal": "stock_above_ma200", "value": _fmt_value(_to_bool(row.get("above_ma200") or row.get("stock_ma200_ok"))), "threshold": "True", "status": _fmt_value(_to_bool(row.get("above_ma200") or row.get("stock_ma200_ok")))},
            {"signal": "falling_knife", "value": knife_label, "threshold": "21d>-10% and dd52w>-20%", "status": knife_detail},
        ]
        self._populate_table("timing", rows)

    def _get_model_row_for_yahoo_ticker(self, yahoo_ticker: str) -> Dict[str, Any]:
        decision = self.loaded.dataframes.get("decision", pd.DataFrame())
        screen = self.loaded.dataframes.get("screen_basic", pd.DataFrame())
        row: Dict[str, Any] = {}
        try:
            row = self._find_row(decision, yahoo_ticker) if yahoo_ticker else {}
            if not row:
                row = self._find_row(screen, yahoo_ticker) if yahoo_ticker else {}
        except Exception as exc:
            self.join_key_error = str(exc)
            row = {}
        return row

    def _find_ticker_label_by_query(self, query: str) -> Tuple[str | None, int]:
        q = str(query or "").strip().lower()
        if not q:
            return None, 0

        exact: List[str] = []
        prefix: List[str] = []
        contains: List[str] = []
        for label, profile in self.ticker_items.items():
            ticker = str(profile.get("ticker", "")).strip().lower()
            yahoo_ticker = str(profile.get("yahoo_ticker", "")).strip().lower()
            company = str(profile.get("company", "")).strip().lower()
            label_norm = str(label).strip().lower()
            yahoo_base = yahoo_ticker.split(".", 1)[0]

            if q in {ticker, yahoo_ticker, yahoo_base, company, label_norm}:
                exact.append(label)
                continue
            if ticker.startswith(q) or yahoo_ticker.startswith(q) or yahoo_base.startswith(q) or company.startswith(q):
                prefix.append(label)
                continue
            if q in label_norm or q in company:
                contains.append(label)

        bucket = exact or prefix or contains
        if not bucket:
            return None, 0
        return bucket[0], len(bucket)

    def _build_ticker_feedback(self, profile: Mapping[str, Any], row: Mapping[str, Any], meta: DataBlockMeta) -> str:
        ticker = str(profile.get("ticker", "")).strip() or str(profile.get("yahoo_ticker", "")).strip() or "IKKE FUNNET"
        company = str(profile.get("company", "")).strip() or "IKKE FUNNET"
        if not row:
            return (
                f"{ticker} ({company}): Ingen rad i decision.csv/screen_basic.csv for aktiv run "
                f"({self.loaded.run_id})."
            )

        data_blocked = meta.qc_status == "FAIL" or _to_bool(row.get("dq_blocked")) is True
        timing = calculate_timing_label(
            fundamental_ok=row.get("fundamental_ok"),
            technical_ok=row.get("technical_ok"),
            data_blocked=data_blocked,
        )
        if data_blocked:
            timing = "IKKE BESLUTNINGSGRUNNLAG"
        reasons = str(
            row.get("decision_reasons", "")
            or row.get("reason_fundamental_fail", "")
            or row.get("reason_technical_fail", "")
            or "none"
        ).strip()
        return (
            f"{ticker} ({company}): fundamental={_fmt_value(_to_bool(row.get('fundamental_ok')))}, "
            f"technical={_fmt_value(_to_bool(row.get('technical_ok')))}, timing={timing}, "
            f"qc={meta.qc_status}, trust={meta.trust_score}, mos={_fmt_percent(row.get('mos'))}, reasons={reasons}"
        )

    def _evaluate_ticker_query(self) -> None:
        query = self.ticker_query_var.get().strip()
        if not query:
            self.ticker_feedback_var.set("Skriv en ticker først.")
            self._set_status("Ticker mangler", tone="warn")
            return

        if not self.ticker_items:
            self._refresh_results()
        label, match_count = self._find_ticker_label_by_query(query)
        if label is None:
            self.ticker_feedback_var.set(
                f"Ingen treff for '{query}' i aktiv run ({self.loaded.run_id})."
            )
            self._set_status(f"Ingen ticker-treff for {query}", tone="warn")
            self._select_results_tab("snapshot")
            return

        self.selected_ticker_var.set(label)
        self._select_results_tab("snapshot")
        profile = self.ticker_items.get(label, {"yahoo_ticker": "", "ticker": "", "company": ""})
        row = self._get_model_row_for_yahoo_ticker(str(profile.get("yahoo_ticker", "")))
        meta = self._build_meta_for_tab("snapshot", row)
        feedback = self._build_ticker_feedback(profile, row, meta)
        if match_count > 1:
            feedback = f"{feedback} (flere treff, viser beste match: {label})"
        self.ticker_feedback_var.set(feedback)
        status_tone = "ok" if meta.qc_status == "PASS" else ("warn" if meta.qc_status == "WARN" else "error")
        self._set_status(
            f"Ticker vurdert: {profile.get('ticker') or profile.get('yahoo_ticker') or label}",
            tone=status_tone,
        )

    def _render_results_for_ticker(self) -> None:
        if not self.tab_cards:
            return
        selected_label = self.selected_ticker_var.get().strip()
        profile = self.ticker_items.get(selected_label)
        if profile is None and self.ticker_items:
            first = next(iter(self.ticker_items.keys()))
            if selected_label != first:
                self.selected_ticker_var.set(first)
                return
            profile = self.ticker_items.get(first)
        if profile is None:
            profile = {"yahoo_ticker": "", "ticker": "", "company": ""}

        ticker = profile.get("ticker", "")
        yahoo_ticker = profile.get("yahoo_ticker", "")
        company = profile.get("company", "")
        self.active_company_var.set(company or "IKKE FUNNET")

        row = self._get_model_row_for_yahoo_ticker(yahoo_ticker)

        fallback_text = self._decision_text_fallback()
        if self.preview_box is not None:
            self._set_text_widget_text(self.preview_box, self.last_preview_text or fallback_text)

        for tab_key in ("snapshot", "fundamentals", "dq", "timing"):
            meta = self._build_meta_for_tab(tab_key, row)
            self._update_tab_header(tab_key, ticker, company, meta)
            self.active_trust_var.set(str(meta.trust_score))
            self.active_qc_var.set(meta.qc_status)

        self._render_snapshot_tab(ticker, company, row, self._build_meta_for_tab("snapshot", row))
        self._render_fundamentals_tab(yahoo_ticker, row, self._build_meta_for_tab("fundamentals", row))
        self._render_dq_tab(yahoo_ticker, row, self._build_meta_for_tab("dq", row))
        self._render_timing_tab(yahoo_ticker, row, self._build_meta_for_tab("timing", row))

    def _pick_latest(self, finder) -> Path | None:
        latest: Path | None = None
        for root in self._result_search_roots():
            try:
                candidate = finder(root)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
                latest = candidate
        return latest

    def _pick_latest_model_run_dir(self) -> Path | None:
        latest_run: Path | None = None
        latest_ts = -1.0
        for root in self._result_search_roots():
            candidate = find_latest_model_run_dir(root)
            if candidate is None:
                continue
            marker = candidate / "decision.csv"
            try:
                ts = float(marker.stat().st_mtime if marker.exists() else candidate.stat().st_mtime)
            except Exception:
                continue
            if ts > latest_ts:
                latest_ts = ts
                latest_run = candidate
        return latest_run

    def _find_preferred_result_path_for_run(self, run_dir: Path) -> Path | None:
        preferred_names = (
            "decision_report.html",
            "decision.md",
            "top_candidates.md",
            "result.html",
            "result.txt",
            "data_qc_report.html",
            "data_quality_report.md",
            "wft_data_quality_report.md",
        )
        for name in preferred_names:
            candidate = run_dir / name
            if candidate.exists() and candidate.is_file():
                return candidate

        try:
            run_dir_resolved = run_dir.resolve()
        except Exception:
            run_dir_resolved = run_dir

        same_run: List[Path] = []
        for path in self.result_choices.values():
            try:
                parent = path.resolve().parent
            except Exception:
                parent = path.parent
            if parent == run_dir_resolved:
                same_run.append(path)
        if not same_run:
            return None
        same_run.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return same_run[0]

    def _select_results_tab(self, tab_key: str) -> None:
        if self.result_notebook is None:
            return
        tab_order = {"snapshot": 0, "fundamentals": 1, "dq": 2, "timing": 3}
        idx = tab_order.get(tab_key, 0)
        try:
            self._select_results_main_tab()
            self.result_notebook.select(idx)
        except Exception:
            pass

    def _show_latest_model_run_in_gui(self, silent: bool = False, refresh: bool = True) -> bool:
        latest_run = self._pick_latest_model_run_dir()
        if latest_run is None:
            if not silent:
                self._show_error("Could not find latest model run (decision.csv) in selected run-dir or runs/.")
            return False
        if refresh:
            self._refresh_results()
        preferred_path = self._find_preferred_result_path_for_run(latest_run)
        if preferred_path is not None and self._select_result_path(preferred_path):
            self._select_results_tab("snapshot")
            self._set_status(f"Previewing latest model run: {latest_run}", tone="ok")
            return True
        self._apply_loaded_run(latest_run)
        self._render_results_for_ticker()
        self._select_results_tab("snapshot")
        self._set_status(f"Loaded latest model run: {latest_run}", tone="ok")
        return True

    def _show_latest_decision_in_gui(self, silent: bool = False, refresh: bool = True) -> None:
        latest = self._pick_latest(find_latest_decision_artifact)
        if latest is None:
            if self._show_latest_model_run_in_gui(silent=True, refresh=refresh):
                return
            if not silent:
                self._show_error("Could not find decision_report.html/decision.md in selected run-dir or runs/.")
            return
        if refresh:
            self._refresh_results()
        self._select_result_path(latest)
        self._select_results_tab("snapshot")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _show_latest_test_report_in_gui(self, silent: bool = False) -> None:
        latest = self._pick_latest(find_latest_test_report)
        if latest is None:
            if not silent:
                self._show_error("Could not find test report in selected run-dir or runs/.")
            return
        self._refresh_results()
        self._select_result_path(latest)
        self._select_results_tab("snapshot")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _show_latest_dq_report_in_gui(self, silent: bool = False) -> None:
        latest = self._pick_latest(find_latest_data_quality_report)
        if latest is None:
            if not silent:
                self._show_error("Could not find data quality report in selected run-dir or runs/.")
            return
        self._refresh_results()
        self._select_result_path(latest)
        self._select_results_tab("dq")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _show_latest_result_in_gui(self, silent: bool = False) -> None:
        latest = self._pick_latest(find_latest_result_file)
        if latest is None:
            if not silent:
                self._show_error("Could not find prior result file.")
            return
        self._refresh_results()
        self._select_result_path(latest)
        self._select_results_tab("snapshot")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _refresh_results(self) -> None:
        roots = self._result_search_roots()
        filter_txt = self.result_filter_var.get().strip().lower()

        hits: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in list_recent_result_files(root, limit=120):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    hits.append(path)

        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if filter_txt:
            hits = [p for p in hits if filter_txt in str(p).lower() or filter_txt in p.name.lower()]
        hits = hits[:120]

        self.result_choices = {}
        labels: List[str] = []
        for path in hits:
            label = format_result_label(path, Path.cwd())
            self.result_choices[label] = path
            labels.append(label)
        self.result_stats_var.set(f"{len(labels)} files")

        if self.results_combo is not None:
            self.results_combo["values"] = labels
        if labels:
            current = self.result_var.get().strip()
            if current not in self.result_choices:
                self.result_var.set(labels[0])
            self._preview_selected_result()
        else:
            self.result_var.set("")
            self.preview_path_var.set("No result selected")
            self._set_preview_text("No prior result files found.")

        selected_path = self.result_choices.get(self.result_var.get().strip())
        run_dir = resolve_active_run_dir(selected_path.parent if selected_path is not None else self.run_dir_var.get().strip())
        self._apply_loaded_run(run_dir)

    def _apply_loaded_run(self, run_dir: Path | None) -> None:
        self.loaded = self._load_run_artifacts(run_dir)
        self.active_run_dir_var.set(str(self.loaded.run_dir) if self.loaded.run_dir is not None else "IKKE FUNNET")
        self.active_run_id_var.set(self.loaded.run_id)
        self.active_asof_var.set(self.loaded.asof)
        self.active_config_path_var.set(self.loaded.config_path)
        self.thresholds = load_decision_thresholds(self.loaded.config_path or self.config_var.get())
        self.ticker_feedback_var.set("Skriv ticker og trykk 'Vurder ticker'.")

        self.ticker_items = self._build_ticker_items()
        ticker_labels = list(self.ticker_items.keys())
        for combo in self.ticker_selectors:
            combo["values"] = ticker_labels
        if ticker_labels:
            current_ticker = self.selected_ticker_var.get().strip()
            if current_ticker not in self.ticker_items:
                self.selected_ticker_var.set(ticker_labels[0])
            else:
                self._render_results_for_ticker()
        else:
            self.selected_ticker_var.set("")
            self._render_results_for_ticker()

    def _set_preview_text(self, text: str) -> None:
        self.last_preview_text = str(text or "")
        if self.preview_box is not None:
            self._set_text_widget_text(self.preview_box, self.last_preview_text)

    def _preview_selected_result(self) -> None:
        label = self.result_var.get().strip()
        path = self.result_choices.get(label)
        if path is None:
            self._set_preview_text("No selected result file.")
            self.preview_path_var.set("No result selected")
            return
        self.preview_path_var.set(str(path))
        self._set_preview_text(build_result_preview(path))
        if path.name.lower() == "decision.md":
            self.loaded.texts["decision_md"] = self._read_text_safe(path)
        self._render_results_for_ticker()

    def _select_result_path(self, path: Path) -> bool:
        try:
            target = path.resolve()
        except Exception:
            target = path
        for label, candidate in self.result_choices.items():
            try:
                cand_resolved = candidate.resolve()
            except Exception:
                cand_resolved = candidate
            if cand_resolved == target:
                self.result_var.set(label)
                self._preview_selected_result()
                return True
        return False

    def _append_log(self, text: str) -> None:
        lines = text.splitlines(True)
        if not lines:
            return
        self.log_box.configure(state="normal")
        for line in lines:
            lo = line.lower()
            tag = "log_info"
            if line.startswith("$ "):
                tag = "log_cmd"
            elif "[error]" in lo or " fatal:" in lo or "traceback" in lo or " failed" in lo:
                tag = "log_error"
            elif "warning" in lo or " warn" in lo:
                tag = "log_warn"
            elif "step ok" in lo or "pipeline ok" in lo or " passed" in lo or " model finished" in lo:
                tag = "log_ok"
            self.log_box.insert(tk.END, line, tag)
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")

    def _poll_logs(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(msg, dict) and msg.get("type") == "done":
                self._set_busy(False)
                task = str(msg.get("task", ""))
                rc = int(msg.get("rc", 1))
                if task == "tests":
                    report_path_raw = str(msg.get("report_path", "")).strip()
                    if report_path_raw:
                        self.last_test_report = Path(report_path_raw)
                    status = "Tests finished" if rc == 0 else "Tests failed"
                    self._set_status(status, tone=("ok" if rc == 0 else "error"))
                    self._refresh_results()
                    if self.last_test_report and self.last_test_report.exists():
                        self._select_result_path(self.last_test_report)
                        self._select_results_main_tab()
                elif task == "model":
                    self._set_status("Model finished" if rc == 0 else "Model failed", tone=("ok" if rc == 0 else "error"))
                    self._refresh_results()
                    if rc == 0:
                        self._show_latest_model_run_in_gui(silent=True)
                else:
                    self._set_status("Finished" if rc == 0 else "Failed", tone=("ok" if rc == 0 else "error"))
                    self._refresh_results()
                    if rc == 0:
                        self._show_latest_model_run_in_gui(silent=True)
            else:
                self._append_log(str(msg))
        self.root.after(120, self._poll_logs)

    def _show_error(self, text: str) -> None:
        self._set_status("Error", tone="error")
        if messagebox is not None:
            messagebox.showerror("Deecon GUI", text)
        self._append_log(f"[error] {text}\n")


def main() -> int:
    if RUN_WEEKLY_SUBPROCESS_FLAG in sys.argv:
        from src.run_weekly import main as run_weekly_main

        args = [a for a in sys.argv[1:] if a != RUN_WEEKLY_SUBPROCESS_FLAG]
        orig_argv = sys.argv[:]
        try:
            sys.argv = [orig_argv[0], *args]
            return int(run_weekly_main())
        finally:
            sys.argv = orig_argv

    ensure_runtime_working_directory()

    if tk is None:
        print("Tkinter is not available in this Python environment.", file=sys.stderr)
        return 1
    root = tk.Tk()
    DeeconGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
