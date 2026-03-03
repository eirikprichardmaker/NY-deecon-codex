from __future__ import annotations

import datetime as _dt
import html
import queue
import re
import subprocess
import sys
import threading
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.run_weekly import DEFAULT_STEPS, OPTIONAL_STEPS

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


class DeeconGui:
    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.root.title("Deecon Pipeline GUI")
        self.root.geometry("1240x820")
        self.root.minsize(1080, 720)
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
        self.result_page_var = tk.StringVar(value="overview")
        self.result_page_buttons: Dict[str, "ttk.Button"] = {}
        self.result_pages: Dict[str, "ttk.Frame"] = {}
        self.section_path_vars: Dict[str, "tk.StringVar"] = {}
        self.section_text_widgets: Dict[str, "tk.Text"] = {}
        self.decision_sections: Dict[str, str] = {}
        self.decision_section_var = tk.StringVar(value="overview")
        self.decision_section_buttons: Dict[str, "ttk.Button"] = {}

        self._configure_styles()
        self._build_layout()
        self._wire_reactive_updates()
        self._set_status("Ready", tone="ok")
        self._poll_logs()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        available = set(style.theme_names())
        for theme in ("vista", "xpnative", "clam", "alt", "default"):
            if theme in available:
                style.theme_use(theme)
                break

        self.root.option_add("*Font", "{Segoe UI} 10")
        self.root.configure(background="#f2f5fa")

        style.configure("App.TFrame", background="#f2f5fa")
        style.configure("Hero.TFrame", background="#ffffff", relief="flat")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Section.TLabelframe", background="#ffffff", bordercolor="#d8e2f0", borderwidth=1, relief="solid")
        style.configure("Section.TLabelframe.Label", background="#ffffff", foreground="#1f2a3d", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 18, "bold"))
        style.configure("SubHeader.TLabel", background="#ffffff", foreground="#51627a")
        style.configure("Info.TLabel", background="#ffffff", foreground="#334155")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b")
        style.configure("Chip.TLabel", background="#eef2ff", foreground="#334155", padding=(8, 3), font=("Segoe UI", 9, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Secondary.TButton", font=("Segoe UI", 10))
        style.configure("Nav.TButton", font=("Segoe UI", 10, "bold"), anchor="w")
        style.configure("NavActive.TButton", font=("Segoe UI", 10, "bold"), anchor="w")
        style.map("NavActive.TButton", background=[("!disabled", "#dbeafe")], foreground=[("!disabled", "#1d4ed8")])
        style.configure("StatusNeutral.TLabel", background="#e7edf7", foreground="#1f2937", padding=(10, 4), font=("Segoe UI", 10, "bold"))
        style.configure("StatusRun.TLabel", background="#dbeafe", foreground="#1d4ed8", padding=(10, 4), font=("Segoe UI", 10, "bold"))
        style.configure("StatusOk.TLabel", background="#dcfce7", foreground="#166534", padding=(10, 4), font=("Segoe UI", 10, "bold"))
        style.configure("StatusWarn.TLabel", background="#fef3c7", foreground="#92400e", padding=(10, 4), font=("Segoe UI", 10, "bold"))
        style.configure("StatusBad.TLabel", background="#fee2e2", foreground="#991b1b", padding=(10, 4), font=("Segoe UI", 10, "bold"))
        style.configure("Accent.Horizontal.TProgressbar", troughcolor="#e5e7eb", bordercolor="#e5e7eb", background="#2563eb", lightcolor="#2563eb", darkcolor="#2563eb")
        style.configure("TNotebook", background="#f2f5fa", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])

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
        if hasattr(self, "steps_list"):
            self.steps_list.bind("<<ListboxSelect>>", lambda _e: self._update_command_preview())
        self.root.bind("<F5>", lambda _e: self._run_model())
        self.root.bind("<Control-r>", lambda _e: self._run_pipeline())
        self.root.bind("<Control-t>", lambda _e: self._run_tests())
        self._update_command_preview()

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
        self.tabs.add(run_tab, text="Kjoring")
        self.tabs.add(results_tab, text="Resultater")

        self._build_run_tab(run_tab)

        self._build_results_tab(results_tab)
        self._refresh_results()

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
            font=("Segoe UI", 10),
        )
        self.steps_list.pack(fill="both", expand=True, pady=(6, 0))
        for name, _ in DEFAULT_STEPS:
            self.steps_list.insert(tk.END, name)
        for name, _ in OPTIONAL_STEPS:
            self.steps_list.insert(tk.END, f"{name} (optional)")
        self.steps_list.select_set(0, len(DEFAULT_STEPS) - 1)

        cmd_box = ttk.LabelFrame(control_col, text="Command Preview", padding=10, style="Section.TLabelframe")
        cmd_box.pack(fill="x", pady=(10, 0))
        ttk.Label(cmd_box, textvariable=self.command_preview_var, style="Muted.TLabel", wraplength=420, justify="left").pack(anchor="w")

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
        ttk.Button(actions_box, text="Show latest decision", style="Secondary.TButton", command=self._show_latest_decision_in_gui).grid(row=2, column=0, sticky="we", pady=(8, 0))
        ttk.Button(actions_box, text="Show latest test report", style="Secondary.TButton", command=self._show_latest_test_report_in_gui).grid(row=2, column=1, sticky="we", padx=(8, 0), pady=(8, 0))
        ttk.Button(actions_box, text="Clear log", style="Secondary.TButton", command=self._clear_log).grid(row=3, column=0, columnspan=2, sticky="we", pady=(8, 0))
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
            font=("Consolas", 10),
        )
        self.log_box.tag_configure("log_cmd", foreground="#93c5fd")
        self.log_box.tag_configure("log_info", foreground="#cbd5e1")
        self.log_box.tag_configure("log_warn", foreground="#fbbf24")
        self.log_box.tag_configure("log_error", foreground="#f87171")
        self.log_box.tag_configure("log_ok", foreground="#4ade80")
        self.log_box.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(log_box_wrap, orient="vertical", command=self.log_box.yview)
        yscroll.pack(side="right", fill="y")
        self.log_box["yscrollcommand"] = yscroll.set

    def _switch_results_page(self, key: str) -> None:
        key_norm = str(key).strip().lower()
        if key_norm not in self.result_pages:
            key_norm = "overview"
        for page_key, frame in self.result_pages.items():
            if page_key == key_norm:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        for btn_key, btn in self.result_page_buttons.items():
            btn.configure(style="NavActive.TButton" if btn_key == key_norm else "Nav.TButton")
        self.result_page_var.set(key_norm)

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

    def _build_section_preview_widget(self, parent: "ttk.Frame", key: str) -> "tk.Text":
        path_var = tk.StringVar(value="No result selected")
        self.section_path_vars[key] = path_var
        ttk.Label(parent, textvariable=path_var, style="Muted.TLabel").pack(anchor="w")
        txt = tk.Text(
            parent,
            wrap="word",
            height=26,
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            borderwidth=0,
            font=("Consolas", 10),
        )
        txt.pack(side="left", fill="both", expand=True, pady=(8, 0))
        txt.tag_configure("flag_red", foreground="#f87171")
        txt.tag_configure("flag_green", foreground="#4ade80")
        txt.configure(state="disabled")
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=txt.yview)
        yscroll.pack(side="right", fill="y", pady=(8, 0))
        txt["yscrollcommand"] = yscroll.set
        self.section_text_widgets[key] = txt
        return txt

    def _set_section_preview(self, key: str, path: Path) -> None:
        key_norm = str(key).strip().lower()
        txt = self.section_text_widgets.get(key_norm)
        var = self.section_path_vars.get(key_norm)
        if txt is None or var is None:
            return
        var.set(str(path))
        self._set_text_widget_text(txt, build_result_preview(path))

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
        try:
            if src.exists():
                raw = src.read_text(encoding="utf-8")
            else:
                raw = ""
        except UnicodeDecodeError:
            raw = src.read_text(encoding="latin-1", errors="replace") if src.exists() else ""
        except Exception:
            raw = ""

        if src.suffix.lower() == ".md" and raw.strip():
            sections = extract_decision_sections_from_markdown(raw)
            self.decision_sections = sections
            text = sections.get(self.decision_section_var.get(), "") or sections.get("overview", "") or sections.get("all", "")
            if not text:
                text = raw
            var = self.section_path_vars.get("decision")
            txt_widget = self.section_text_widgets.get("decision")
            if var is not None:
                var.set(f"{src} | {self.decision_section_var.get()}")
            if txt_widget is not None:
                self._set_text_widget_text(txt_widget, text)
        else:
            self.decision_sections = {"all": build_result_preview(artifact_path)}
            var = self.section_path_vars.get("decision")
            txt_widget = self.section_text_widgets.get("decision")
            if var is not None:
                var.set(str(artifact_path))
            if txt_widget is not None:
                self._set_text_widget_text(txt_widget, self.decision_sections["all"])

        self._show_decision_subsection(self.decision_section_var.get(), silent=True)

    def _show_decision_subsection(self, key: str, silent: bool = False) -> None:
        key_norm = str(key).strip().lower()
        if key_norm not in {"overview", "fundamental", "stock", "products", "media", "quality", "schema", "all"}:
            key_norm = "overview"
        self.decision_section_var.set(key_norm)
        for btn_key, btn in self.decision_section_buttons.items():
            btn.configure(style="NavActive.TButton" if btn_key == key_norm else "Nav.TButton")

        txt_widget = self.section_text_widgets.get("decision")
        var = self.section_path_vars.get("decision")
        if txt_widget is None or var is None:
            return

        if not self.decision_sections:
            if not silent:
                self._set_text_widget_text(txt_widget, "Ingen decision-data lastet enda.")
            return

        text = self.decision_sections.get(key_norm, "").strip()
        if not text:
            fallback = self.decision_sections.get("overview", "").strip() or self.decision_sections.get("all", "").strip()
            text = fallback or "(Tom seksjon)"

        current_path = var.get().split(" | ")[0].strip() if var.get().strip() else "decision"
        var.set(f"{current_path} | section={key_norm}")
        self._set_text_widget_text(txt_widget, text)

    def _build_results_tab(self, tab: "ttk.Frame") -> None:
        splitter = ttk.Panedwindow(tab, orient="horizontal")
        splitter.pack(fill="both", expand=True)

        nav_col = ttk.Frame(splitter, padding=6, style="App.TFrame")
        content_col = ttk.Frame(splitter, padding=6, style="App.TFrame")
        splitter.add(nav_col, weight=2)
        splitter.add(content_col, weight=8)

        nav_box = ttk.LabelFrame(nav_col, text="Sider", padding=10, style="Section.TLabelframe")
        nav_box.pack(fill="x")
        nav_items = [
            ("overview", "Oversikt"),
            ("decision", "Decision"),
            ("tests", "Test Reports"),
            ("dq", "Data Quality"),
        ]
        for key, label in nav_items:
            btn = ttk.Button(nav_box, text=label, style="Nav.TButton", command=lambda k=key: self._switch_results_page(k))
            btn.pack(fill="x", pady=(0, 6))
            self.result_page_buttons[key] = btn

        quick_box = ttk.LabelFrame(nav_col, text="Quick Actions", padding=10, style="Section.TLabelframe")
        quick_box.pack(fill="x", pady=(10, 0))
        ttk.Button(quick_box, text="Kjor model", style="Primary.TButton", command=self._run_model).pack(fill="x")
        ttk.Button(quick_box, text="Kjor pipeline", style="Primary.TButton", command=self._run_pipeline).pack(fill="x", pady=(6, 0))
        ttk.Button(quick_box, text="Kjor test", style="Secondary.TButton", command=self._run_tests).pack(fill="x", pady=(6, 0))

        pages_root = ttk.Frame(content_col, style="Card.TFrame")
        pages_root.pack(fill="both", expand=True)

        # Overview page
        overview = ttk.Frame(pages_root, style="Card.TFrame", padding=8)
        self.result_pages["overview"] = overview
        ttk.Label(overview, text="Resultatoversikt", style="Info.TLabel").pack(anchor="w")
        filter_row = ttk.Frame(overview, style="Card.TFrame")
        filter_row.pack(fill="x", pady=(8, 0))
        ttk.Label(filter_row, text="Filter:", style="Muted.TLabel").pack(side="left")
        ttk.Entry(filter_row, textvariable=self.result_filter_var, width=24).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Label(filter_row, textvariable=self.result_stats_var, style="Muted.TLabel").pack(side="left", padx=(8, 0))
        self.results_combo = ttk.Combobox(overview, textvariable=self.result_var, state="readonly")
        self.results_combo.pack(fill="x", pady=(8, 0))
        self.results_combo.bind("<<ComboboxSelected>>", lambda _e: self._preview_selected_result())
        overview_btns = ttk.Frame(overview, style="Card.TFrame")
        overview_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(overview_btns, text="Refresh", style="Secondary.TButton", command=self._refresh_results).pack(side="left")
        ttk.Button(overview_btns, text="Preview", style="Secondary.TButton", command=self._preview_selected_result).pack(side="left", padx=(8, 0))
        ttk.Button(overview_btns, text="Latest result", style="Secondary.TButton", command=self._show_latest_result_in_gui).pack(side="left", padx=(8, 0))
        ttk.Button(overview_btns, text="Latest decision", style="Secondary.TButton", command=self._show_latest_decision_in_gui).pack(side="left", padx=(8, 0))
        ttk.Button(overview_btns, text="Latest test", style="Secondary.TButton", command=self._show_latest_test_report_in_gui).pack(side="left", padx=(8, 0))
        ttk.Button(overview_btns, text="Latest DQ", style="Secondary.TButton", command=self._show_latest_dq_report_in_gui).pack(side="left", padx=(8, 0))
        preview_wrap = ttk.LabelFrame(overview, text="Preview", padding=10, style="Section.TLabelframe")
        preview_wrap.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(preview_wrap, textvariable=self.preview_path_var, style="Muted.TLabel").pack(anchor="w")
        self.preview_box = self._build_section_preview_widget(preview_wrap, key="overview")

        # Decision page
        decision = ttk.Frame(pages_root, style="Card.TFrame", padding=8)
        self.result_pages["decision"] = decision
        ttk.Label(decision, text="Decision Side", style="Info.TLabel").pack(anchor="w")
        ttk.Label(decision, text="Egne undersider for decision-seksjoner.", style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        decision_btns = ttk.Frame(decision, style="Card.TFrame")
        decision_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(decision_btns, text="Latest decision", style="Secondary.TButton", command=self._show_latest_decision_in_gui).pack(side="left")
        ttk.Button(decision_btns, text="Refresh list", style="Secondary.TButton", command=self._refresh_results).pack(side="left", padx=(8, 0))
        subsection_row = ttk.Frame(decision, style="Card.TFrame")
        subsection_row.pack(fill="x", pady=(8, 0))
        subsection_items = [
            ("overview", "Oversikt"),
            ("fundamental", "Fundamental"),
            ("stock", "Aksje"),
            ("products", "Produkter"),
            ("media", "Media"),
            ("quality", "Vurdering"),
            ("schema", "Skjema"),
        ]
        for key, label in subsection_items:
            btn = ttk.Button(subsection_row, text=label, style="Nav.TButton", command=lambda k=key: self._show_decision_subsection(k))
            btn.pack(side="left", padx=(0, 6))
            self.decision_section_buttons[key] = btn
        decision_wrap = ttk.LabelFrame(decision, text="Decision Preview", padding=10, style="Section.TLabelframe")
        decision_wrap.pack(fill="both", expand=True, pady=(10, 0))
        self._build_section_preview_widget(decision_wrap, key="decision")
        self._show_decision_subsection("overview", silent=True)

        # Tests page
        tests = ttk.Frame(pages_root, style="Card.TFrame", padding=8)
        self.result_pages["tests"] = tests
        ttk.Label(tests, text="Test Report Side", style="Info.TLabel").pack(anchor="w")
        ttk.Label(tests, text="Viser nyeste test report fra GUI-kjoring.", style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        tests_btns = ttk.Frame(tests, style="Card.TFrame")
        tests_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(tests_btns, text="Latest test report", style="Secondary.TButton", command=self._show_latest_test_report_in_gui).pack(side="left")
        ttk.Button(tests_btns, text="Refresh list", style="Secondary.TButton", command=self._refresh_results).pack(side="left", padx=(8, 0))
        tests_wrap = ttk.LabelFrame(tests, text="Test Preview", padding=10, style="Section.TLabelframe")
        tests_wrap.pack(fill="both", expand=True, pady=(10, 0))
        self._build_section_preview_widget(tests_wrap, key="tests")

        # Data quality page
        dq = ttk.Frame(pages_root, style="Card.TFrame", padding=8)
        self.result_pages["dq"] = dq
        ttk.Label(dq, text="Data Quality Side", style="Info.TLabel").pack(anchor="w")
        ttk.Label(dq, text="Viser nyeste data quality-rapport.", style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
        dq_btns = ttk.Frame(dq, style="Card.TFrame")
        dq_btns.pack(fill="x", pady=(8, 0))
        ttk.Button(dq_btns, text="Latest DQ report", style="Secondary.TButton", command=self._show_latest_dq_report_in_gui).pack(side="left")
        ttk.Button(dq_btns, text="Refresh list", style="Secondary.TButton", command=self._refresh_results).pack(side="left", padx=(8, 0))
        dq_wrap = ttk.LabelFrame(dq, text="Data Quality Preview", padding=10, style="Section.TLabelframe")
        dq_wrap.pack(fill="both", expand=True, pady=(10, 0))
        self._build_section_preview_widget(dq_wrap, key="dq")

        self._switch_results_page("overview")

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
        self.log_box.delete("1.0", tk.END)

    def _result_search_roots(self) -> List[Path]:
        selected_run_dir = self.run_dir_var.get().strip()
        roots: List[Path] = []
        if selected_run_dir:
            roots.append(Path(selected_run_dir))
        roots.append(Path("runs"))
        return roots

    def _show_latest_decision_in_gui(self, silent: bool = False) -> None:
        latest: Path | None = None
        for root in self._result_search_roots():
            candidate = find_latest_decision_artifact(root)
            if candidate is not None:
                if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
                    latest = candidate
        if latest is None:
            if not silent:
                self._show_error("Could not find decision_report.html/decision.md in selected run-dir or runs/.")
            return
        self._refresh_results()
        found = self._select_result_path(latest)
        if not found:
            self.preview_path_var.set(str(latest))
            self._set_preview_text(build_result_preview(latest))
        self._load_decision_sections(latest)
        try:
            self.tabs.select(1)
        except Exception:
            pass
        self._switch_results_page("decision")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _show_latest_test_report_in_gui(self, silent: bool = False) -> None:
        latest: Path | None = None
        for root in self._result_search_roots():
            candidate = find_latest_test_report(root)
            if candidate is not None:
                if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
                    latest = candidate
        if latest is None:
            if not silent:
                self._show_error("Could not find test report in selected run-dir or runs/.")
            return
        self._refresh_results()
        found = self._select_result_path(latest)
        if not found:
            self.preview_path_var.set(str(latest))
            self._set_preview_text(build_result_preview(latest))
        self._set_section_preview("tests", latest)
        try:
            self.tabs.select(1)
        except Exception:
            pass
        self._switch_results_page("tests")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _show_latest_dq_report_in_gui(self, silent: bool = False) -> None:
        latest: Path | None = None
        for root in self._result_search_roots():
            candidate = find_latest_data_quality_report(root)
            if candidate is not None:
                if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
                    latest = candidate
        if latest is None:
            if not silent:
                self._show_error("Could not find data quality report in selected run-dir or runs/.")
            return
        self._refresh_results()
        found = self._select_result_path(latest)
        if not found:
            self.preview_path_var.set(str(latest))
            self._set_preview_text(build_result_preview(latest))
        self._set_section_preview("dq", latest)
        try:
            self.tabs.select(1)
        except Exception:
            pass
        self._switch_results_page("dq")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _show_latest_result_in_gui(self, silent: bool = False) -> None:
        latest: Path | None = None
        for root in self._result_search_roots():
            candidate = find_latest_result_file(root)
            if candidate is not None:
                if latest is None or candidate.stat().st_mtime > latest.stat().st_mtime:
                    latest = candidate
        if latest is None:
            if not silent:
                self._show_error("Could not find prior result file.")
            return
        self._refresh_results()
        found = self._select_result_path(latest)
        if not found:
            self.preview_path_var.set(str(latest))
            self._set_preview_text(build_result_preview(latest))
        try:
            self.tabs.select(1)
        except Exception:
            pass
        self._switch_results_page("overview")
        self._set_status(f"Previewing: {latest}", tone="ok")

    def _refresh_results(self) -> None:
        roots = self._result_search_roots()
        filter_txt = self.result_filter_var.get().strip().lower()

        hits: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            for path in list_recent_result_files(root, limit=80):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    hits.append(path)

        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if filter_txt:
            hits = [p for p in hits if filter_txt in str(p).lower() or filter_txt in p.name.lower()]
        hits = hits[:80]

        self.result_choices = {}
        labels: List[str] = []
        for path in hits:
            label = format_result_label(path, Path.cwd())
            self.result_choices[label] = path
            labels.append(label)
        self.result_stats_var.set(f"{len(labels)} files")

        self.results_combo["values"] = labels
        if labels:
            current = self.result_var.get().strip()
            if current not in self.result_choices:
                self.result_var.set(labels[0])
            self._preview_selected_result()
        else:
            self.result_var.set("")
            self._set_preview_text("No prior result files found.")
            self.preview_path_var.set("No result selected")

    def _set_preview_text(self, text: str) -> None:
        self._set_text_widget_text(self.preview_box, text)

    def _preview_selected_result(self) -> None:
        label = self.result_var.get().strip()
        path = self.result_choices.get(label)
        if path is None:
            self._set_preview_text("No selected result file.")
            self.preview_path_var.set("No result selected")
            return
        self.preview_path_var.set(str(path))
        self._set_preview_text(build_result_preview(path))
        self._set_section_preview("overview", path)
        name = path.name.lower()
        if name in {"decision.md", "decision_report.html"}:
            self._load_decision_sections(path)

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
                try:
                    self.tabs.select(1)
                except Exception:
                    pass
                return True
        return False

    def _append_log(self, text: str) -> None:
        lines = text.splitlines(True)
        if not lines:
            return
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
                        try:
                            self.tabs.select(1)
                        except Exception:
                            pass
                elif task == "model":
                    self._set_status("Model finished" if rc == 0 else "Model failed", tone=("ok" if rc == 0 else "error"))
                    self._refresh_results()
                    if rc == 0:
                        self._show_latest_decision_in_gui(silent=True)
                else:
                    self._set_status("Finished" if rc == 0 else "Failed", tone=("ok" if rc == 0 else "error"))
                    self._refresh_results()
                    if rc == 0:
                        self._show_latest_decision_in_gui(silent=True)
            else:
                self._append_log(str(msg))
        self.root.after(120, self._poll_logs)

    def _show_error(self, text: str) -> None:
        self._set_status("Error", tone="error")
        if messagebox is not None:
            messagebox.showerror("Deecon GUI", text)
        self._append_log(f"[error] {text}\n")


def main() -> int:
    if tk is None:
        print("Tkinter is not available in this Python environment.", file=sys.stderr)
        return 1
    root = tk.Tk()
    DeeconGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
