from __future__ import annotations

import datetime as _dt
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, List

from src.run_weekly import DEFAULT_STEPS, OPTIONAL_STEPS

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


def open_in_default_app(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


class DeeconGui:
    def __init__(self, root: "tk.Tk") -> None:
        self.root = root
        self.root.title("Deecon Pipeline GUI")
        self.root.geometry("980x680")
        self.proc: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.asof_var = tk.StringVar(value=_dt.date.today().isoformat())
        self.config_var = tk.StringVar(value=r"config\config.yaml")
        self.run_dir_var = tk.StringVar(value="")
        self.dry_run_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()
        self._poll_logs()

    def _build_layout(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Asof (YYYY-MM-DD)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.asof_var, width=20).grid(row=0, column=1, sticky="w")

        ttk.Label(frm, text="Config").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.config_var, width=70).grid(row=1, column=1, sticky="we", pady=(8, 0))
        ttk.Button(frm, text="Browse", command=self._pick_config).grid(row=1, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Run dir (optional)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.run_dir_var, width=70).grid(row=2, column=1, sticky="we", pady=(8, 0))
        ttk.Button(frm, text="Browse", command=self._pick_run_dir).grid(row=2, column=2, padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(frm, text="Dry run", variable=self.dry_run_var).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frm, text="Steps").grid(row=4, column=0, sticky="nw", pady=(10, 0))
        self.steps_list = tk.Listbox(frm, selectmode="multiple", exportselection=False, height=11)
        self.steps_list.grid(row=4, column=1, sticky="we", pady=(10, 0))
        for name, _ in DEFAULT_STEPS:
            self.steps_list.insert(tk.END, name)
        for name, _ in OPTIONAL_STEPS:
            self.steps_list.insert(tk.END, f"{name} (optional)")
        self.steps_list.select_set(0, len(DEFAULT_STEPS) - 1)

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=1, sticky="w", pady=(10, 0))
        self.run_btn = ttk.Button(btns, text="Run pipeline", command=self._run_pipeline)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop_pipeline, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Open latest decision.md", command=self._open_latest_decision).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Clear log", command=self._clear_log).pack(side="left", padx=(8, 0))

        ttk.Label(frm, textvariable=self.status_var).grid(row=6, column=1, sticky="w", pady=(8, 0))

        self.log_box = tk.Text(frm, wrap="word", height=20)
        self.log_box.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

        yscroll = ttk.Scrollbar(frm, orient="vertical", command=self.log_box.yview)
        yscroll.grid(row=7, column=3, sticky="ns", pady=(10, 0))
        self.log_box["yscrollcommand"] = yscroll.set

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(7, weight=1)

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
            self.status_var.set("Pipeline already running")
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
        self.status_var.set("Running...")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def worker() -> None:
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
            self.log_queue.put("__PROCESS_DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def _stop_pipeline(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.status_var.set("Stopping...")

    def _clear_log(self) -> None:
        self.log_box.delete("1.0", tk.END)

    def _open_latest_decision(self) -> None:
        selected_run_dir = self.run_dir_var.get().strip()
        if selected_run_dir:
            direct = Path(selected_run_dir) / "decision.md"
            if direct.exists():
                open_in_default_app(direct)
                self.status_var.set(f"Opened: {direct}")
                return

        latest = find_latest_decision_md(Path("runs"))
        if latest is None:
            self._show_error("Could not find decision.md in selected run-dir or runs/.")
            return
        open_in_default_app(latest)
        self.status_var.set(f"Opened: {latest}")

    def _append_log(self, text: str) -> None:
        self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)

    def _poll_logs(self) -> None:
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if msg == "__PROCESS_DONE__":
                self.run_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.status_var.set("Finished")
            else:
                self._append_log(msg)
        self.root.after(120, self._poll_logs)

    def _show_error(self, text: str) -> None:
        self.status_var.set("Error")
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
