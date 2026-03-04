"""PyInstaller entrypoint for Windows GUI build.

This wrapper keeps relative paths stable in frozen mode and supports
`--run-module <module>` dispatch so the bundled EXE can execute pipeline/test
modules without an external Python install.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import List, Optional

from src.gui import main as _gui_main


def _set_runtime_working_dir() -> None:
    if not getattr(sys, "frozen", False):
        return
    exe_dir = Path(sys.executable).resolve().parent
    os.chdir(exe_dir)
    (exe_dir / "runs").mkdir(parents=True, exist_ok=True)


def _ensure_std_streams() -> None:
    if sys.stdout is None:
        try:
            sys.stdout = open(1, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
        except OSError:
            sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
    if sys.stderr is None:
        try:
            sys.stderr = open(2, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


def _run_module_mode(argv: List[str]) -> Optional[int]:
    if len(argv) < 3 or argv[1] != "--run-module":
        return None
    module_name = str(argv[2]).strip()
    if not module_name:
        print("Missing module name after --run-module", file=sys.stderr)
        return 2
    _ensure_std_streams()
    sys.argv = [module_name, *argv[3:]]
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return 0
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return int(code)
        return 1


def main() -> int:
    _set_runtime_working_dir()
    module_rc = _run_module_mode(sys.argv)
    if module_rc is not None:
        return int(module_rc)
    return _gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
