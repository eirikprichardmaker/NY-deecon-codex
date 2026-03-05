"""Thin wrapper for Qt GUI entrypoint.

This module keeps backward-compatible exports for helper functions used by tests,
while delegating all GUI runtime behavior to `src.gui_qt`.
"""

from src.gui_qt import *  # noqa: F401,F403
from src.gui_qt import main as _qt_main


def main() -> int:
    return _qt_main()


if __name__ == "__main__":
    raise SystemExit(main())
