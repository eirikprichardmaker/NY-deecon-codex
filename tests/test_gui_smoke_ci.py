import os

import pytest


def test_qt_gui_smoke_init():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from src.gui_qt import DeeconGui, QtWidgets
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PySide6/gui import unavailable: {exc}")

    if QtWidgets is None:
        pytest.skip("PySide6 is not available in this environment")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    win = DeeconGui()
    app.processEvents()
    win.close()
