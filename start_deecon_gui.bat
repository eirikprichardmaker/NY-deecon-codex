@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\pythonw.exe" -m src.gui
) else (
  pyw -3 -m src.gui
)
