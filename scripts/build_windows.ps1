param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$VenvPath = Join-Path $RepoRoot ".venv-windows-build"

if (-not (Test-Path $VenvPath)) {
    & $Python -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m pip install pyinstaller

& $VenvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name DeeconControlCenter `
    --collect-submodules src `
    --collect-submodules bs4 `
    --hidden-import tkinter `
    --add-data "config;config" `
    --add-data "configs;configs" `
    src/gui.py

$DistAppDir = Join-Path $RepoRoot "dist\DeeconControlCenter"
New-Item -ItemType Directory -Force -Path (Join-Path $DistAppDir "runs") | Out-Null
Copy-Item -Force README_run_windows.md (Join-Path $DistAppDir "README_run_windows.md")

Write-Host "Build complete: dist/DeeconControlCenter/DeeconControlCenter.exe"
