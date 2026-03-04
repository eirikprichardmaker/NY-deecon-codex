[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv-build-gui",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$venvPath = Join-Path $repoRoot $VenvDir
if ($Clean -and (Test-Path $venvPath)) {
    Remove-Item -Recurse -Force $venvPath
}
if (-not (Test-Path $venvPath)) {
    & $Python -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Could not find venv python at $venvPython"
}

& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install pyinstaller

$distPath = Join-Path $repoRoot "dist"
$workPath = Join-Path $repoRoot "build\pyinstaller"
$specPath = $workPath
$configPath = Join-Path $repoRoot "config"
$configsPath = Join-Path $repoRoot "configs"

if ($Clean) {
    if (Test-Path $distPath) {
        Remove-Item -Recurse -Force $distPath
    }
    if (Test-Path $workPath) {
        Remove-Item -Recurse -Force $workPath
    }
}

$pyinstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", "DeeconControlCenter",
    "--distpath", $distPath,
    "--workpath", $workPath,
    "--specpath", $specPath,
    "--paths", $repoRoot,
    "--collect-submodules", "src",
    "--collect-submodules", "bs4",
    "--collect-data", "bs4",
    "--hidden-import", "pytest",
    "--add-data", "$configPath;config",
    "--add-data", "$configsPath;configs",
    "src\gui_windows_entry.py"
)

& $venvPython -m PyInstaller @pyinstallerArgs

$outDir = Join-Path $distPath "DeeconControlCenter"
$outExe = Join-Path $outDir "DeeconControlCenter.exe"
if (-not (Test-Path $outExe)) {
    throw "Build failed: missing $outExe"
}

if (Test-Path (Join-Path $repoRoot "README_run_windows.md")) {
    Copy-Item -Force (Join-Path $repoRoot "README_run_windows.md") (Join-Path $outDir "README_run_windows.md")
}

Write-Host "Build completed:"
Write-Host "  $outExe"
