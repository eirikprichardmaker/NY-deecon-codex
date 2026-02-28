param(
    [string]$AsOf = (Get-Date -Format "yyyy-MM-dd"),
    [string]$ProjectRoot = "",
    [string]$PythonExe = "python",
    [string]$IdsCsv = "config/tickers_with_insid_clean.csv",
    [string]$IdsCol = "ins_id",
    [string]$OutRoot = "data/freeze/borsdata_proplus_freeze"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-ProjectRoot {
    param([string]$ProjectRootArg)
    if ($ProjectRootArg -and $ProjectRootArg.Trim().Length -gt 0) {
        return (Resolve-Path $ProjectRootArg).Path
    }
    $scriptDir = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

$root = Resolve-ProjectRoot -ProjectRootArg $ProjectRoot
Set-Location $root

# Ensure BORSDATA_AUTHKEY is present when running from Task Scheduler.
if (-not $env:BORSDATA_AUTHKEY -or $env:BORSDATA_AUTHKEY.Trim().Length -eq 0) {
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        $line = Get-Content $envFile | Where-Object { $_ -match '^BORSDATA_AUTHKEY=' } | Select-Object -First 1
        if ($line) {
            $env:BORSDATA_AUTHKEY = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'")
        }
    }
}

$cmd = "$PythonExe tools/borsdata_freeze.py --asof $AsOf --ids-csv $IdsCsv --ids-col $IdsCol --out-root $OutRoot --only holdings --holdings --pace-s 0.05"
Write-Host ">> $cmd" -ForegroundColor Cyan
Invoke-Expression $cmd
$exitCode = $LASTEXITCODE
if ($null -ne $exitCode -and $exitCode -ne 0) {
    throw ("Holdings snapshot failed with exit code {0}" -f $exitCode)
}
Write-Host "OK holdings snapshot." -ForegroundColor Green
