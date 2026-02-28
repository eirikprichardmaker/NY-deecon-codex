param(
    [string]$AsOf = (Get-Date -Format "yyyy-MM-dd"),
    [string]$ProjectRoot = "",
    [string]$PythonExe = "python",
    [string]$ConfigPath = "config/config.yaml",
    [int]$RequireFreshDays = 3,
    [switch]$RunWeeklyDecision,
    [string]$RunWeeklySteps = "valuation,decision",
    [switch]$NoDownloadReports,
    [switch]$DryRun
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

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Command
    )
    Write-Host ""
    Write-Host (">> {0}" -f $Name) -ForegroundColor Cyan
    Write-Host $Command
    if ($DryRun) {
        Write-Host "DryRun: skipped." -ForegroundColor Yellow
        return
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-Expression $Command
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw ("Step failed ({0}) exit={1}: {2}" -f $Name, $exitCode, $Command)
    }
    Write-Host ("OK {0} ({1:n1}s)" -f $Name, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

$root = Resolve-ProjectRoot -ProjectRootArg $ProjectRoot
Set-Location $root

$logDir = Join-Path $root ("logs/scheduler/" + (Get-Date -Format "yyyyMMdd"))
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("daily_" + (Get-Date -Format "HHmmss") + ".log")
Start-Transcript -Path $logFile -Append | Out-Null

try {
    Write-Host "Daily automation start" -ForegroundColor Magenta
    Write-Host ("ProjectRoot={0}" -f $root)
    Write-Host ("AsOf={0}" -f $AsOf)
    Write-Host ("LogFile={0}" -f $logFile)

    $refreshCmd = "$PythonExe -m src.refresh_prices_daily --asof $AsOf --config $ConfigPath --require-fresh-days $RequireFreshDays"
    Invoke-Step -Name "Refresh prices daily" -Command $refreshCmd

    $watchCmd = "$PythonExe -m src.report_watch --sources-csv config/report_watch_sources.csv --db data/processed/report_watch/report_watch.db --downloads-dir data/raw/ir_auto"
    if ($NoDownloadReports) {
        $watchCmd += " --no-download"
    }
    Invoke-Step -Name "Watch quarterly reports" -Command $watchCmd

    if ($RunWeeklyDecision) {
        $weeklyCmd = "$PythonExe -m src.run_weekly --asof $AsOf --config $ConfigPath --steps $RunWeeklySteps"
        Invoke-Step -Name "Run weekly model steps" -Command $weeklyCmd
    } else {
        Write-Host ""
        Write-Host ">> Run weekly model steps" -ForegroundColor Cyan
        Write-Host "Skipped (use -RunWeeklyDecision to enable)." -ForegroundColor Yellow
    }

    Write-Host "Daily automation completed." -ForegroundColor Green
    exit 0
}
catch {
    Write-Host ("ERROR: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}

