param(
    [string]$TaskName = "Deecon_Holdings_Daily",
    [string]$RunAt = "19:20",
    [string]$ProjectRoot = "",
    [string]$PythonExe = "python",
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

$root = Resolve-ProjectRoot -ProjectRootArg $ProjectRoot
$runner = Join-Path $root "scripts\run_holdings_snapshot.ps1"
$args = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -ProjectRoot `"$root`" -PythonExe `"$PythonExe`""

Write-Host "Task registration start" -ForegroundColor Magenta
Write-Host ("ProjectRoot={0}" -f $root)
Write-Host ("TaskName={0} at {1}" -f $TaskName, $RunAt)
Write-Host ("Action=powershell.exe {0}" -f $args)

if ($DryRun) {
    Write-Host "DryRun: skipped." -ForegroundColor Yellow
    exit 0
}

$trigger = New-ScheduledTaskTrigger -Daily -At $RunAt
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $args
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Daily holdings snapshot (Borsdata --holdings)." -Force | Out-Null

Write-Host ("OK registered task: {0}" -f $TaskName) -ForegroundColor Green
