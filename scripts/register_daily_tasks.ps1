param(
    [string]$TaskNameMorning = "Deecon_Daily_Morning",
    [string]$TaskNameEvening = "Deecon_Daily_Evening",
    [string]$MorningTime = "06:30",
    [string]$EveningTime = "18:10",
    [string]$ProjectRoot = "",
    [string]$PythonExe = "python",
    [switch]$RunWeeklyDecisionInEvening,
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

function Build-Action {
    param(
        [string]$Root,
        [string]$Python,
        [bool]$RunWeekly,
        [bool]$NoDownload
    )
    $runner = Join-Path $Root "scripts/run_daily_automation.ps1"
    $cmd = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -ProjectRoot `"$Root`" -PythonExe `"$Python`""
    if ($RunWeekly) { $cmd += " -RunWeeklyDecision" }
    if ($NoDownload) { $cmd += " -NoDownloadReports" }
    return $cmd
}

function Register-Or-UpdateTask {
    param(
    [Parameter(Mandatory = $true)][string]$TaskName,
    [Parameter(Mandatory = $true)][string]$Time,
    [Parameter(Mandatory = $true)][string]$ActionCmd
    )
    $cmd = "Register-ScheduledTask -TaskName `"$TaskName`" -Action powershell.exe `"$ActionCmd`" -Trigger Daily@$Time"
    Write-Host ""
    Write-Host ">> $cmd" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host "DryRun: skipped." -ForegroundColor Yellow
        return
    }
    $trigger = New-ScheduledTaskTrigger -Daily -At $Time
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionCmd
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Deecon daily automation task" -Force | Out-Null
    Write-Host ("OK registered task: {0}" -f $TaskName) -ForegroundColor Green
}

$root = Resolve-ProjectRoot -ProjectRootArg $ProjectRoot
$morningAction = Build-Action -Root $root -Python $PythonExe -RunWeekly:$false -NoDownload:$NoDownloadReports
$eveningAction = Build-Action -Root $root -Python $PythonExe -RunWeekly:$RunWeeklyDecisionInEvening -NoDownload:$NoDownloadReports

Write-Host "Task registration start" -ForegroundColor Magenta
Write-Host ("ProjectRoot={0}" -f $root)
Write-Host ("Morning={0} at {1}" -f $TaskNameMorning, $MorningTime)
Write-Host ("Evening={0} at {1}" -f $TaskNameEvening, $EveningTime)

Register-Or-UpdateTask -TaskName $TaskNameMorning -Time $MorningTime -ActionCmd $morningAction
Register-Or-UpdateTask -TaskName $TaskNameEvening -Time $EveningTime -ActionCmd $eveningAction

Write-Host "Done." -ForegroundColor Green
