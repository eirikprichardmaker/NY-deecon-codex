param(
    [string]$TaskPrefix = "Deecon-MaxDownload",
    [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $RepoRoot "scripts\resume_download_runner.ps1"
if (-not (Test-Path $Runner)) {
    throw "Missing runner script: $Runner"
}

$taskWake = "$TaskPrefix-OnWake"
$taskLogon = "$TaskPrefix-OnLogon"

$escapedRunner = $Runner.Replace('"', '""')
# schtasks expects doubled quotes around quoted path inside /TR payload
$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"`"$escapedRunner`"`""

# Wake trigger: Microsoft-Windows-Power-Troubleshooter, Event ID 1
$eventQuery = "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]"

$commands = @(
    "schtasks /Create /F /TN ""$taskWake"" /SC ONEVENT /EC System /MO ""$eventQuery"" /TR ""$tr""",
    "schtasks /Create /F /TN ""$taskLogon"" /SC ONLOGON /TR ""$tr"""
)

Write-Host "RepoRoot: $RepoRoot"
Write-Host "Runner : $Runner"
Write-Host "TaskWake: $taskWake"
Write-Host "TaskLogon: $taskLogon"
Write-Host ""

foreach ($cmd in $commands) {
    if ($WhatIf) {
        Write-Host "[WhatIf] $cmd"
        continue
    }
    Write-Host ">> $cmd" -ForegroundColor Cyan
    cmd.exe /c $cmd
}

if (-not $WhatIf) {
    Write-Host ""
    Write-Host "Installed scheduled tasks." -ForegroundColor Green
    Write-Host "List tasks:"
    Write-Host "  schtasks /Query /TN ""$taskWake"" /V /FO LIST"
    Write-Host "  schtasks /Query /TN ""$taskLogon"" /V /FO LIST"
    Write-Host ""
    Write-Host "To remove later:"
    Write-Host "  schtasks /Delete /F /TN ""$taskWake"""
    Write-Host "  schtasks /Delete /F /TN ""$taskLogon"""
}
