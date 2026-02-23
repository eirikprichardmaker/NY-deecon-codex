param(
    [switch]$IncludeDelisted = $true,
    [switch]$SkipTests = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunScript = Join-Path $RepoRoot "scripts\max_download_week.ps1"
$RunsDir = Join-Path $RepoRoot "runs"
$LockPath = Join-Path $RunsDir "resume_download_runner.lock"
$LogDir = Join-Path $RunsDir "automation_logs"

New-Item -ItemType Directory -Path $RunsDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$now = Get-Date
$stamp = $now.ToString("yyyyMMdd_HHmmss")
$logPath = Join-Path $LogDir "resume_download_runner_$stamp.log"

function Write-Log {
    param([string]$Message)
    $line = ("[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Message)
    $line | Tee-Object -FilePath $logPath -Append
}

function Test-RunnerAlreadyActive {
    $selfPid = $PID
    $needle = "resume_download_runner.ps1"
    $procs = Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" |
        Where-Object { $_.ProcessId -ne $selfPid -and $_.CommandLine -and $_.CommandLine -like "*$needle*" }
    return ($procs.Count -gt 0)
}

if (Test-Path $LockPath) {
    try {
        $lockAgeMin = ((Get-Date) - (Get-Item $LockPath).LastWriteTime).TotalMinutes
    } catch {
        $lockAgeMin = 0
    }
    if ($lockAgeMin -lt 480) {
        Write-Log "Exit: lock file exists and is recent ($([math]::Round($lockAgeMin,1)) min)."
        exit 0
    }
}

if (Test-RunnerAlreadyActive) {
    Write-Log "Exit: another resume_download_runner instance is active."
    exit 0
}

Set-Content -Path $LockPath -Value ("started={0}" -f (Get-Date).ToString("o")) -Encoding UTF8

try {
    Push-Location $RepoRoot
    if (-not (Test-Path $RunScript)) {
        throw "Missing script: $RunScript"
    }

    $args = @("-RunAll")
    if ($IncludeDelisted) { $args += "-IncludeDelisted" }
    if ($SkipTests) { $args += "-SkipTests" }

    Write-Log ("Starting max_download_week.ps1 with args: {0}" -f ($args -join " "))
    & $RunScript @args *>> $logPath
    $rc = $LASTEXITCODE
    if ($null -eq $rc) { $rc = 0 }
    Write-Log "Finished with exit code: $rc"
    exit $rc
}
catch {
    Write-Log ("Runner failed: {0}" -f $_.Exception.Message)
    throw
}
finally {
    Pop-Location
    Remove-Item -Path $LockPath -Force -ErrorAction SilentlyContinue
}
