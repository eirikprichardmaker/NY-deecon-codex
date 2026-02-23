param(
    [ValidateRange(1, 7)]
    [int]$Day = 1,
    [switch]$RunAll,
    [string]$ArchiveAsOf = "2026-02-18",
    [string]$IdsCsv = "config/tickers_with_insid.csv",
    [string]$IdsCol = "ins_id",
    [string]$OutRoot = "data/freeze/borsdata",
    [string]$DataDir = "data",
    [string]$MarketSet = "NO,SE,DK,FI",
    [switch]$IncludeDelisted,
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-LoggedCommand {
    param([Parameter(Mandatory = $true)][string]$Command)
    Write-Host ""
    Write-Host ">> $Command" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-Expression $Command
    $exitCode = $LASTEXITCODE
    $sw.Stop()
    if ($null -ne $exitCode -and $exitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $exitCode, $Command)
    }
    Write-Host ("OK ({0:n1}s)" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
}

function Get-DailyAsOf {
    param([int]$OffsetDays)
    return (Get-Date).Date.AddDays($OffsetDays).ToString("yyyy-MM-dd")
}

function Get-LatestRawAsOf {
    $rawRoot = Join-Path $DataDir "raw"
    if (-not (Test-Path $rawRoot)) {
        return $null
    }
    $candidates = Get-ChildItem $rawRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' } |
        Sort-Object Name
    if (-not $candidates -or $candidates.Count -eq 0) {
        return $null
    }
    return $candidates[-1].Name
}

function Run-Day1 {
    $asof = Get-DailyAsOf 0
    $incDel = if ($IncludeDelisted) { "true" } else { "false" }
    Invoke-LoggedCommand "python -m src.freeze_golden_fundamentals_history --asof $asof --datasets prices --markets $MarketSet --include-delisted $incDel --skip-existing true --refetch-invalid-cache true"
    Invoke-LoggedCommand "python -m src.freeze_golden_fundamentals_history --asof $asof --datasets reports_y,reports_q,reports_r12 --markets $MarketSet --include-delisted $incDel --skip-existing true --refetch-invalid-cache true"
    Invoke-LoggedCommand "python -m src.freeze_golden_fundamentals_history --asof $asof --datasets kpis,dividends,splits --markets $MarketSet --include-delisted $incDel --skip-existing true --refetch-invalid-cache true"
    Invoke-LoggedCommand "python -m src.local_data_method --asof $asof --data-dir $DataDir --out-dir runs/$asof"
}

function Run-KpiBackfill {
    Invoke-LoggedCommand "python tools/borsdata_freeze.py --asof $ArchiveAsOf --ids-csv $IdsCsv --ids-col $IdsCol --out-root $OutRoot --only kpi --kpi-ids 37,10,42,135,57,63,50,49,60,55,54 --kpi-period year --kpi-value-type mean"
    Invoke-LoggedCommand "python tools/borsdata_freeze.py --asof $ArchiveAsOf --ids-csv $IdsCsv --ids-col $IdsCol --out-root $OutRoot --only kpi --kpi-ids 37,10,42,135,57,63,50,49,60,55,54 --kpi-period r12 --kpi-value-type mean"
    # Quarter only for KPIs known to support quarter in this project
    Invoke-LoggedCommand "python tools/borsdata_freeze.py --asof $ArchiveAsOf --ids-csv $IdsCsv --ids-col $IdsCol --out-root $OutRoot --only kpi --kpi-ids 135,63,55,54 --kpi-period quarter --kpi-value-type mean"
}

function Run-Day2 {
    Run-KpiBackfill
}

function Run-Day3 {
    $asof = Get-DailyAsOf 2
    Invoke-LoggedCommand "python tools/borsdata_freeze.py --asof $asof --ids-csv $IdsCsv --ids-col $IdsCol --out-root $OutRoot --only reports,reports_pi,prices_last,holdings --reports-per-instrument --reports-pi-maxcount 20"
}

function Run-Day4 {
    $asof = Get-DailyAsOf 3
    $incDel = if ($IncludeDelisted) { "true" } else { "false" }
    Invoke-LoggedCommand "python -m src.freeze_golden_fundamentals_history --asof $asof --datasets prices,reports_y,reports_q,reports_r12,kpis,dividends,splits --markets $MarketSet --include-delisted $incDel --skip-existing true --refetch-invalid-cache true"
    Invoke-LoggedCommand "python -m src.local_data_method --asof $asof --data-dir $DataDir --out-dir runs/$asof"
}

function Run-Day5 {
    Run-KpiBackfill
}

function Run-Day6 {
    $asof = Get-DailyAsOf 5
    Invoke-LoggedCommand "python tools/borsdata_freeze.py --asof $asof --ids-csv $IdsCsv --ids-col $IdsCol --out-root $OutRoot --only reports,reports_pi,prices_last,holdings --reports-per-instrument --reports-pi-maxcount 20"
}

function Run-Day7 {
    $asof = Get-DailyAsOf 6
    $targetRaw = Join-Path (Join-Path $DataDir "raw") $asof
    if (-not (Test-Path $targetRaw)) {
        $fallback = Get-LatestRawAsOf
        if ($null -eq $fallback) {
            throw "Day 7: no dated snapshot found under $DataDir/raw."
        }
        Write-Host ("Day 7: snapshot {0} not found, using latest available {1}." -f $asof, $fallback) -ForegroundColor Yellow
        $asof = $fallback
    }
    Invoke-LoggedCommand "python -m src.local_data_method --asof $asof --data-dir $DataDir --out-dir runs/$asof"
    $statePath = Join-Path (Join-Path $OutRoot $ArchiveAsOf) "state.json"
    if (Test-Path $statePath) {
        Write-Host ""
        Write-Host ">> State summary: $statePath" -ForegroundColor Cyan
        $s = Get-Content $statePath -Raw | ConvertFrom-Json
        $s.done.PSObject.Properties.Name |
            Group-Object { ($_ -split '\|')[0] } |
            Sort-Object Name |
            Format-Table Name, Count -AutoSize
        Write-Host "OK (state summary)" -ForegroundColor Green
    } else {
        Write-Host ("WARN: state file not found: {0}" -f $statePath) -ForegroundColor Yellow
    }
}

function Run-Validation {
    if ($SkipTests) {
        Write-Host "Skipping tests (--SkipTests)." -ForegroundColor Yellow
        return
    }
    Invoke-LoggedCommand "python -m pytest -q"
}

function Run-DayByNumber {
    param([int]$Index)
    switch ($Index) {
        1 { Run-Day1; break }
        2 { Run-Day2; break }
        3 { Run-Day3; break }
        4 { Run-Day4; break }
        5 { Run-Day5; break }
        6 { Run-Day6; break }
        7 { Run-Day7; break }
        default { throw "Unsupported day index: $Index" }
    }
}

Write-Host ("Plan start. RunAll={0} Day={1}" -f [bool]$RunAll, $Day) -ForegroundColor Magenta
Write-Host ("ArchiveAsOf={0} MarketSet={1}" -f $ArchiveAsOf, $MarketSet) -ForegroundColor Magenta

if ($RunAll) {
    1..7 | ForEach-Object { Run-DayByNumber $_ }
} else {
    Run-DayByNumber $Day
}

Run-Validation
Write-Host "Done." -ForegroundColor Green
