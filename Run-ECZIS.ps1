# UNFILED-IP
[CmdletBinding()]
param(
    [Parameter()]
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'artifacts\demo'),

    [Parameter()]
    [int]$Seed = 20260803,

    [Parameter()]
    [ValidateSet(4, 9, 16, 25)]
    [int]$Frames = 16,

    [Parameter()]
    [ValidateRange(1, 200)]
    [int]$Iterations = 36,

    [Parameter()]
    [switch]$OpenViewer,

    [Parameter()]
    [switch]$RunStaticAnalysis
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [scriptblock]$Action
    )
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script.'
}

$IpStatus = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'IP_STATUS.yaml') -Raw
if ($IpStatus -notmatch 'status:\s*UNFILED' -or $IpStatus -notmatch 'public_disclosure_allowed:\s*false') {
    throw 'IP gate failed: expected UNFILED and public_disclosure_allowed: false.'
}

Invoke-Checked -Name 'Resolve and synchronize locked environment' -Action {
    uv sync --python 3.12
}

Invoke-Checked -Name 'ECZIS environment doctor' -Action {
    uv run eczis doctor
}

if ($RunStaticAnalysis) {
    Invoke-Checked -Name 'Ruff' -Action {
        uv run ruff check .
    }

    Invoke-Checked -Name 'Mypy strict' -Action {
        uv run mypy --strict src tests
    }
}

Invoke-Checked -Name 'Automated tests' -Action {
    uv run pytest
}

Invoke-Checked -Name 'Synthetic evidence demonstration' -Action {
    uv run eczis demo --output $OutputDirectory --seed $Seed --frames $Frames --iterations $Iterations
}

$Bundle = Join-Path $OutputDirectory 'eczis_demo.eczis'
$Key = Join-Path $OutputDirectory 'DEMO_ONLY.key'
$Viewer = Join-Path $OutputDirectory 'ECZIS_VIEWER.html'

Invoke-Checked -Name 'Independent bundle verification' -Action {
    uv run eczis verify --bundle $Bundle --key $Key
}

Write-Host "`nECZIS BUILD PASSED" -ForegroundColor Green
Write-Host "Bundle : $Bundle"
Write-Host "Viewer : $Viewer"
Write-Host "Report : $(Join-Path $OutputDirectory 'REPORT.md')"

if ($OpenViewer) {
    Start-Process -FilePath $Viewer
}
