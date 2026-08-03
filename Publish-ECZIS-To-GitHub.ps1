# UNFILED-IP
#Requires -Version 7.2
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$Owner = "rblake2320",
    [string]$Repository = "eczis",
    [string]$Description = "Evidence-Aware Continuous Zoom Imaging System — private synthetic software proof",
    [switch]$SkipTests,
    [switch]$SkipBranchProtection
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Step([string]$Message) {
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Assert-Command([string]$Name, [string]$WingetId) {
    if (Get-Command $Name -ErrorAction SilentlyContinue) { return }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "'$Name' is required and winget is unavailable. Install $WingetId, then rerun."
    }
    Write-Step "Installing $Name"
    winget install --id $WingetId --exact --accept-package-agreements --accept-source-agreements
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "User")
    }
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Installation completed, but '$Name' is not available in PATH. Open a new PowerShell 7 window and rerun."
    }
}

function Invoke-Native([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$FullName = "$Owner/$Repository"

Write-Step "Validating private-IP safety gate"
if (-not (Test-Path "IP_STATUS.yaml")) { throw "IP_STATUS.yaml is missing." }
$IpStatus = Get-Content "IP_STATUS.yaml" -Raw
if ($IpStatus -notmatch '(?m)^status:\s*UNFILED\s*$' -or
    $IpStatus -notmatch '(?m)^public_disclosure_allowed:\s*false\s*$') {
    throw "Publication refused: IP_STATUS.yaml does not declare UNFILED and public_disclosure_allowed: false."
}

$Forbidden = @(
    Get-ChildItem -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notmatch '[\\/]\.git[\\/]' -and (
            $_.Extension -in '.key', '.eczis' -or
            $_.Name -in '.env', 'DEMO_ONLY.key' -or
            $_.FullName -match '[\\/]artifacts[\\/]'
        )
    }
)
if ($Forbidden.Count -gt 0) {
    $Names = $Forbidden.FullName -join [Environment]::NewLine
    throw "Publication refused because generated evidence or secret-like files are present:`n$Names"
}

Assert-Command -Name "git" -WingetId "Git.Git"
Assert-Command -Name "gh" -WingetId "GitHub.cli"
Assert-Command -Name "uv" -WingetId "astral-sh.uv"

Write-Step "Checking GitHub authentication"
& gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "GitHub authentication is required. A browser login will open." -ForegroundColor Yellow
    Invoke-Native "gh" @("auth", "login", "--hostname", "github.com", "--git-protocol", "https", "--web")
}
$Login = (& gh api user --jq '.login').Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Login)) { throw "Unable to resolve authenticated GitHub user." }
if ($Login -ne $Owner) {
    throw "Authenticated as '$Login', but this publication targets '$Owner'. Switch GitHub accounts and rerun."
}

Write-Step "Checking repository state"
$RepoExists = $false
& gh repo view $FullName --json name,isPrivate,defaultBranchRef 1>$null 2>$null
if ($LASTEXITCODE -eq 0) { $RepoExists = $true }
if ($RepoExists) {
    $RepoJson = & gh repo view $FullName --json isPrivate,defaultBranchRef | ConvertFrom-Json
    if (-not $RepoJson.isPrivate) { throw "Publication refused: $FullName exists but is public." }
    if ($RepoJson.defaultBranchRef) {
        throw "$FullName already contains a default branch. This bootstrap script will not overwrite an existing repository."
    }
}

Write-Step "Creating locked development environment"
Invoke-Native "uv" @("lock", "--python", "3.12")
Invoke-Native "uv" @("sync", "--all-groups", "--python", "3.12")

if (-not $SkipTests) {
    Write-Step "Running release gates"
    Invoke-Native "uv" @("run", "python", "tools/ip_gate.py")
    Invoke-Native "uv" @("run", "python", "tools/claims_check.py")
    Invoke-Native "uv" @("run", "ruff", "check", ".")
    Invoke-Native "uv" @("run", "mypy", "src", "tools")
    Invoke-Native "uv" @("run", "pytest", "--cov=eczis", "--cov-report=term-missing", "--cov-fail-under=70")
}

Write-Step "Creating local Git history"
if (-not (Test-Path ".git")) {
    Invoke-Native "git" @("init", "--initial-branch=main")
}
Invoke-Native "git" @("config", "user.name", "Ron Blake")
$UserId = (& gh api user --jq '.id').Trim()
Invoke-Native "git" @("config", "user.email", "$UserId+$Login@users.noreply.github.com")
Invoke-Native "git" @("add", "--all")
$Status = (& git status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "git status failed." }
if (-not [string]::IsNullOrWhiteSpace(($Status -join "`n"))) {
    Invoke-Native "git" @("commit", "-m", "feat: bootstrap ECZIS v0.1R1 private research repository")
}

Write-Step "Creating and pushing private GitHub repository"
if (-not $RepoExists) {
    Invoke-Native "gh" @(
        "repo", "create", $FullName,
        "--private",
        "--source", ".",
        "--remote", "origin",
        "--push",
        "--description", $Description
    )
} else {
    $Remote = "https://github.com/$FullName.git"
    & git remote get-url origin 1>$null 2>$null
    if ($LASTEXITCODE -eq 0) {
        Invoke-Native "git" @("remote", "set-url", "origin", $Remote)
    } else {
        Invoke-Native "git" @("remote", "add", "origin", $Remote)
    }
    Invoke-Native "git" @("push", "-u", "origin", "main")
}

Write-Step "Configuring repository"
Invoke-Native "gh" @("repo", "edit", $FullName, "--enable-issues", "--enable-wiki=false")
$Topics = '{"names":["computational-imaging","super-resolution","inverse-problems","provenance","computer-vision","python","research"]}'
$Topics | gh api --method PUT "repos/$FullName/topics" --input - | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to set repository topics." }

$Labels = @(
    @{ Name = "research"; Color = "5319e7"; Description = "Experiment or scientific validation work" },
    @{ Name = "needs-evidence-review"; Color = "b60205"; Description = "Could affect evidence or claim boundaries" },
    @{ Name = "hardware"; Color = "0052cc"; Description = "Physical rig or device integration" },
    @{ Name = "provenance"; Color = "1d76db"; Description = "M/R/I/U semantics or confidence" },
    @{ Name = "security"; Color = "d93f0b"; Description = "Evidence sealing, keys, or threat model" },
    @{ Name = "dependencies"; Color = "0366d6"; Description = "Dependency maintenance" }
)
foreach ($Label in $Labels) {
    & gh label create $Label.Name --repo $FullName --color $Label.Color --description $Label.Description --force | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to create label '$($Label.Name)'." }
}

Write-Step "Creating initial growth backlog"
$Issues = @(
    @{
        Title = "Execute the five-target locked benchmark matrix"
        Labels = "research,needs-evidence-review"
        Body = @"
## Goal
Run microglyph, Siemens star, coin, iris, and PCB targets through calibration, development, locked-test, model-mismatch, and adversarial splits.

## Acceptance
- Frozen seeds and configurations
- SNR-stratified metrics
- Sealed reports and manifests
- No claim expansion until the claims gate maps new tests
"@
    },
    @{
        Title = "Add DPAPI/CNG production key protection"
        Labels = "security"
        Body = @"
## Goal
Replace demo-key handling with a versioned Windows DPAPI/CNG key provider while preserving deterministic test doubles.

## Acceptance
- Keys never enter manifests or logs
- Wrong-user and tamper tests
- Documented Linux interface delta
"@
    },
    @{
        Title = "Implement coordinate-field backend with provenance preservation"
        Labels = "provenance,research"
        Body = @"
## Goal
Add the optional coordinate MLP/Fourier-feature renderer without allowing provenance promotion or evidence-label drift.

## Acceptance
- Seam and arbitrary-scale tests
- Class-preservation property tests
- CPU and RTX 5090 benchmark profiles
"@
    },
    @{
        Title = "Build the first physical CaptureRig adapter"
        Labels = "hardware,needs-evidence-review"
        Body = @"
## Goal
Implement a real camera plus XY-stage adapter against the frozen CaptureRig protocol.

## Acceptance
- Calibration and timing evidence
- Raw-frame preservation
- Real-capture re-execution of applicable T2/T3 validation
- No optical-performance claim before microscope comparison
"@
    }
)
foreach ($Issue in $Issues) {
    $Existing = & gh issue list --repo $FullName --state all --search ('"' + $Issue.Title + '" in:title') --json title --jq '.[].title'
    if ($LASTEXITCODE -ne 0) { throw "Failed to search issues." }
    if ($Existing -notcontains $Issue.Title) {
        Invoke-Native "gh" @("issue", "create", "--repo", $FullName, "--title", $Issue.Title, "--body", $Issue.Body, "--label", $Issue.Labels, "--assignee", $Owner)
    }
}

if (-not $SkipBranchProtection) {
    Write-Step "Applying main-branch protection"
    $Protection = @{
        required_status_checks = @{
            strict = $true
            contexts = @("Python 3.12 / ubuntu-latest", "Python 3.12 / windows-latest")
        }
        enforce_admins = $true
        required_pull_request_reviews = $null
        restrictions = $null
        required_conversation_resolution = $true
        allow_force_pushes = $false
        allow_deletions = $false
    } | ConvertTo-Json -Depth 8
    $Protection | gh api --method PUT "repos/$FullName/branches/main/protection" --input - 1>$null 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Branch protection was not applied. The GitHub plan or token may not permit it; configure it manually after CI names are visible."
    }
}

Write-Step "Publication complete"
$Url = (& gh repo view $FullName --json url --jq '.url').Trim()
Write-Host "Private repository: $Url" -ForegroundColor Green
Write-Host "The repository remains blocked from public disclosure by IP_STATUS.yaml and CI." -ForegroundColor Yellow
