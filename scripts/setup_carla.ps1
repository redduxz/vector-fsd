#Requires -Version 5.1
<#
.SYNOPSIS
  Downloads and unpacks the CARLA simulator (Windows release) for NVIDIA-fsd.

.DESCRIPTION
  Fetches the CARLA release zip from the carla-simulator GitHub releases page,
  extracts it under -InstallDir, locates CarlaUE4.exe, and optionally sets the
  CARLA_ROOT user environment variable so scripts/run_sim.py can find it.

.EXAMPLE
  .\setup_carla.ps1
  .\setup_carla.ps1 -Version 0.9.15 -InstallDir D:\CARLA -SetCarlaRootEnv
  .\setup_carla.ps1 -SkipDownload   # re-extract an already-downloaded zip
#>
[CmdletBinding()]
param(
    [string]$Version = "0.9.15",
    [string]$InstallDir = "$env:USERPROFILE\CARLA",
    [switch]$SkipDownload,
    [switch]$SetCarlaRootEnv
)

$ErrorActionPreference = "Stop"

$zipName   = "CARLA_$Version.zip"
$url       = "https://github.com/carla-simulator/carla/releases/download/$Version/$zipName"
$zipPath   = Join-Path $InstallDir $zipName
$extractTo = Join-Path $InstallDir "CARLA_$Version"

Write-Host "=== NVIDIA-fsd : CARLA $Version setup ==="
Write-Host "Install dir : $InstallDir"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if ($SkipDownload) {
    Write-Host "Skipping download (using $zipPath)"
    if (-not (Test-Path $zipPath)) {
        throw "-SkipDownload was given but $zipPath does not exist."
    }
} elseif (Test-Path $zipPath) {
    Write-Host "Archive already downloaded: $zipPath"
} else {
    Write-Host "Downloading:"
    Write-Host "  $url"
    Write-Host "(the package is ~2.5 GB - this can take a while)"
    # The default progress bar slows Invoke-WebRequest massively; disable it.
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    Write-Host "Download complete: $zipPath"
}

if (Test-Path $extractTo) {
    Write-Host "Already extracted: $extractTo"
} else {
    Write-Host "Extracting to $extractTo ..."
    Expand-Archive -Path $zipPath -DestinationPath $extractTo -Force
    Write-Host "Extraction complete."
}

# The release zip nests differently across versions (WindowsNoEditor/ etc.),
# so locate the simulator binary rather than assuming a fixed layout.
$exe = Get-ChildItem -Path $extractTo -Filter "CarlaUE4.exe" -Recurse `
       -ErrorAction SilentlyContinue | Select-Object -First 1

if ($null -eq $exe) {
    Write-Warning "CarlaUE4.exe not found under $extractTo - check the archive layout."
    $carlaRoot = $extractTo
} else {
    $carlaRoot = $exe.DirectoryName
    Write-Host "Simulator binary: $($exe.FullName)"
}

if ($SetCarlaRootEnv) {
    [Environment]::SetEnvironmentVariable("CARLA_ROOT", $carlaRoot, "User")
    $env:CARLA_ROOT = $carlaRoot
    Write-Host "Set user env var CARLA_ROOT=$carlaRoot"
} else {
    Write-Host "Tip: re-run with -SetCarlaRootEnv to persist CARLA_ROOT=$carlaRoot"
}

Write-Host ""
Write-Host "=== Next steps ==="
Write-Host "  1. Start the server:   `"$carlaRoot\CarlaUE4.exe`""
Write-Host "  2. Install deps:       pip install -r requirements.txt"
Write-Host "  3. Run the autopilot:  python -m fsd.agents.autopilot --config configs/default.yaml"
Write-Host "     or one step:        python scripts/run_sim.py --config configs/default.yaml --launch-server"
