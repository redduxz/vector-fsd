#Requires -Version 5.1
<#
.SYNOPSIS
  Configure + build the fsd_cpp C++ extension (cpp/ -> cmake -> fsd_cpp.*.pyd).

.DESCRIPTION
  Runs the two-step cmake build for the hot-path C++ port:

      cmake -B cpp/build -S cpp -DFSD_PYBIND=ON
      cmake --build cpp/build --config Release

  then reports the produced fsd_cpp binaries. With -InstallToRepo the built
  extension is copied next to the fsd package so `import fsd_cpp` works
  without touching PYTHONPATH (fsd.compat.backend also auto-discovers
  cpp/build, so copying is optional).

.EXAMPLE
  .\build_cpp.ps1                      # configure + Release build
  .\build_cpp.ps1 -Clean               # wipe cpp/build first
  .\build_cpp.ps1 -Config Debug        # debug build
  .\build_cpp.ps1 -InstallToRepo       # copy fsd_cpp.*.pyd to repo root
#>
[CmdletBinding()]
param(
    [string]$BuildDir = "cpp\build",
    [string]$Config = "Release",
    [string]$Generator = "",
    [string]$Python = "",
    [switch]$Clean,
    [switch]$InstallToRepo
)

$ErrorActionPreference = "Stop"

$repoRoot  = Split-Path -Parent $PSScriptRoot
$srcDir    = Join-Path $repoRoot "cpp"
$buildPath = Join-Path $repoRoot $BuildDir

Write-Host "=== NVIDIA-fsd : fsd_cpp build ==="
Write-Host "Repo root   : $repoRoot"
Write-Host "Build dir   : $buildPath"
Write-Host "Config      : $Config"

# --------------------------------------------------------------------------
# Precondition checks
# --------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $srcDir "CMakeLists.txt"))) {
    throw ("cpp\CMakeLists.txt not found under $repoRoot - the C++ port " +
           "tree has not landed yet (parallel development).")
}

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($null -eq $cmake) {
    throw ("cmake not found on PATH. Install cmake >= 3.18 or run from a " +
           "Visual Studio Developer Prompt.")
}
$cmakePath = $cmake.Source
$cmakeVer  = (& $cmakePath --version | Select-Object -First 1)
Write-Host "cmake       : $cmakePath ($cmakeVer)"

# Give cmake a concrete Python so pybind11 binds the right interpreter.
if (-not $Python) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { $Python = $py.Source }
}

# --------------------------------------------------------------------------
# Clean / configure / build
# --------------------------------------------------------------------------
if ($Clean -and (Test-Path $buildPath)) {
    Write-Host "Cleaning $buildPath ..."
    Remove-Item -Recurse -Force $buildPath
}

$configureArgs = @("-B", $buildPath, "-S", $srcDir, "-DFSD_PYBIND=ON")
if ($Generator) { $configureArgs += @("-G", $Generator) }
if ($Python)    { $configureArgs += "-DPython_EXECUTABLE=$Python" }

Write-Host ""
Write-Host "--- configure ---"
Write-Host "cmake $($configureArgs -join ' ')"
& $cmakePath @configureArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "--- build ($Config) ---"
& $cmakePath --build $buildPath --config $Config
if ($LASTEXITCODE -ne 0) { throw "cmake build failed (exit $LASTEXITCODE)" }

# --------------------------------------------------------------------------
# Artifact report
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "--- artifacts ---"
$artifacts = Get-ChildItem -Path $buildPath -Recurse -File `
    -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "fsd_cpp*" -and
                   $_.Extension -in (".pyd", ".so", ".dll", ".lib", ".exp") }

if (-not $artifacts) {
    Write-Warning ("no fsd_cpp* artifacts found under $buildPath - check " +
                   "the cmake target name in cpp\CMakeLists.txt.")
} else {
    foreach ($a in $artifacts) {
        $kb = [math]::Round($a.Length / 1KB, 1)
        Write-Host ("  {0}  ({1} KB)" -f $a.FullName, $kb)
    }
}

$extension = $artifacts | Where-Object { $_.Extension -in (".pyd", ".so") } |
             Select-Object -First 1

if ($extension -and $InstallToRepo) {
    $dest = Join-Path $repoRoot $extension.Name
    Copy-Item -Force $extension.FullName $dest
    Write-Host ""
    Write-Host "Installed: $dest"
}

Write-Host ""
Write-Host "=== Next steps ==="
if ($extension) {
    Write-Host "  Benchmark:  python scripts\bench_cpp.py"
    Write-Host "  Parity:     python -m unittest tests.test_cpp_parity -v"
    Write-Host "  (fsd.compat auto-discovers cpp\build - or re-run with -InstallToRepo)"
} else {
    Write-Host "  Fix the build above, then re-run .\build_cpp.ps1"
}
