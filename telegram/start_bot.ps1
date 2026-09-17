$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Find-Python {
    foreach ($name in @("py", "python")) {
        try {
            $command = Get-Command $name -ErrorAction Stop
            return $command.Source
        } catch {}
    }
    throw "Python was not found. Install Python 3.10+ and run this script again."
}

$Python = Find-Python
$Venv = Join-Path $Root ".venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/3] Creating Python environment..."
    if ((Split-Path -Leaf $Python) -eq "py.exe") {
        & $Python -3 -m venv $Venv
    } else {
        & $Python -m venv $Venv
    }
}

Write-Host "[2/3] Installing Telegram dependency..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "telegram\requirements.txt")

$Engine = $null
$BuildDir = Join-Path $Root "build"
if (Test-Path $BuildDir) {
    $Engine = Get-ChildItem -Path $BuildDir -Filter "viger-sr.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
}

if (-not $Engine) {
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        throw "CMake was not found. Install CMake, then run this script again."
    }

    Write-Host "[2/3] Building the native VIGER SR engine..."
    & cmake -S $Root -B $BuildDir
    & cmake --build $BuildDir --config Release --target viger-sr

    $Engine = Get-ChildItem -Path $BuildDir -Filter "viger-sr.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $Engine) {
        throw "Build finished, but viger-sr.exe was not found."
    }
}

$Env:VIGER_SR_EXE = $Engine.FullName

if (-not $Env:VIGER_TELEGRAM_TOKEN) {
    Write-Host ""
    Write-Host "Open Telegram -> @BotFather -> /newbot and paste the token below."
    $Env:VIGER_TELEGRAM_TOKEN = Read-Host "Telegram bot token"
}

if (-not $Env:VIGER_TELEGRAM_TOKEN) {
    throw "Telegram token is empty."
}

Write-Host "[3/3] Starting VIGER SR Telegram bot..."
Write-Host "Engine: $($Engine.FullName)"
Write-Host "Press Ctrl+C to stop the bot."
Write-Host ""
& $VenvPython (Join-Path $Root "telegram\bot.py")
