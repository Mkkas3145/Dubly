$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if (!(Get-Command ffmpeg -ErrorAction SilentlyContinue) -or !(Get-Command ffprobe -ErrorAction SilentlyContinue)) { throw 'FFmpeg and ffprobe are required.' }
if (!(Test-Path '.venv\Scripts\python.exe')) { uv venv .venv --python 3.11 --seed; Check-Exit }
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $python -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128; Check-Exit
& $python install_deux.py; Check-Exit
& $python install_speedy.py; Check-Exit
& $python install_mfa.py; Check-Exit
& $python install_mms.py; Check-Exit
if (!(Test-Path '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
Write-Host 'Ready. Enter OPENROUTER_API_KEY in .env, then run start.bat.'
