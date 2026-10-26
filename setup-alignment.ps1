$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
$basePython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path '.align-venv\Scripts\python.exe')) {
    & $basePython -m venv --system-site-packages .align-venv; Check-Exit
}
# Keep the pinned aligner's Transformers independent of separation/TTS libraries.
$basePackages=Join-Path $PSScriptRoot '.venv\Lib\site-packages'
$pathFile=Join-Path $PSScriptRoot '.align-venv\Lib\site-packages\fast-dubbing-base.pth'
[System.IO.File]::WriteAllText($pathFile,$basePackages+[Environment]::NewLine,(New-Object System.Text.UTF8Encoding($false)))
$alignPython=Join-Path $PSScriptRoot '.align-venv\Scripts\python.exe'
& $alignPython -m pip install --no-deps -r requirements-alignment.txt; Check-Exit
& $alignPython -c "from qwen_asr import Qwen3ForcedAligner; from alignment import load_model; print('Reference alignment runtime ready')"; Check-Exit
