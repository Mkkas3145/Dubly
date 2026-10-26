$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Get-NetTCPConnection -LocalPort 20806 -State Listen -ErrorAction SilentlyContinue) { Write-Host 'http://localhost:20806 is already running.'; exit 0 }
& "$PSScriptRoot\.venv\Scripts\python.exe" -u "$PSScriptRoot\app.py"
