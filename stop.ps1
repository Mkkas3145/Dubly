$ErrorActionPreference='Stop'
$listener=Get-NetTCPConnection -LocalPort 20806 -State Listen -ErrorAction SilentlyContinue
if (!$listener) { Write-Host 'Already stopped.'; exit 0 }
$server=Get-CimInstance Win32_Process -Filter "ProcessId = $($listener[0].OwningProcess)"
$parent=Get-CimInstance Win32_Process -Filter "ProcessId = $($server.ParentProcessId)"
$python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$owned=@($server,$parent) | Where-Object { $_ -and $_.CommandLine -and $_.CommandLine.Contains($python) -and $_.CommandLine.Contains('app.py') } | Select-Object -Last 1
if (!$owned) { throw 'Port 20806 belongs to another application.' }
& taskkill.exe /PID $owned.ProcessId /T /F
if ($LASTEXITCODE -ne 0) { throw 'Unable to stop the server.' }
