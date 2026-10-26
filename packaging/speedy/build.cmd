@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo Visual Studio C++ Build Tools are required. 1>&2 & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (echo Visual Studio C++ Build Tools were not found. 1>&2 & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
where clang.exe >nul 2>nul
if errorlevel 1 set "PATH=%ProgramFiles%\LLVM\bin;%PATH%"
cd /d "%~dp0.."
clang.exe -O3 -DSONIC_INTERNAL -Isonic -c sonic/sonic.c -o windows/sonic_internal.obj
if errorlevel 1 exit /b 1
clang.exe -O3 -DKISS_FFT -Isonic -Ikissfft -Ispeedy -c speedy/speedy.c -o windows/speedy.obj
if errorlevel 1 exit /b 1
clang.exe -O3 -Isonic -Ispeedy -c speedy/soniclib.c -o windows/soniclib.obj
if errorlevel 1 exit /b 1
clang.exe -O3 -Ikissfft -c kissfft/kiss_fft.c -o windows/kiss_fft.obj
if errorlevel 1 exit /b 1
link /nologo /DLL /DEFAULTLIB:libcmt.lib /DEFAULTLIB:libucrt.lib /DEFAULTLIB:libvcruntime.lib kernel32.lib /DEF:windows/exports.def /OUT:windows/speedy.dll windows/sonic_internal.obj windows/speedy.obj windows/soniclib.obj windows/kiss_fft.obj
