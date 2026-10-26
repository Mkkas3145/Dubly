#ifndef BundleRoot
  #error BundleRoot must identify the assembled offline payload.
#endif
#ifndef OutputDir
  #error OutputDir must identify the deliverable directory.
#endif
[Setup]
AppId={{A19D951E-9241-4B30-9D41-117DA8DD9C61}
AppName=Dubly
AppVersion=0.1.0
AppPublisher=Mkkas3145
AppPublisherURL=https://github.com/Mkkas3145/Dubly
DefaultDirName={localappdata}\Programs\Dubly
DefaultGroupName=Dubly
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.22000
OutputDir={#OutputDir}
OutputBaseFilename=Dubly-Setup-0.1.0-win-x64
SetupIconFile=..\desktop\Dubly.Launcher\app.ico
UninstallDisplayIcon={app}\Dubly.exe
#ifdef GitHubRelease
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
#endif
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
LZMADictionarySize=64
WizardStyle=modern
AppMutex=Local\Dubly.Server.Launcher
CloseApplications=yes
RestartApplications=no
LicenseFile=..\LICENSE
[Files]
Source: "{#BundleRoot}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "server\work\*,server\outputs\*,server\logs\*,server\.env,server\.env.*,*.pyc,*.pdb,*\__pycache__\*"
[Tasks]
Name: desktopicon; Description: "Create a desktop shortcut"; Flags: unchecked
[Icons]
Name: "{group}\Dubly"; Filename: "{app}\Dubly.exe"
Name: "{autodesktop}\Dubly"; Filename: "{app}\Dubly.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\Dubly.exe"; Description: "Open Dubly"; Flags: nowait postinstall skipifsilent
