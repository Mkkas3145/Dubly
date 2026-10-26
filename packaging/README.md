# Windows installer build

End users only run the installer. They do not need Python, .NET, FFmpeg, WSL, or model downloads. The current build requires Windows 10 version 1809 or later (x64) and a supported NVIDIA GPU/driver.

For developers, build from the configured production environment with .NET 8 SDK, Inno Setup 7, Visual Studio C++ Build Tools, and LLVM installed. Run `setup.ps1` first to prepare the production dependencies and models; MFA provisioning uses Miniforge.

1. Publish the launcher:
   `dotnet publish desktop/Dubly.Launcher/Dubly.Launcher.csproj -c Release -r win-x64 --self-contained true -o work/desktop-publish`
2. Download Python 3.11.9 Windows embeddable package (64-bit) from python.org.
3. Assemble the payload:
   `.venv/Scripts/python.exe packaging/build_bundle.py --output work/windows-bundle --python-embed <python-3.11.9-embed-amd64.zip>`
4. Compile:
   `ISCC /DBundleRoot=<absolute-payload-path> /DOutputDir=<absolute-output-path> packaging/Dubly.iss`

The builder copies only selected runtimes/models and application source. API keys, user videos, generated outputs and logs are excluded. Third-party licenses accompany the installer. `.gitignore` keeps local runtime and build artifacts out of the source repository.

The optional launcher smoke check uses an unused local port and a placeholder API key:
`Dubly.exe --smoke-test <report.json> <port>`

For GitHub Releases, add `/DGitHubRelease=1` to the ISCC command. This produces an EXE and companion BIN files, each below the release asset limit. Download all files into the same folder before running the EXE.
