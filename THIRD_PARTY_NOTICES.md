# Third-party components

Dubly source code is licensed under Apache-2.0. Bundled dependencies retain their own licenses.

- Python 3.11.9: Python Software Foundation License. https://www.python.org/downloads/release/python-3119/
- .NET 8 Windows Desktop runtime: MIT. https://github.com/dotnet/runtime
- Google Speedy and Sonic: Apache-2.0. Upstream source pins are recorded in `server/tools/google-speedy/build.json`.
- KissFFT: BSD-3-Clause and Unlicense; see COPYING and upstream source. https://github.com/mborgerding/kissfft
- FFmpeg 8.1 full build: GPL-3.0-or-later, including optional linked codecs. Corresponding source and build configuration: https://www.gyan.dev/ffmpeg/builds/ and https://github.com/FFmpeg/FFmpeg/tree/n8.1 . FFmpeg is distributed as a separate executable.
- Mel-Band RoFormer Deux weights: CC-BY-NC-4.0. Attribution: becruily. Noncommercial use only. https://huggingface.co/becruily/mel-band-roformer-deux
- Mel-Band RoFormer source: https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model . Source is included with the application.
- Montreal Forced Aligner 3.4.3 and Korean MFA acoustic, dictionary and G2P models: upstream package/model licenses apply. https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner and https://mfa-models.readthedocs.io/
- Meta MMS-FA alignment weights: CC-BY-NC-4.0. Attribution: Vineel Pratap et al., Scaling Speech Technology to 1,000+ Languages (2023). https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html
- Pretendard: SIL Open Font License 1.1. https://github.com/orioncactus/pretendard
- PyTorch, CUDA libraries and other Python packages: licenses and notices are included in their package directories or `.dist-info` directories. NVIDIA runtime libraries are subject to NVIDIA redistribution terms. https://docs.nvidia.com/cuda/eula/

The installer targets Windows 10 version 1809 or later (x64) with a supported NVIDIA GPU and installed graphics driver. API usage requires an OpenRouter account and is billed separately by the provider.


- BandIt v2 Multi: upstream source license in `server/vendor/bandit-v2/LICENSE`; official multilingual weights from Zenodo record 12701995 retain their upstream terms. https://github.com/kwatcharasupat/bandit-v2 and https://zenodo.org/records/12701995
