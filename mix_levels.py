"""One fixed gain for dubbed speech; source/background levels stay untouched."""

# Measured on 399 utterances from nine videos: source speech minus fitted TTS
# median 3.69 LU, full source mix minus fitted TTS median 4.16 LU.
TTS_GAIN_DB = 4.0
TTS_GAIN = 10 ** (TTS_GAIN_DB / 20)
VOLUME_VERSION = 5
