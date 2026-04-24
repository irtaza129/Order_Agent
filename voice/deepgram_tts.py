"""
TTS Gateway — Deepgram Aura streaming synthesis (SDK v6).

Raw PCM is streamed chunk-by-chunk into sounddevice as it arrives from the
network — no full-buffer wait. First audio plays within ~200ms of the request.

speak()            — blocking, call via asyncio.to_thread
synthesize_to_array — blocking, returns numpy array (used for filler caching)
"""

import logging
import os

import numpy as np
import sounddevice as sd
from deepgram import DeepgramClient

log = logging.getLogger("tts")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or os.getenv("deepgram", "")
_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")

# Deepgram wire format — must match one of: linear16, flac, mulaw, alaw, mp3, opus, aac.
# "linear16" = raw 16-bit little-endian PCM, which maps to numpy int16 / sounddevice "int16".
_DG_ENCODING = "linear16"
_PCM_DTYPE = "int16"
_SAMPLE_RATE = 24000   # Aura default output rate
_CHANNELS = 1
_BLOCKSIZE = 4096      # ~170ms of audio per write — small enough for low latency

# Reuse a single client so we don't re-create HTTP/TLS pools per call.
_client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None


def _generate(text: str):
    if _client is None:
        raise RuntimeError("DEEPGRAM_API_KEY not set")
    return _client.speak.v1.audio.generate(
        text=text,
        model=_TTS_MODEL,
        encoding=_DG_ENCODING,
        container="none",       # raw PCM — no WAV header overhead
        sample_rate=_SAMPLE_RATE,
    )


def speak(text: str) -> None:
    """Synthesize text and stream to speakers. Blocks until playback is done."""
    if not text or not text.strip():
        return
    log.info("TTS: %d chars", len(text))
    try:
        chunks = _generate(text)
        with sd.RawOutputStream(
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype=_PCM_DTYPE,
            blocksize=_BLOCKSIZE,
        ) as stream:
            for chunk in chunks:
                if chunk:
                    stream.write(chunk)
    except Exception as e:
        log.error("TTS failed: %s", e)


def synthesize_to_array(text: str) -> np.ndarray:
    """
    Synthesize text and return as float32 numpy array at _SAMPLE_RATE.
    Used once at startup to pre-cache filler clips.
    """
    chunks = _generate(text)
    raw = b"".join(c for c in chunks if c)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
