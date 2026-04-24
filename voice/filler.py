"""
Filler audio — pre-synthesized clips that play instantly to mask agent latency.

Usage:
    preload()               — call once at startup (parallel; ~1.5s total)
    play(intent, lang)      — non-blocking, starts playing immediately
    stop()                  — cut playback before real response starts
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd

from voice.deepgram_tts import synthesize_to_array, _SAMPLE_RATE

log = logging.getLogger("filler")

# Short, natural filler phrases per intent × language
_TEXTS: dict[tuple[str, str], str] = {
    ("PLACE_ORDER",  "roman_urdu"): "Haan ji, ek second...",
    ("PLACE_ORDER",  "en"):         "Sure, one moment...",
    ("CHECKOUT",     "roman_urdu"): "Ji, order confirm kar raha hun...",
    ("CHECKOUT",     "en"):         "Confirming your order...",
    ("GET_MENU",     "roman_urdu"): "Abhi batata hun...",
    ("GET_MENU",     "en"):         "Here's our menu...",
    ("ORDER_QUERY",  "roman_urdu"): "Ek second, dekh raha hun...",
    ("ORDER_QUERY",  "en"):         "Let me check...",
    ("SMALL_TALK",   "roman_urdu"): "",   # instant reply, no filler needed
    ("SMALL_TALK",   "en"):         "",
}

_cache: dict[tuple[str, str], np.ndarray | None] = {}


def _synth_one(key: tuple[str, str], text: str) -> tuple[tuple[str, str], np.ndarray | None]:
    try:
        arr = synthesize_to_array(text)
        return key, arr
    except Exception as e:
        log.warning("Filler preload failed %s: %s", key, e)
        return key, None


def preload() -> None:
    """Synthesize every filler clip in parallel and cache as numpy arrays."""
    jobs = [(k, t) for k, t in _TEXTS.items() if t]
    log.info("Filler: preloading %d clips...", len(jobs))

    for key, text in _TEXTS.items():
        if not text:
            _cache[key] = None

    with ThreadPoolExecutor(max_workers=min(8, len(jobs) or 1)) as pool:
        for key, arr in pool.map(lambda kt: _synth_one(*kt), jobs):
            _cache[key] = arr

    ok = sum(1 for v in _cache.values() if v is not None and len(v) > 0)
    log.info("Filler: preload complete (%d/%d cached)", ok, len(jobs))


def play(intent: str, lang: str) -> None:
    """
    Start playing the filler for this intent/language. Returns immediately.
    Falls back to English filler if no Roman Urdu clip exists.
    """
    arr = _cache.get((intent, lang))
    if arr is None or len(arr) == 0:
        arr = _cache.get((intent, "en"))
    if arr is not None and len(arr) > 0:
        sd.play(arr, samplerate=_SAMPLE_RATE, blocking=False)


def stop() -> None:
    """Stop any filler that is currently playing."""
    sd.stop()
