"""
STT Gateway — Deepgram live streaming (SDK v6).

Pipeline:
    mic → NoiseGate (zero-out non-speech) → Deepgram WebSocket → text

Latency:
  - Returns as soon as Deepgram flags `speech_final=True`, which fires at
    `endpointing` (~300ms silence). This saves ~700ms vs waiting for
    `utterance_end_ms` (minimum 1000ms per Deepgram API).
  - `utterance_end_ms` is still wired as a safety fallback if speech_final
    never arrives (shouldn't happen in practice).

Roman Urdu:
  - Default model `nova-3` + language `multi` — true multilingual, handles
    code-switched Hinglish/Roman-Urdu far better than Nova-2 English-only.
  - Model-specific term boosting: `keyterm` for nova-3, `keywords:boost` for
    nova-2. Both surface the same Pakistani food + Urdu phrase vocabulary.

Env overrides:
  DEEPGRAM_STT_MODEL          default "nova-3"
  DEEPGRAM_STT_LANGUAGE       default "multi" (nova-3) / "en" (nova-2)
  DEEPGRAM_UTTERANCE_END_MS   default 1000  — Deepgram hard minimum
  DEEPGRAM_ENDPOINTING_MS     default 300   — silence that closes a segment
  DEEPGRAM_VAD_ENABLED        default "1"   — set to "0" to bypass noise gate
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time

import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.listen.v1.socket_client import EventType
from deepgram.listen.v1.types import ListenV1Results, ListenV1UtteranceEnd

from voice.vad import NoiseGate

log = logging.getLogger("stt")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or os.getenv("deepgram", "")
_STT_MODEL = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
_DEFAULT_LANG = "multi" if _STT_MODEL.startswith("nova-3") else "en"
_STT_LANGUAGE = os.getenv("DEEPGRAM_STT_LANGUAGE", _DEFAULT_LANG)

# Deepgram enforces utterance_end_ms >= 1000. Anything lower -> HTTP 400.
_UTTERANCE_END_MS = max(1000, int(os.getenv("DEEPGRAM_UTTERANCE_END_MS", "1000")))
_ENDPOINTING_MS = int(os.getenv("DEEPGRAM_ENDPOINTING_MS", "300"))
_VAD_ENABLED = os.getenv("DEEPGRAM_VAD_ENABLED", "1") not in ("0", "false", "False", "")

_RATE = 16000
_FRAME_MS = 32  # 32ms @ 16kHz = 512 samples = 1024 bytes — aligns w/ sounddevice block
_FRAME_SAMPLES = int(_RATE * _FRAME_MS / 1000)

# Pakistani food + Roman Urdu vocabulary to bias recognition.
_TERMS: list[str] = [
    # Food items
    "pulao", "biryani", "zinger", "krispo", "burger",
    "halwa", "puri", "paratha", "kabab", "chana",
    "fries", "wings", "nuggets", "kheer", "zarda",
    "falooda", "lassi", "chai", "raita", "shami",
    "roti", "naan", "tikka", "karahi", "nihari",
    # Roman Urdu ordering phrases
    "chahiye", "kardo", "dedo", "lagao", "dijiye",
    "eik", "ek", "aur", "bas", "haan", "nahi",
    "theek", "kitna", "total", "bill", "order",
    "acha", "thik", "mujhe", "mera", "apka",
    "shukriya", "salam", "please", "yaar",
    # Pulao variants
    "special", "single", "choice", "plate", "plain",
]

_client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None


def _term_params() -> dict:
    """Pick keyterm (nova-3) or keywords+boost (nova-2) based on model."""
    if _STT_MODEL.startswith("nova-3"):
        return {"keyterm": _TERMS}
    return {"keywords": [f"{w}:3" for w in _TERMS]}


class _Session:
    """Mutable state for one transcribe_once call — isolated, no globals."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.done = threading.Event()
        self.error: Exception | None = None
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.t_first_audio: float | None = None
        self.t_final: float | None = None

    def on_message(self, message) -> None:
        if isinstance(message, ListenV1Results):
            alt = message.channel.alternatives[0] if message.channel.alternatives else None
            text = (alt.transcript or "").strip() if alt else ""
            if message.is_final and text:
                self.parts.append(text)
                log.debug("STT segment: %s", text)
            # speech_final fires at endpointing (~300ms silence) — return fast.
            if getattr(message, "speech_final", False):
                self.t_final = time.perf_counter()
                self.done.set()
        elif isinstance(message, ListenV1UtteranceEnd):
            # Fallback if we somehow missed speech_final.
            if not self.done.is_set():
                self.t_final = time.perf_counter()
                self.done.set()

    def on_error(self, error) -> None:
        log.error("STT socket error: %s", error)
        self.error = RuntimeError(str(error))
        self.done.set()


def _mic_stream(sess: _Session):
    """Yield sounddevice input stream configured for fixed-size frames."""
    def cb(indata, frames, time_info, status):
        if sess.t_first_audio is None:
            sess.t_first_audio = time.perf_counter()
        sess.audio_queue.put(bytes(indata))
    return sd.RawInputStream(
        samplerate=_RATE,
        channels=1,
        dtype="int16",
        blocksize=_FRAME_SAMPLES,
        callback=cb,
    )


def transcribe_once() -> str:
    """
    Open mic, stream gated audio to Deepgram until end-of-utterance,
    return full transcript. Blocking.
    """
    if _client is None:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    sess = _Session()
    gate = NoiseGate(sample_rate=_RATE, frame_ms=_FRAME_MS) if _VAD_ENABLED else None

    connect_kwargs = dict(
        model=_STT_MODEL,
        encoding="linear16",
        sample_rate=_RATE,
        channels=1,
        language=_STT_LANGUAGE,
        smart_format="true",
        interim_results="true",
        utterance_end_ms=_UTTERANCE_END_MS,
        endpointing=_ENDPOINTING_MS,
        vad_events="true",
        **_term_params(),
    )

    t_open = time.perf_counter()
    with _client.listen.v1.connect(**connect_kwargs) as socket:
        socket.on(EventType.MESSAGE, sess.on_message)
        socket.on(EventType.ERROR, sess.on_error)

        listener = threading.Thread(target=socket.start_listening, daemon=True)
        listener.start()

        log.info(
            "STT listening (model=%s lang=%s endpoint=%dms vad=%s)",
            _STT_MODEL, _STT_LANGUAGE, _ENDPOINTING_MS, "on" if gate else "off",
        )

        with _mic_stream(sess):
            while not sess.done.is_set():
                try:
                    frame = sess.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                out = gate.process(frame) if gate else frame
                if out:
                    socket.send_media(out)

        try:
            socket.send_finalize()
        except Exception:
            pass
        listener.join(timeout=2)

    if sess.error and not sess.parts:
        raise sess.error

    text = " ".join(sess.parts).strip()
    if sess.t_final:
        log.info(
            "STT result: '%s' (socket→final %.2fs)",
            text, sess.t_final - t_open,
        )
    else:
        log.info("STT result: '%s'", text)
    return text
