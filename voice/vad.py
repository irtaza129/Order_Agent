"""
Lightweight energy-based voice activity detector.

Pure-numpy, no native deps — works on Python 3.14 where webrtcvad can't build.
Used to suppress background noise (fan hum, typing, distant voices) before
audio reaches Deepgram. Non-speech frames are zeroed, not dropped, so the
Deepgram socket keeps a continuous timing reference.

Design:
  - Auto-calibrate a noise floor from the first ~500ms of audio.
  - Open the gate when frame RMS exceeds `noise_floor * open_ratio`.
  - Close the gate after `hangover_ms` of sub-threshold audio (hysteresis).
  - Emit a short pre-roll buffer on open so onset consonants aren't clipped.
"""

from __future__ import annotations

import collections
import numpy as np


class NoiseGate:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        calibration_ms: int = 500,
        open_ratio: float = 3.0,
        close_ratio: float = 1.8,
        hangover_ms: int = 400,
        preroll_ms: int = 240,
        min_open_floor: float = 180.0,
    ) -> None:
        self._sr = sample_rate
        self._frame_samples = int(sample_rate * frame_ms / 1000)
        self._frame_bytes = self._frame_samples * 2  # int16
        self._open_ratio = open_ratio
        self._close_ratio = close_ratio
        self._min_open_floor = min_open_floor

        self._calib_target = max(1, calibration_ms // frame_ms)
        self._calib_rms: list[float] = []
        self._noise_floor: float | None = None

        self._hangover_frames = max(1, hangover_ms // frame_ms)
        self._below_count = 0
        self._open = False

        preroll_frames = max(1, preroll_ms // frame_ms)
        self._preroll: collections.deque[bytes] = collections.deque(maxlen=preroll_frames)

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    @property
    def calibrated(self) -> bool:
        return self._noise_floor is not None

    def _rms(self, pcm: bytes) -> float:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))

    def process(self, pcm: bytes) -> bytes:
        """
        Return audio to forward. Non-speech frames are zeroed (silent PCM),
        so the downstream socket keeps its clock but transcribes nothing.
        """
        if len(pcm) != self._frame_bytes:
            # Frame size mismatch — pass through unmodified.
            return pcm

        rms = self._rms(pcm)

        if self._noise_floor is None:
            self._calib_rms.append(rms)
            if len(self._calib_rms) >= self._calib_target:
                floor = float(np.median(self._calib_rms))
                self._noise_floor = max(floor, self._min_open_floor)
            self._preroll.append(pcm)
            return b"\x00" * self._frame_bytes

        open_thresh = self._noise_floor * self._open_ratio
        close_thresh = self._noise_floor * self._close_ratio

        if not self._open:
            if rms > open_thresh:
                self._open = True
                self._below_count = 0
                # Flush pre-roll so the onset isn't clipped.
                preroll = b"".join(self._preroll)
                self._preroll.clear()
                self._preroll.append(pcm)
                return preroll + pcm
            self._preroll.append(pcm)
            return b"\x00" * self._frame_bytes

        # Gate is open.
        if rms < close_thresh:
            self._below_count += 1
            if self._below_count >= self._hangover_frames:
                self._open = False
                self._below_count = 0
        else:
            self._below_count = 0

        self._preroll.append(pcm)
        return pcm
