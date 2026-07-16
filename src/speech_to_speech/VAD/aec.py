"""Optional acoustic echo cancellation (AEC) wrapper around pywebrtc-audio.

The module is importable with zero side effects when ``pywebrtc-audio`` is not
installed; ``HAS_AEC`` is set to ``False`` in that case.  Constructing
:class:`AecProcessor` while the library is absent raises a :exc:`RuntimeError`
with an actionable install message.

Audio is processed in WebRTC AEC3's native 10 ms / 160-sample frames at 16 kHz.
The processor buffers any remainder across calls so callers feed arbitrary-length
PCM blobs without managing alignment themselves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional-dependency guard – must never raise at import time
# ---------------------------------------------------------------------------
try:
    import pywebrtc_audio  # type: ignore[import-untyped]

    HAS_AEC: bool = True
except (ImportError, ModuleNotFoundError):
    pywebrtc_audio = None  # type: ignore[assignment]
    HAS_AEC = False

if TYPE_CHECKING:
    pass

# WebRTC AEC3 constants at 16 kHz
_SAMPLE_RATE: int = 16_000
_FRAME_SAMPLES: int = 160  # 10 ms × 16 000 Hz


class AecProcessor:
    """Stateful AEC processor for a single 16 kHz mono audio stream.

    Parameters
    ----------
    sample_rate:
        Must be 16 000 – the only rate supported by WebRTC AEC3.

    Raises
    ------
    RuntimeError
        Raised at construction time (not import time) when ``pywebrtc-audio``
        is not installed, with an actionable message.
    ValueError
        Raised when *sample_rate* is not 16 000.
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE) -> None:
        if not HAS_AEC:
            raise RuntimeError(
                "pywebrtc-audio is not installed.  "
                "Install it with:  pip install 'speech-to-speech[aec]'  "
                "or:  pip install 'pywebrtc-audio>=0.1.0'"
            )
        if sample_rate != _SAMPLE_RATE:
            raise ValueError(f"AecProcessor only supports sample_rate={_SAMPLE_RATE}; got {sample_rate}.")

        self._processor = pywebrtc_audio.AudioProcessor(
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
        )

        # Partial-frame buffers (int16 flat arrays)
        self._near_buf: np.ndarray = np.empty(0, dtype=np.int16)
        self._far_buf: np.ndarray = np.empty(0, dtype=np.int16)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, near: bytes, far: bytes) -> bytes:
        """AEC-process *near* (microphone) against *far* (speaker/TTS) audio.

        Both inputs are raw 16-bit little-endian mono PCM at 16 kHz.  They do
        not need to be the same length or aligned to 10 ms frames – internal
        buffering handles arbitrary chunk sizes.

        Returns the echo-cancelled near-end audio (same byte length as *near*).
        """
        near_pcm = np.frombuffer(near, dtype=np.int16)
        far_pcm = np.frombuffer(far, dtype=np.int16)

        # Append to running buffers
        near_samples = np.concatenate([self._near_buf, near_pcm])
        far_samples = np.concatenate([self._far_buf, far_pcm])

        out_chunks: list[np.ndarray] = []

        # Process as many complete 160-sample frames as possible
        n_frames = min(len(near_samples), len(far_samples)) // _FRAME_SAMPLES

        for i in range(n_frames):
            start = i * _FRAME_SAMPLES
            end = start + _FRAME_SAMPLES

            near_frame = near_samples[start:end]
            far_frame = far_samples[start:end]

            try:
                result = self._processor.process(near_frame, far_frame)
                out_chunks.append(np.asarray(result, dtype=np.int16))
            except Exception:
                logger.exception("AEC process() failed; passing near-end frame through unmodified")
                out_chunks.append(near_frame)

        processed_samples = n_frames * _FRAME_SAMPLES

        # Keep unprocessed remainders for the next call
        self._near_buf = near_samples[processed_samples:]
        # Far-end remainder: keep aligned with near buffer
        far_processed = min(processed_samples, len(far_samples))
        self._far_buf = far_samples[far_processed:]

        if not out_chunks:
            # No complete frame yet – return silence of the same length as input
            return bytes(len(near))

        processed = np.concatenate(out_chunks)

        # Pad/trim to match input length so callers get back the same byte count
        input_len = len(near_pcm)
        if len(processed) < input_len:
            padding = np.zeros(input_len - len(processed), dtype=np.int16)
            processed = np.concatenate([processed, padding])
        elif len(processed) > input_len:
            processed = processed[:input_len]

        return processed.tobytes()
