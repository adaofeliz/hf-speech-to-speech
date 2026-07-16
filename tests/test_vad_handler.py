"""Tests for VADHandler threshold escalation and noise-floor calibration."""

from __future__ import annotations

import time
from threading import Event
from unittest.mock import patch

import numpy as np
import torch

from speech_to_speech.VAD.vad_handler import VADHandler


class _FakeVADIterator:
    """Minimal VAD iterator stub for unit tests (no torch hub, no silero download)."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self.triggered = False
        self.buffer: list[torch.Tensor] = []
        self.active_speech_samples: int = 0
        self.last_utterance_active_speech_samples: int = 0

    def __call__(self, x: torch.Tensor) -> None:
        return None

    def speech_buffer(self) -> list[torch.Tensor]:
        return []

    def reset_states(self) -> None:
        pass


def _make_handler(
    thresh: float = 0.6,
    response_playing: Event | None = None,
) -> VADHandler:
    """Construct a VADHandler without the silero model download.

    Mirrors the ``_vad_handler_for_iterator`` helper convention used in
    test_speculative_turns.py: ``object.__new__`` bypasses ``setup()`` and we
    set every attribute the tested code-paths actually read.
    """
    handler = object.__new__(VADHandler)
    handler.should_listen = Event()
    handler.should_listen.set()
    handler.sample_rate = 16000
    handler.min_silence_ms = 300
    handler.min_speech_ms = 384
    handler.min_speech_continuation_ms = 384
    handler.max_speech_ms = float("inf")
    handler.enable_realtime_transcription = False
    handler.realtime_processing_pause = 0.5
    handler.text_output_queue = None
    handler.speculative_turns = None
    handler.speculative_reopen_ms = 1000
    handler.unanswered_reopen_ms = 1000
    handler._last_turn_detection = None
    handler.iterator = _FakeVADIterator(threshold=thresh)
    handler._base_thresh = thresh
    handler.response_playing = response_playing
    handler.audio_enhancement = False
    handler.last_process_time = 0.0
    handler._total_samples = 0
    handler._last_log_time = time.time()
    handler._log_chunks = 0
    handler._log_speech_starts = 0
    handler._log_speech_ends = 0
    handler._log_progressive_yields = 0
    handler._speech_started_emitted = False
    handler._turn_counter = 0
    handler._current_turn_id = None
    handler._current_turn_revision = None
    handler._speculative_audio_prefix = None
    handler._last_final_wall_time = None
    handler._last_final_audio_ms = None
    handler._pending_reopen_candidate = None
    handler.short_segment_merge_ms = 0
    handler._pending_short_segment = None
    return handler


def _silence_bytes(samples: int = 512) -> bytes:
    return np.zeros(samples, dtype=np.int16).tobytes()


# ---------------------------------------------------------------------------
# Threshold escalation tests
# ---------------------------------------------------------------------------


def test_threshold_escalates_when_response_playing_set() -> None:
    """Threshold rises to min(0.95, base+0.25) while the response Event is set."""
    evt = Event()
    handler = _make_handler(thresh=0.6, response_playing=evt)

    evt.set()
    list(handler.process(_silence_bytes()))

    expected = min(0.95, 0.6 + 0.25)  # 0.85
    assert handler.iterator.threshold == expected


def test_threshold_reverts_to_base_when_response_playing_cleared() -> None:
    """After clearing the Event the threshold drops back to the base value."""
    evt = Event()
    handler = _make_handler(thresh=0.6, response_playing=evt)

    # First call: escalate
    evt.set()
    list(handler.process(_silence_bytes()))
    assert handler.iterator.threshold == min(0.95, 0.6 + 0.25)

    # Second call: revert
    evt.clear()
    list(handler.process(_silence_bytes()))
    assert handler.iterator.threshold == 0.6


def test_threshold_escalation_caps_at_0_95() -> None:
    """base + 0.25 > 0.95 must be clamped to exactly 0.95."""
    evt = Event()
    handler = _make_handler(thresh=0.75, response_playing=evt)

    evt.set()
    list(handler.process(_silence_bytes()))

    # 0.75 + 0.25 == 1.0 > 0.95 → should be capped
    assert handler.iterator.threshold == 0.95


def test_threshold_escalation_composes_with_runtime_turn_detection_update() -> None:
    """A runtime turn_detection threshold update changes _base_thresh; subsequent
    escalation is on top of the new base, not the original constructor value."""
    from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

    from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig

    evt = Event()
    handler = _make_handler(thresh=0.5, response_playing=evt)

    # --- Phase 1: runtime update changes base to 0.65 (response not playing) ---
    cfg = RuntimeConfig()
    cfg.session.audio.input.turn_detection = ServerVad(type="server_vad", threshold=0.65)
    # Feed the tuple form so _apply_runtime_turn_detection fires
    list(handler.process((_silence_bytes(), cfg)))

    assert handler._base_thresh == 0.65, "runtime update must write _base_thresh"

    # --- Phase 2: now start playing; escalation should use 0.65 as base ---
    evt.set()
    list(handler.process(_silence_bytes()))

    expected = min(0.95, 0.65 + 0.25)  # 0.90
    assert handler.iterator.threshold == expected


# ---------------------------------------------------------------------------
# Noise-floor calibration tests
# ---------------------------------------------------------------------------


def test_calibration_enabled_default_thresh_updates_threshold() -> None:
    """With _calibrate=True (enable_noise_calibration=True + thresh_is_default=True),
    chunks during the window are withheld (nothing yielded), and after the deadline
    both iterator.threshold and _base_thresh are set to the calibrated value."""
    handler = _make_handler(thresh=0.6)
    handler._calibrate = True
    handler._calibration_deadline = None
    handler._calibration_samples = []
    handler._calibration_window_s = 1.5

    # time.time() calls per chunk:
    #   within-window chunk: 1 call (calibration check) then early return
    #   fall-through chunk:  1 call (calibration) + 1 call (logging check)
    time_sequence = [0.0, 0.5, 1.0, 2.0, 2.0]
    with patch("speech_to_speech.VAD.vad_handler.time.time", side_effect=time_sequence):
        for i in range(3):
            result = list(handler.process(_silence_bytes()))
            assert result == [], f"chunk {i + 1}: must be withheld during calibration window"
            assert handler._calibrate is True, f"chunk {i + 1}: _calibrate must remain True in window"

        list(handler.process(_silence_bytes()))

    assert handler._calibrate is False
    assert 0.4 <= handler.iterator.threshold <= 0.9
    assert handler.iterator.threshold == handler._base_thresh


def test_calibration_skipped_when_thresh_not_default() -> None:
    """When thresh_is_default=False, _calibrate is False and no calibration occurs."""
    handler = _make_handler(thresh=0.7)
    handler.enable_noise_calibration = True
    assert handler._calibrate is False

    list(handler.process(_silence_bytes()))

    assert handler._calibrate is False
    assert handler.iterator.threshold == 0.7
    assert handler._base_thresh == 0.7


def test_calibration_disabled_by_default_is_noop() -> None:
    """Default handler (enable_noise_calibration=False) must behave exactly as pre-calibration baseline."""
    handler = _make_handler(thresh=0.6)
    assert handler._calibrate is False

    list(handler.process(_silence_bytes()))

    assert handler._calibrate is False
    assert handler._calibration_deadline is None
    assert handler.iterator.threshold == 0.6
    assert handler._base_thresh == 0.6
