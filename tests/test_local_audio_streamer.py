"""Tests for LocalAudioStreamer.

Key non-AEC-dependent assertions that must pass regardless of whether
``pywebrtc-audio`` is installed:

- Near-end audio is ALWAYS captured into ``input_queue`` (even when the output
  queue has data to play).
- ``outdata.fill(0)`` is used for silence paths (no stale-data leak from
  ``0 * outdata``).
- The bare ``except Exception`` branch logs via ``logger.exception`` and still
  captures near-end audio.

AEC-specific assertions are skipped when ``pywebrtc-audio`` is absent.
"""

from __future__ import annotations

import sys
import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# sounddevice raises OSError at import time when the native PortAudio library is
# not installed (e.g. CI Ubuntu runners).  The tests mock sd.Stream anyway, so
# stub the whole module before importing LocalAudioStreamer when needed.
try:
    import sounddevice  # noqa: F401
except OSError:
    sys.modules["sounddevice"] = MagicMock()

from speech_to_speech.connections.local_audio_streamer import LocalAudioStreamer
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE
from speech_to_speech.VAD.aec import HAS_AEC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHUNK = 512
_SR = 16_000


def _make_streamer(enable_aec: bool = False) -> LocalAudioStreamer:
    return LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=threading.Event(),
        list_play_chunk_size=_CHUNK,
        enable_aec=enable_aec,
    )


def _run_callback(streamer: LocalAudioStreamer) -> tuple[callable, np.ndarray, np.ndarray]:
    """Start the stream, capture the callback, return (callback, indata, outdata)."""
    captured: dict[str, object] = {}
    indata = np.zeros((_CHUNK, 1), dtype=np.int16)
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    class FakeStream:
        def __init__(self, **kw):
            captured["callback"] = kw["callback"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("speech_to_speech.connections.local_audio_streamer.sd.Stream", FakeStream):
        with patch("speech_to_speech.connections.local_audio_streamer.sd.query_devices"):
            # run() blocks; we stop immediately after the stream context is entered
            streamer.stop_event.set()
            import threading as _t

            t = _t.Thread(target=streamer.run, daemon=True)
            t.start()
            t.join(timeout=2.0)

    cb = captured.get("callback")
    return cb, indata, outdata


def _get_callback(streamer: LocalAudioStreamer):
    """Return the sounddevice callback without actually running the stream loop."""
    captured: dict = {}

    class FakeStream:
        def __init__(self, **kw):
            captured["callback"] = kw["callback"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import threading as _t

    with patch("speech_to_speech.connections.local_audio_streamer.sd.Stream", FakeStream):
        with patch("speech_to_speech.connections.local_audio_streamer.sd.query_devices"):
            streamer.stop_event.set()
            t = _t.Thread(target=streamer.run, daemon=True)
            t.start()
            t.join(timeout=2.0)

    cb = captured.get("callback")
    # Clear stop_event so tests can invoke the callback in normal (non-shutdown) mode.
    streamer.stop_event.clear()
    return cb


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_without_aec_does_not_raise() -> None:
    streamer = _make_streamer(enable_aec=False)
    assert streamer._aec is None


@pytest.mark.skipif(not HAS_AEC, reason="pywebrtc-audio not installed")
def test_construction_with_aec_creates_processor() -> None:
    streamer = _make_streamer(enable_aec=True)
    assert streamer._aec is not None


def test_construction_enable_aec_false_default() -> None:
    streamer = LocalAudioStreamer(
        input_queue=Queue(),
        output_queue=Queue(),
        should_listen=threading.Event(),
    )
    assert streamer._aec is None


# ---------------------------------------------------------------------------
# Near-end audio always captured
# ---------------------------------------------------------------------------


def _invoke_cb(cb, indata, outdata, *, queue_item=None, stop=False, trigger_exception=False):
    """Helper to invoke the captured callback with optional queue/stop state."""
    return cb(indata, outdata, len(indata), 0.0, "")  # type: ignore[call-arg]


def _build_cb_env(queue_item=None) -> tuple:
    """Build a streamer + callback setup for near-end capture tests."""
    input_q: Queue = Queue()
    output_q: Queue = Queue()
    should_listen = threading.Event()

    streamer = LocalAudioStreamer(
        input_queue=input_q,
        output_queue=output_q,
        should_listen=should_listen,
        list_play_chunk_size=_CHUNK,
        enable_aec=False,
    )

    if queue_item is not None:
        output_q.put(queue_item)

    cb = _get_callback(streamer)
    return streamer, input_q, output_q, should_listen, cb


def test_near_end_captured_when_output_queue_empty() -> None:
    streamer, input_q, output_q, _, cb = _build_cb_env(queue_item=None)
    assert cb is not None, "callback was not captured"

    indata = np.ones((_CHUNK, 1), dtype=np.int16) * 100
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    cb(indata, outdata, _CHUNK, 0.0, "")
    assert not input_q.empty(), "near-end should be enqueued"
    item = input_q.get_nowait()
    assert item == indata.astype(np.int16).tobytes()


def test_near_end_captured_when_output_queue_has_pcm() -> None:
    """Near-end must be captured even while a PCM chunk is playing (the bug that was fixed)."""
    audio_chunk = np.ones(_CHUNK, dtype=np.int16) * 500
    streamer, input_q, output_q, _, cb = _build_cb_env(queue_item=audio_chunk)
    assert cb is not None

    indata = np.ones((_CHUNK, 1), dtype=np.int16) * 200
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    cb(indata, outdata, _CHUNK, 0.0, "")

    assert not input_q.empty(), "near-end must be captured while playing audio"


def test_near_end_captured_on_audio_response_done() -> None:
    streamer, input_q, output_q, should_listen, cb = _build_cb_env(queue_item=AUDIO_RESPONSE_DONE)
    assert cb is not None

    indata = np.ones((_CHUNK, 1), dtype=np.int16) * 300
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    cb(indata, outdata, _CHUNK, 0.0, "")

    assert not input_q.empty(), "near-end must be captured on AUDIO_RESPONSE_DONE"
    assert should_listen.is_set(), "should_listen must be set after AUDIO_RESPONSE_DONE"


def test_near_end_captured_on_exception() -> None:
    """Near-end stays in the queue even when get_nowait raises."""
    input_q: Queue = Queue()
    output_q: Queue = Queue()
    should_listen = threading.Event()

    streamer = LocalAudioStreamer(
        input_queue=input_q,
        output_queue=output_q,
        should_listen=should_listen,
        list_play_chunk_size=_CHUNK,
        enable_aec=False,
    )
    # Seed non-empty queue so we enter the else branch, then patch get_nowait to raise
    output_q.put(np.zeros(_CHUNK, dtype=np.int16))

    cb = _get_callback(streamer)
    assert cb is not None

    indata = np.ones((_CHUNK, 1), dtype=np.int16) * 400
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    with patch.object(output_q, "get_nowait", side_effect=RuntimeError("boom")):
        cb(indata, outdata, _CHUNK, 0.0, "")

    assert not input_q.empty(), "near-end must be captured even when an exception occurs"


# ---------------------------------------------------------------------------
# Silence paths use fill(0), not stale-data-preserving 0 * outdata
# ---------------------------------------------------------------------------


def test_stop_event_fills_outdata_with_zeros() -> None:
    input_q: Queue = Queue()
    output_q: Queue = Queue()
    streamer = LocalAudioStreamer(
        input_queue=input_q,
        output_queue=output_q,
        should_listen=threading.Event(),
        list_play_chunk_size=_CHUNK,
    )

    cb = _get_callback(streamer)
    assert cb is not None

    streamer.stop_event.set()  # trigger the shutdown branch AFTER callback is captured

    indata = np.zeros((_CHUNK, 1), dtype=np.int16)
    outdata = np.ones((_CHUNK, 1), dtype=np.int16) * 999  # pre-fill with garbage

    cb(indata, outdata, _CHUNK, 0.0, "")

    assert np.all(outdata == 0), "stop event branch must zero outdata"


def test_audio_response_done_fills_outdata_with_zeros() -> None:
    input_q: Queue = Queue()
    output_q: Queue = Queue()
    output_q.put(AUDIO_RESPONSE_DONE)
    streamer = LocalAudioStreamer(
        input_queue=input_q,
        output_queue=output_q,
        should_listen=threading.Event(),
        list_play_chunk_size=_CHUNK,
    )

    cb = _get_callback(streamer)
    assert cb is not None

    indata = np.zeros((_CHUNK, 1), dtype=np.int16)
    outdata = np.ones((_CHUNK, 1), dtype=np.int16) * 999

    cb(indata, outdata, _CHUNK, 0.0, "")

    assert np.all(outdata == 0), "AUDIO_RESPONSE_DONE branch must zero outdata"


# ---------------------------------------------------------------------------
# Exception branch logs via logger.exception
# ---------------------------------------------------------------------------


def test_exception_branch_calls_logger_exception() -> None:
    input_q: Queue = Queue()
    output_q: Queue = Queue()
    output_q.put(np.zeros(_CHUNK, dtype=np.int16))

    streamer = LocalAudioStreamer(
        input_queue=input_q,
        output_queue=output_q,
        should_listen=threading.Event(),
        list_play_chunk_size=_CHUNK,
    )

    cb = _get_callback(streamer)
    assert cb is not None

    indata = np.zeros((_CHUNK, 1), dtype=np.int16)
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    with patch("speech_to_speech.connections.local_audio_streamer.logger") as mock_logger:
        with patch.object(output_q, "get_nowait", side_effect=RuntimeError("oops")):
            cb(indata, outdata, _CHUNK, 0.0, "")
        mock_logger.exception.assert_called_once()


# ---------------------------------------------------------------------------
# AEC wiring path
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_AEC, reason="pywebrtc-audio not installed")
def test_aec_process_called_when_playing_pcm_chunk() -> None:
    """When enable_aec=True and a PCM chunk is playing, AEC must be invoked."""
    input_q: Queue = Queue()
    output_q: Queue = Queue()

    audio_chunk = np.ones(_CHUNK, dtype=np.int16) * 500
    output_q.put(audio_chunk)

    streamer = LocalAudioStreamer(
        input_queue=input_q,
        output_queue=output_q,
        should_listen=threading.Event(),
        list_play_chunk_size=_CHUNK,
        enable_aec=True,
    )

    cb = _get_callback(streamer)
    assert cb is not None
    assert streamer._aec is not None

    indata = np.ones((_CHUNK, 1), dtype=np.int16) * 200
    outdata = np.zeros((_CHUNK, 1), dtype=np.int16)

    with patch.object(streamer._aec, "process", wraps=streamer._aec.process) as mock_process:
        cb(indata, outdata, _CHUNK, 0.0, "")
        mock_process.assert_called_once()

    assert not input_q.empty(), "AEC-processed near-end must be enqueued"


@pytest.mark.skipif(HAS_AEC, reason="only relevant when pywebrtc-audio absent")
def test_enable_aec_raises_runtime_error_when_library_absent() -> None:
    """Constructing LocalAudioStreamer with enable_aec=True must fail loudly."""
    with pytest.raises(RuntimeError, match="pywebrtc-audio"):
        _make_streamer(enable_aec=True)
