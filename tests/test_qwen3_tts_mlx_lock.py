"""Tests for per-chunk MLX lock scoping in Qwen3-TTS (todo 5).

Verifies:
- Lock is acquired-and-released once per chunk pull, not once for the whole stream.
- Short-circuited (priority-preempted) acquire retries via ``sleep``, never hangs.
- ``StopIteration`` from an exhausted generator terminates the loop cleanly.
- ``use_mlx_lock=False`` (the default) is byte-identical to the pre-todo-5 path.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pytest

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.TTS.qwen3_tts_handler import (
    _CHUNK_LOCK_MAX_RETRIES,
    _CHUNK_LOCK_RETRY_SLEEP,
    Qwen3TTSHandler,
)
from speech_to_speech.utils.mlx_lock import _stt_priority_requested

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_stt_priority():
    _stt_priority_requested.clear()
    yield
    _stt_priority_requested.clear()


def _make_handler() -> Qwen3TTSHandler:
    """Minimal handler via ``object.__new__`` bypass — no ``setup()`` called."""
    handler = object.__new__(Qwen3TTSHandler)
    handler.cancel_scope = None
    handler.blocksize = 512
    return handler


def _audible_tuple(n_samples: int = 512, sr: int = 16_000) -> tuple[Any, int, dict]:
    """Return a ``(audio_float32, sr, timing)`` tuple that passes the speech-trim gate."""
    # Values well above 0.01 threshold so the silent-ramp trimmer keeps this chunk.
    return (np.full(n_samples, 0.5, dtype=np.float32), sr, {})


class _FakeMLXLockCtx:
    """Configurable fake MLXLockContext for test isolation.

    Class-level ``events`` and ``fail_remaining`` are reset via ``_reset_fake_ctx``
    before each test that needs this class.
    """

    events: list[tuple[str, str]] = []
    fail_remaining: int = 0

    def __init__(self, handler_name: str = "", timeout: float | None = None) -> None:
        self._handler_name = handler_name
        self._timeout = timeout
        self._acquired = False

    def __enter__(self) -> bool:
        if _FakeMLXLockCtx.fail_remaining > 0:
            _FakeMLXLockCtx.fail_remaining -= 1
            self._acquired = False
            return False
        _FakeMLXLockCtx.events.append(("enter", self._handler_name))
        self._acquired = True
        return True

    def __exit__(self, *_: Any) -> bool:
        if self._acquired:
            _FakeMLXLockCtx.events.append(("exit", self._handler_name))
        return False


def _reset_fake_ctx(fail_times: int = 0) -> None:
    _FakeMLXLockCtx.events = []
    _FakeMLXLockCtx.fail_remaining = fail_times


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: lock acquired-and-released once per chunk (not once for the whole stream)
# ─────────────────────────────────────────────────────────────────────────────


def test_per_chunk_lock_acquire_release(monkeypatch):
    """MLX lock must cycle once per chunk pull, not be held across the whole stream."""
    _reset_fake_ctx()
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockCtx)

    handler = _make_handler()
    gen = iter([_audible_tuple(), _audible_tuple(), _audible_tuple()])

    results = list(handler._stream(gen, label="test", use_mlx_lock=True))

    enters = [h for ev, h in _FakeMLXLockCtx.events if ev == "enter"]
    exits = [h for ev, h in _FakeMLXLockCtx.events if ev == "exit"]

    # 3 successful chunk pulls + 1 acquire that discovers StopIteration = 4 cycles.
    # All 4 are released: the StopIteration probe releases its lock before propagating.
    assert len(enters) == 4, f"Expected 4 lock enters (3 chunks + StopIteration probe), got {len(enters)}"
    assert len(exits) == 4, f"Expected 4 lock exits (enter=exit always), got {len(exits)}"
    assert all(h == "Qwen3TTS-chunk" for h in enters), f"Lock handler_name must be 'Qwen3TTS-chunk', got {set(enters)}"
    # Events must strictly interleave: (enter,Qwen3TTS-chunk), (exit,Qwen3TTS-chunk), …
    evs = [ev for ev, _ in _FakeMLXLockCtx.events]
    for i, ev in enumerate(evs):
        expected = "enter" if i % 2 == 0 else "exit"
        assert ev == expected, f"Event at index {i} must be '{expected}', got '{ev}'"

    # Output must contain at least 1 audio block.
    assert len(results) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: use_mlx_lock=False never touches MLXLockContext (default path unchanged)
# ─────────────────────────────────────────────────────────────────────────────


def test_use_mlx_lock_false_never_acquires_lock(monkeypatch):
    """Default ``use_mlx_lock=False`` must not invoke MLXLockContext at all."""
    _reset_fake_ctx()
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockCtx)

    handler = _make_handler()
    gen = iter([_audible_tuple(), _audible_tuple()])

    results = list(handler._stream(gen, label="test"))  # use_mlx_lock defaults to False

    assert _FakeMLXLockCtx.events == [], "No lock operations expected for use_mlx_lock=False"
    assert len(results) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: locked and unlocked paths produce byte-identical audio
# ─────────────────────────────────────────────────────────────────────────────


def test_use_mlx_lock_true_produces_same_output_as_false(monkeypatch):
    """use_mlx_lock=True and False must yield byte-identical audio blocks."""
    _reset_fake_ctx()
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockCtx)
    monkeypatch.setattr(qwen3_tts_module, "sleep", lambda _s: None)

    handler = _make_handler()
    chunks_input = [_audible_tuple(), _audible_tuple()]

    locked = list(handler._stream(iter(chunks_input), label="locked", use_mlx_lock=True))
    unlocked = list(handler._stream(iter(chunks_input), label="unlocked", use_mlx_lock=False))

    assert len(locked) == len(unlocked), "Must yield same number of audio blocks regardless of use_mlx_lock"
    for i, (a, b) in enumerate(zip(locked, unlocked)):
        np.testing.assert_array_equal(a, b, err_msg=f"Block {i} differed between locked and unlocked paths")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: retry backoff when acquire is short-circuited by STT priority
# ─────────────────────────────────────────────────────────────────────────────


def test_acquire_and_pull_next_chunk_retries_on_failed_acquire(monkeypatch):
    """Failing acquire must trigger one ``sleep(_CHUNK_LOCK_RETRY_SLEEP)`` per failure."""
    _reset_fake_ctx(fail_times=3)  # fail 3 times, succeed on the 4th attempt
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockCtx)

    sleep_calls: list[float] = []
    monkeypatch.setattr(qwen3_tts_module, "sleep", lambda s: sleep_calls.append(s))

    handler = _make_handler()
    item = _audible_tuple()
    gen_iter = iter([item])

    result = handler._acquire_and_pull_next_chunk(gen_iter)

    assert sleep_calls == [_CHUNK_LOCK_RETRY_SLEEP] * 3, (
        f"Must sleep exactly 3 times with _CHUNK_LOCK_RETRY_SLEEP={_CHUNK_LOCK_RETRY_SLEEP}, got {sleep_calls}"
    )
    assert result == item, "Must return the item after retries succeed"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: exhausted retry budget raises TimeoutError (never hangs)
# ─────────────────────────────────────────────────────────────────────────────


def test_acquire_and_pull_next_chunk_raises_timeout_after_budget(monkeypatch):
    """After _CHUNK_LOCK_MAX_RETRIES failed acquires, must raise TimeoutError, not hang."""
    _reset_fake_ctx(fail_times=_CHUNK_LOCK_MAX_RETRIES)  # all attempts fail
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockCtx)
    monkeypatch.setattr(qwen3_tts_module, "sleep", lambda _s: None)  # instant

    handler = _make_handler()
    gen_iter = iter([_audible_tuple()])

    with pytest.raises(TimeoutError):
        handler._acquire_and_pull_next_chunk(gen_iter)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: StopIteration from an exhausted generator ends the loop cleanly
# ─────────────────────────────────────────────────────────────────────────────


def test_stop_iteration_ends_stream_cleanly(monkeypatch):
    """A single-item generator must not raise; loop must terminate without error."""
    _reset_fake_ctx()
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockCtx)

    handler = _make_handler()
    gen = iter([_audible_tuple()])  # exhausts after 1 item

    # Must NOT raise StopIteration or RuntimeError
    results = list(handler._stream(gen, label="stop_iter_test", use_mlx_lock=True))

    assert len(results) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: _stream_mlx_generation uses "Qwen3TTS-setup" for the setup lock
# ─────────────────────────────────────────────────────────────────────────────


def test_stream_mlx_generation_setup_lock_name_and_per_chunk_lock(monkeypatch):
    """_stream_mlx_generation must use 'Qwen3TTS-setup' for the setup lock and
    'Qwen3TTS-chunk' for each per-chunk pull."""
    lock_names: list[str] = []

    class _CaptureName:
        def __init__(self, handler_name: str = "", timeout: float | None = None) -> None:
            self._name = handler_name
            self._acquired = False

        def __enter__(self) -> bool:
            lock_names.append(self._name)
            self._acquired = True
            return True

        def __exit__(self, *_: Any) -> bool:
            return False

    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _CaptureName)
    monkeypatch.setattr(qwen3_tts_module, "sleep", lambda _s: None)

    handler = _make_handler()
    handler.gen_kwargs = {}
    handler.streaming_chunk_size = 4

    def _fake_generation_fn(**kwargs: Any) -> Iterator[Any]:
        yield _audible_tuple()

    results = list(handler._stream_mlx_generation(_fake_generation_fn, label="test", max_tokens=64))

    assert len(lock_names) >= 1, "At least one MLXLockContext must be created"
    assert lock_names[0] == "Qwen3TTS-setup", f"First lock must be 'Qwen3TTS-setup', got {lock_names[0]!r}"
    assert lock_names.count("Qwen3TTS-setup") == 1, (
        "Setup lock must be acquired exactly once (wraps only generator construction)"
    )
    assert lock_names.count("Qwen3TTS-chunk") >= 1, "Per-chunk lock must be acquired at least once during streaming"
    assert len(results) >= 1
