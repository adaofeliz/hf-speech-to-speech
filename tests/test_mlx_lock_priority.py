"""Tests for the STT-priority preemption signal in mlx_lock."""

from __future__ import annotations

import threading
import time

import pytest

from speech_to_speech.utils.mlx_lock import (
    STT_PRIORITY_NON_STT_TIMEOUT,
    MLXLockContext,
    _stt_priority_requested,
    acquire_mlx_lock,
    clear_stt_priority,
    release_mlx_lock,
    request_stt_priority,
)


@pytest.fixture(autouse=True)
def reset_priority():
    _stt_priority_requested.clear()
    yield
    _stt_priority_requested.clear()


def _hold_lock_in_thread() -> tuple[threading.Event, threading.Event]:
    ready = threading.Event()
    release = threading.Event()

    def _hold():
        acquire_mlx_lock(handler_name="TestHolder")
        ready.set()
        release.wait(timeout=10.0)
        release_mlx_lock(handler_name="TestHolder")

    t = threading.Thread(target=_hold, daemon=True)
    t.start()
    ready.wait(timeout=2.0)
    return ready, release


def test_fast_fail_under_priority():
    _ready, release = _hold_lock_in_thread()
    try:
        request_stt_priority("VAD-barge-in")
        start = time.perf_counter()
        acquired = acquire_mlx_lock(timeout=2.0, handler_name="Qwen3TTS")
        elapsed = time.perf_counter() - start

        assert not acquired, "Non-Parakeet acquire must fail under priority"
        assert elapsed < STT_PRIORITY_NON_STT_TIMEOUT + 0.2, (
            f"Must fail fast (~{STT_PRIORITY_NON_STT_TIMEOUT}s), got {elapsed:.3f}s"
        )
    finally:
        release.set()


def test_parakeet_exemption():
    _ready, release = _hold_lock_in_thread()
    threading.Timer(0.3, release.set).start()
    request_stt_priority("VAD-barge-in")

    start = time.perf_counter()
    acquired = acquire_mlx_lock(timeout=1.0, handler_name="ParakeetSTT-Progressive")
    elapsed = time.perf_counter() - start

    assert acquired, "Parakeet STT must not be short-circuited by priority"
    assert elapsed >= 0.1, f"Should have waited for lock holder, got {elapsed:.3f}s"
    release_mlx_lock(handler_name="ParakeetSTT-Progressive")
    clear_stt_priority("VAD-barge-in")


def test_priority_clear_restores_timeout():
    _ready, release = _hold_lock_in_thread()
    try:
        request_stt_priority("VAD-barge-in")
        start = time.perf_counter()
        acquire_mlx_lock(timeout=2.0, handler_name="Qwen3TTS")
        capped = time.perf_counter() - start
        assert capped < 0.5, f"Under priority should fail fast, got {capped:.3f}s"

        clear_stt_priority("VAD-barge-in")

        start = time.perf_counter()
        acquire_mlx_lock(timeout=0.1, handler_name="Qwen3TTS")
        restored = time.perf_counter() - start
        assert restored >= 0.08, f"After clear, full 0.1s timeout must apply, got {restored:.3f}s"
    finally:
        release.set()


def test_multi_cycle_no_state_leak():
    for _ in range(5):
        assert not _stt_priority_requested.is_set()
        request_stt_priority("VAD-barge-in")
        assert _stt_priority_requested.is_set()
        clear_stt_priority("VAD-barge-in")
        assert not _stt_priority_requested.is_set()


def test_default_unset_zero_behavior_change():
    assert not _stt_priority_requested.is_set()
    start = time.perf_counter()
    acquired = acquire_mlx_lock(timeout=1.0, handler_name="Qwen3TTS")
    elapsed = time.perf_counter() - start

    assert acquired, "Free lock must be acquired immediately with no priority set"
    assert elapsed < 0.1, f"Must acquire instantly, got {elapsed:.3f}s"
    release_mlx_lock(handler_name="Qwen3TTS")


def test_mlxlockcontext_honors_cap():
    _ready, release = _hold_lock_in_thread()
    try:
        request_stt_priority("VAD-barge-in")
        start = time.perf_counter()
        with MLXLockContext(handler_name="Qwen3TTS", timeout=2.0) as acquired:
            elapsed = time.perf_counter() - start
            assert not acquired, "MLXLockContext must be capped under priority"
            assert elapsed < STT_PRIORITY_NON_STT_TIMEOUT + 0.2, f"Must fail fast, got {elapsed:.3f}s"
    finally:
        release.set()


def test_idempotent_request_clear():
    request_stt_priority("A")
    request_stt_priority("B")
    request_stt_priority("C")
    assert _stt_priority_requested.is_set()

    clear_stt_priority("A")
    clear_stt_priority("B")
    clear_stt_priority("C")
    assert not _stt_priority_requested.is_set()


def test_module_exports():
    import speech_to_speech.utils.mlx_lock as mlx_lock

    assert hasattr(mlx_lock, "request_stt_priority")
    assert hasattr(mlx_lock, "clear_stt_priority")
    assert hasattr(mlx_lock, "STT_PRIORITY_NON_STT_TIMEOUT")
    assert hasattr(mlx_lock, "_stt_priority_requested")
    assert hasattr(mlx_lock, "acquire_mlx_lock")
    assert hasattr(mlx_lock, "release_mlx_lock")
    assert hasattr(mlx_lock, "MLXLockContext")
