"""Unit tests for the optional AEC wrapper (speech_to_speech.VAD.aec).

When ``pywebrtc-audio`` is not installed, ``HAS_AEC`` is ``False``, the module
is importable with no side effects, and constructing :class:`AecProcessor`
raises a :exc:`RuntimeError` with an actionable install message.  All tests
that actually exercise the processor are skipped in that case.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.VAD.aec import HAS_AEC, AecProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SR = 16_000
_FRAME = 160  # 10 ms


def _silence(n_samples: int) -> bytes:
    return bytes(n_samples * 2)


def _pcm(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def _tone(freq: float, n_samples: int, amplitude: int = 16_000) -> bytes:
    t = np.arange(n_samples) / _SR
    sig = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return sig.tobytes()


# ---------------------------------------------------------------------------
# Import-time guard – these must pass regardless of whether the library exists
# ---------------------------------------------------------------------------


def test_module_importable_without_pywebrtc_audio() -> None:
    """The AEC module must be importable with no side effects."""
    import speech_to_speech.VAD.aec as aec_mod  # noqa: F401

    assert isinstance(aec_mod.HAS_AEC, bool)


def test_has_aec_is_bool() -> None:
    assert isinstance(HAS_AEC, bool)


# ---------------------------------------------------------------------------
# RuntimeError path (always exercisable regardless of whether library exists)
# ---------------------------------------------------------------------------


def test_constructor_raises_runtime_error_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing AecProcessor while HAS_AEC is False must raise RuntimeError."""
    import speech_to_speech.VAD.aec as aec_mod

    monkeypatch.setattr(aec_mod, "HAS_AEC", False)

    with pytest.raises(RuntimeError, match="pywebrtc-audio"):
        AecProcessor.__new__(AecProcessor).__init__()


def test_constructor_runtime_error_contains_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RuntimeError message must contain a pip install hint."""
    import speech_to_speech.VAD.aec as aec_mod

    monkeypatch.setattr(aec_mod, "HAS_AEC", False)

    with pytest.raises(RuntimeError, match="pip install"):
        AecProcessor.__new__(AecProcessor).__init__()


def test_constructor_raises_for_wrong_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    import speech_to_speech.VAD.aec as aec_mod

    monkeypatch.setattr(aec_mod, "HAS_AEC", True)
    fake_proc = type("FakeAP", (), {"process": lambda self, n, f: n})()
    fake_module = type("M", (), {"AudioProcessor": lambda **kw: fake_proc})()
    monkeypatch.setattr(aec_mod, "pywebrtc_audio", fake_module)

    with pytest.raises(ValueError, match="sample_rate"):
        AecProcessor(sample_rate=8000)


# ---------------------------------------------------------------------------
# Tests that exercise the real processor – skip cleanly when library absent
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_AEC, reason="pywebrtc-audio not installed")
class TestAecProcessorWithLibrary:
    def setup_method(self) -> None:
        self.aec = AecProcessor(sample_rate=16_000)

    def test_construction_succeeds(self) -> None:
        assert self.aec is not None

    def test_process_returns_bytes(self) -> None:
        near = _silence(_FRAME)
        far = _silence(_FRAME)
        result = self.aec.process(near, far)
        assert isinstance(result, bytes)

    def test_process_preserves_byte_length(self) -> None:
        for n in [_FRAME, _FRAME * 2, _FRAME * 3 + 80]:
            near = _tone(400.0, n)
            far = _tone(1000.0, n)
            result = self.aec.process(near, far)
            assert len(result) == len(near), f"length mismatch for {n} samples"

    def test_silence_near_returns_silence_or_small_values(self) -> None:
        """Silence near a silence far should produce near-zero output."""
        near = _silence(_FRAME * 4)
        far = _silence(_FRAME * 4)
        result = self.aec.process(near, far)
        samples = np.frombuffer(result, dtype=np.int16)
        # After convergence silence should stay silent
        assert np.max(np.abs(samples)) < 1000

    def test_buffer_accumulation_across_calls(self) -> None:
        """Sub-frame inputs should buffer and be returned in subsequent calls."""
        half_frame = _FRAME // 2
        near1 = _tone(400.0, half_frame)
        far1 = _tone(1000.0, half_frame)
        near2 = _tone(400.0, half_frame)
        far2 = _tone(1000.0, half_frame)
        # First call: not enough for a full frame
        r1 = self.aec.process(near1, far1)
        # Second call: should have processed the accumulated frame
        r2 = self.aec.process(near2, far2)
        assert len(r1) == len(near1)
        assert len(r2) == len(near2)

    def test_process_called_not_process_stream(self) -> None:
        """Verify the processor attribute has ``process``, not ``process_stream``."""
        import pywebrtc_audio as pwa  # type: ignore[import-untyped]

        proc = pwa.AudioProcessor(sample_rate=16_000, num_channels=1)
        assert hasattr(proc, "process"), "pywebrtc_audio.AudioProcessor must have .process()"
