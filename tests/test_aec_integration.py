"""FFT-based integration test for AecProcessor.

Synthesises a 1 kHz far-end (speaker/TTS) tone and a near-end signal that is
the mix of an attenuated, slightly-delayed copy of that 1 kHz reference and a
distinct 400 Hz "user speech" component.  After AEC processing the 1 kHz
energy in the output should be measurably lower than in the raw near-end, while
the 400 Hz energy should be mostly preserved.

Skipped when ``pywebrtc-audio`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from speech_to_speech.VAD.aec import HAS_AEC, AecProcessor

_SR = 16_000
_FRAME = 160  # 10 ms
_N_FRAMES = 50  # 500 ms of audio – enough for AEC3 to converge


def _make_tone(freq: float, n: int, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / _SR
    return (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def _fft_magnitude(pcm_bytes: bytes, target_freq: float, bin_width: int = 3) -> float:
    """Return the RMS FFT magnitude around *target_freq* (±bin_width bins)."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(n, d=1.0 / _SR)
    idx = int(np.argmin(np.abs(freqs - target_freq)))
    lo, hi = max(0, idx - bin_width), min(len(spectrum), idx + bin_width + 1)
    return float(np.mean(spectrum[lo:hi]))


@pytest.mark.skipif(not HAS_AEC, reason="pywebrtc-audio not installed")
def test_aec_attenuates_far_end_frequency() -> None:
    """AEC should reduce the 1 kHz echo while leaving 400 Hz mostly intact."""
    n_total = _FRAME * _N_FRAMES
    delay = 8  # samples of acoustic delay

    far_signal = _make_tone(1000.0, n_total, amplitude=0.8)
    user_speech = _make_tone(400.0, n_total, amplitude=0.6)
    echo = np.zeros(n_total, dtype=np.int16)
    echo[delay:] = (far_signal[: n_total - delay].astype(np.float32) * 0.4).clip(-32768, 32767).astype(np.int16)

    near_signal = np.clip(user_speech.astype(np.int32) + echo.astype(np.int32), -32768, 32767).astype(np.int16)

    aec = AecProcessor(sample_rate=_SR)

    near_raw_chunks: list[bytes] = []
    aec_out_chunks: list[bytes] = []

    for i in range(_N_FRAMES):
        s = i * _FRAME
        e = s + _FRAME
        near_chunk = near_signal[s:e].tobytes()
        far_chunk = far_signal[s:e].tobytes()
        near_raw_chunks.append(near_chunk)
        aec_out_chunks.append(aec.process(near_chunk, far_chunk))

    # Use only the second half (frames 25-50) by which point AEC should have converged.
    half = _N_FRAMES // 2
    raw_bytes = b"".join(near_raw_chunks[half:])
    aec_bytes = b"".join(aec_out_chunks[half:])

    raw_1k = _fft_magnitude(raw_bytes, 1000.0)
    aec_1k = _fft_magnitude(aec_bytes, 1000.0)

    raw_400 = _fft_magnitude(raw_bytes, 400.0)
    aec_400 = _fft_magnitude(aec_bytes, 400.0)

    # 1 kHz (echo) should be at least 10 % lower after AEC
    assert aec_1k < raw_1k * 0.90, f"Expected AEC to reduce 1 kHz energy by >10 %; raw={raw_1k:.1f} aec={aec_1k:.1f}"

    # 400 Hz (user speech) should retain at least 30 % of its original energy
    assert aec_400 > raw_400 * 0.30, (
        f"Expected AEC to preserve >30 % of 400 Hz energy; raw={raw_400:.1f} aec={aec_400:.1f}"
    )


@pytest.mark.skipif(not HAS_AEC, reason="pywebrtc-audio not installed")
def test_aec_output_length_matches_input() -> None:
    """Output byte length must equal the input near-end byte length."""
    n = _FRAME * 10
    aec = AecProcessor(sample_rate=_SR)
    near = _make_tone(400.0, n).tobytes()
    far = _make_tone(1000.0, n).tobytes()
    result = aec.process(near, far)
    assert len(result) == len(near)
