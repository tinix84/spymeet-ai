"""Shared fixtures for audio enhancement tests.

All audio is synthetically generated — no real files needed.
"""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

SR = 16000  # match audio_enhance.TARGET_SR

# ─── noisereduce availability ────────────────────────────────────────────────
try:
    import noisereduce  # noqa: F401
    NOISEREDUCE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NOISEREDUCE_AVAILABLE = False

requires_noisereduce = pytest.mark.skipif(
    not NOISEREDUCE_AVAILABLE,
    reason="noisereduce requires torch (not available in base env)"
)


@pytest.fixture
def patch_reduce_noise():
    """Patch reduce_noise to be a passthrough when noisereduce/torch is missing.

    Use this in tests that call the full pipeline but don't need real noise
    reduction (e.g. file I/O, format, caching tests).
    """
    if NOISEREDUCE_AVAILABLE:
        yield  # no-op, real reduce_noise works
    else:
        import audio_enhance as ae
        with patch.object(ae, "reduce_noise", side_effect=lambda audio, sr: audio):
            yield


@pytest.fixture
def sr():
    """Standard sample rate used throughout the pipeline."""
    return SR


# ─── Synthetic audio generators ──────────────────────────────────────────────

@pytest.fixture
def sine_440hz(sr):
    """3-second 440 Hz sine wave at -20 dBFS (typical speech level)."""
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    amplitude = 10 ** (-20 / 20)  # -20 dBFS
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


@pytest.fixture
def white_noise(sr):
    """3 seconds of white noise, seeded for reproducibility."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(3 * sr) * 0.05).astype(np.float64)


@pytest.fixture
def speech_like_signal(sr):
    """Synthetic 'speech-like' signal: 300 Hz fundamental + harmonics + noise.

    Mimics voiced speech with a low-frequency fundamental, formant-like
    harmonics, and light background noise.
    """
    t = np.linspace(0, 5.0, 5 * sr, endpoint=False)
    # Fundamental + harmonics (speech-like spectral content)
    signal = (
        0.30 * np.sin(2 * np.pi * 300 * t)   # F0
        + 0.15 * np.sin(2 * np.pi * 600 * t)  # H2
        + 0.10 * np.sin(2 * np.pi * 1200 * t) # H4
        + 0.08 * np.sin(2 * np.pi * 3000 * t) # presence region
    )
    # Add light noise floor
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(len(t)) * 0.01
    return (signal + noise).astype(np.float64)


@pytest.fixture
def loud_signal(sr):
    """Near-clipping 440 Hz sine for testing normalization brings it down."""
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    return (0.95 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


@pytest.fixture
def quiet_signal(sr):
    """Very quiet 440 Hz sine (-50 dBFS) for testing normalization brings it up."""
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    amplitude = 10 ** (-50 / 20)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


@pytest.fixture
def silent_signal(sr):
    """Pure silence — edge case."""
    return np.zeros(3 * sr, dtype=np.float64)


@pytest.fixture
def stereo_signal(sr):
    """Stereo WAV data (2D array, 2 channels)."""
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    return np.column_stack([left, right]).astype(np.float64)


@pytest.fixture
def varying_amplitude_signal(sr):
    """Signal with alternating loud/quiet sections for compression testing."""
    t = np.linspace(0, 4.0, 4 * sr, endpoint=False)
    # Loud section (first 2s) → quiet section (last 2s)
    envelope = np.ones(len(t))
    envelope[:2 * sr] = 0.8   # loud
    envelope[2 * sr:] = 0.05  # quiet
    return (envelope * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


@pytest.fixture
def noisy_speech(sr):
    """Speech-like signal buried in noise (low SNR ~6 dB). First 1s is noise-only."""
    rng = np.random.default_rng(99)
    duration = 5.0
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)

    # Background noise (constant throughout)
    noise = rng.standard_normal(n) * 0.05

    # Speech starts at 1s (so first 1s is pure noise for noise profile)
    speech = np.zeros(n)
    speech_start = sr  # 1 second in
    speech_t = t[speech_start:]
    speech[speech_start:] = (
        0.20 * np.sin(2 * np.pi * 300 * speech_t)
        + 0.10 * np.sin(2 * np.pi * 600 * speech_t)
    )
    return (speech + noise).astype(np.float64)


# ─── File-based fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def wav_file(tmp_path, sine_440hz, sr):
    """Write a 16kHz mono WAV to temp dir, return Path."""
    p = tmp_path / "test_input.wav"
    sf.write(str(p), sine_440hz, sr, subtype="PCM_16")
    return p


@pytest.fixture
def stereo_wav_file(tmp_path, stereo_signal, sr):
    """Stereo WAV for mono-conversion testing."""
    p = tmp_path / "test_stereo.wav"
    sf.write(str(p), stereo_signal, sr, subtype="PCM_16")
    return p


@pytest.fixture
def wav_44100(tmp_path, sr):
    """WAV at 44100 Hz — tests resampling to 16 kHz."""
    orig_sr = 44100
    t = np.linspace(0, 2.0, 2 * orig_sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
    p = tmp_path / "test_44100.wav"
    sf.write(str(p), audio, orig_sr, subtype="PCM_16")
    return p
