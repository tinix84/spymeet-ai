"""Tests for audio_enhance.py — each processing step + full pipeline.

Uses synthetic audio from conftest.py fixtures. Advanced assertions include
spectral analysis (FFT), statistical comparisons, and file-format verification.

Note: noisereduce depends on torch. Tests that need it are marked with
@requires_noisereduce. Full-pipeline tests use patch_reduce_noise fixture
to work in either environment.
"""

import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

from conftest import requires_noisereduce

# Module under test
import audio_enhance as ae


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOADING
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadAudio:
    """Tests for load_audio(), _load_via_ffmpeg(), _resample()."""

    def test_load_wav_native(self, wav_file, sr):
        """Native WAV loads correctly: mono, float64, correct sample count."""
        audio = ae.load_audio(wav_file)
        assert audio.ndim == 1, "Must be mono (1D)"
        assert audio.dtype == np.float64
        # 3 seconds at 16 kHz = 48000 samples
        assert len(audio) == 3 * sr

    def test_load_stereo_to_mono(self, stereo_wav_file, sr):
        """Stereo input is averaged down to mono."""
        audio = ae.load_audio(stereo_wav_file)
        assert audio.ndim == 1
        # Check it's the mean of channels (not just left or right)
        raw, _ = sf.read(str(stereo_wav_file), dtype="float64")
        expected_mono = raw.mean(axis=1)
        np.testing.assert_allclose(audio, expected_mono, atol=1e-4)

    def test_load_resamples_to_16khz(self, wav_44100, sr):
        """44.1 kHz WAV is resampled to 16 kHz."""
        audio = ae.load_audio(wav_44100)
        expected_samples = int(2.0 * sr)  # 2 seconds at 16 kHz
        # Allow +-1 sample tolerance from resampling math
        assert abs(len(audio) - expected_samples) <= 1
        assert audio.dtype == np.float64

    def test_load_unsupported_format_raises(self, tmp_path):
        """Unsupported extension raises ValueError."""
        fake = tmp_path / "test.xyz"
        fake.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported format"):
            ae.load_audio(fake)

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Missing file raises an error."""
        missing = tmp_path / "ghost.wav"
        with pytest.raises(Exception):  # could be FileNotFoundError or RuntimeError
            ae.load_audio(missing)


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOUDNESS NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalizeLoudness:
    """Tests for normalize_loudness() — EBU R128 to -16 LUFS."""

    def test_loud_signal_brought_down(self, loud_signal, sr):
        """A near-clipping signal should be attenuated toward -16 LUFS."""
        meter = pyln.Meter(sr)
        before_lufs = meter.integrated_loudness(loud_signal)
        result = ae.normalize_loudness(loud_signal, sr)
        after_lufs = meter.integrated_loudness(result)

        assert before_lufs > ae.TARGET_LUFS, "Precondition: input is louder than target"
        assert after_lufs == pytest.approx(ae.TARGET_LUFS, abs=1.0)

    def test_quiet_signal_brought_up(self, quiet_signal, sr):
        """A very quiet signal should be amplified toward -16 LUFS."""
        meter = pyln.Meter(sr)
        before_lufs = meter.integrated_loudness(quiet_signal)
        result = ae.normalize_loudness(quiet_signal, sr)
        after_lufs = meter.integrated_loudness(result)

        assert before_lufs < ae.TARGET_LUFS - 10, "Precondition: input much quieter than target"
        assert after_lufs == pytest.approx(ae.TARGET_LUFS, abs=1.0)

    def test_output_clipped_to_unit_range(self, loud_signal, sr):
        """Output must never exceed [-1, 1]."""
        result = ae.normalize_loudness(loud_signal, sr)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_silent_audio_passthrough(self, silent_signal, sr):
        """All-zeros input should pass through unchanged (no division by zero)."""
        result = ae.normalize_loudness(silent_signal, sr)
        np.testing.assert_array_equal(result, silent_signal)

    def test_preserves_dtype_and_length(self, sine_440hz, sr):
        """Output has same dtype and length as input."""
        result = ae.normalize_loudness(sine_440hz, sr)
        assert result.dtype == sine_440hz.dtype
        assert len(result) == len(sine_440hz)


# ═════════════════════════════════════════════════════════════════════════════
# 3. NOISE REDUCTION (requires torch via noisereduce)
# ═════════════════════════════════════════════════════════════════════════════

@requires_noisereduce
class TestReduceNoise:
    """Tests for reduce_noise() — spectral gating. Skipped if torch unavailable."""

    def test_reduces_noise_floor(self, noisy_speech, sr):
        """Noise power in the first 1s (noise-only region) should decrease."""
        result = ae.reduce_noise(noisy_speech, sr)
        # RMS of the noise-only region (first 0.5s, safely inside the 1s profile)
        before_rms = np.sqrt(np.mean(noisy_speech[:sr // 2] ** 2))
        after_rms = np.sqrt(np.mean(result[:sr // 2] ** 2))
        assert after_rms < before_rms * 0.5, (
            f"Noise RMS should drop by >50%: before={before_rms:.4f}, after={after_rms:.4f}"
        )

    def test_preserves_signal_energy(self, noisy_speech, sr):
        """Signal region (after 1s) should retain most of its energy."""
        result = ae.reduce_noise(noisy_speech, sr)
        signal_slice = slice(2 * sr, 4 * sr)  # 2-4s region, well into speech
        before_energy = np.sum(noisy_speech[signal_slice] ** 2)
        after_energy = np.sum(result[signal_slice] ** 2)
        # Should retain at least 50% of signal energy
        assert after_energy > before_energy * 0.5

    def test_output_shape_and_dtype(self, noisy_speech, sr):
        """Shape and dtype preserved."""
        result = ae.reduce_noise(noisy_speech, sr)
        assert result.shape == noisy_speech.shape
        assert result.dtype == noisy_speech.dtype

    def test_pure_noise_heavily_attenuated(self, white_noise, sr):
        """Pure white noise should be substantially reduced."""
        result = ae.reduce_noise(white_noise, sr)
        before_rms = np.sqrt(np.mean(white_noise ** 2))
        after_rms = np.sqrt(np.mean(result ** 2))
        assert after_rms < before_rms * 0.6


# ═════════════════════════════════════════════════════════════════════════════
# 4. SPEECH EQ
# ═════════════════════════════════════════════════════════════════════════════

class TestSpeechEQ:
    """Tests for apply_speech_eq() — HP 80 Hz + peak 3 kHz."""

    def test_attenuates_low_frequency(self, sr):
        """A 40 Hz tone (below 80 Hz cutoff) should be heavily attenuated."""
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        low_tone = (0.5 * np.sin(2 * np.pi * 40 * t)).astype(np.float64)

        result = ae.apply_speech_eq(low_tone, sr)
        before_rms = np.sqrt(np.mean(low_tone ** 2))
        after_rms = np.sqrt(np.mean(result ** 2))
        attenuation_db = 20 * np.log10(after_rms / before_rms)
        # 80 Hz HP order 4: a 40 Hz tone should be attenuated by at least 12 dB
        assert attenuation_db < -12, f"40 Hz attenuation only {attenuation_db:.1f} dB"

    def test_boosts_presence_region(self, sr):
        """A 3 kHz tone should gain ~2.5 dB from the peaking EQ."""
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        presence_tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float64)

        result = ae.apply_speech_eq(presence_tone, sr)
        before_rms = np.sqrt(np.mean(presence_tone ** 2))
        after_rms = np.sqrt(np.mean(result ** 2))
        gain_db = 20 * np.log10(after_rms / before_rms)
        # Should be boosted by roughly +2.5 dB (allow +-1 dB tolerance)
        assert 1.0 < gain_db < 4.0, f"3 kHz gain was {gain_db:.1f} dB, expected ~2.5"

    def test_passband_mostly_unchanged(self, sr):
        """A 1 kHz tone (mid-band) should pass through with minimal change."""
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        mid_tone = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float64)

        result = ae.apply_speech_eq(mid_tone, sr)
        before_rms = np.sqrt(np.mean(mid_tone ** 2))
        after_rms = np.sqrt(np.mean(result ** 2))
        change_db = abs(20 * np.log10(after_rms / before_rms))
        assert change_db < 1.5, f"1 kHz changed by {change_db:.1f} dB, should be <1.5"

    def test_spectral_shape_verification(self, sr):
        """FFT-based check: 50 Hz tone attenuated, 3 kHz tone boosted.

        Uses a signal with explicit low-frequency and presence-region components
        to verify the EQ shape via spectral analysis.
        """
        t = np.linspace(0, 5.0, 5 * sr, endpoint=False)
        # Build signal with known spectral peaks
        signal = (
            0.30 * np.sin(2 * np.pi * 50 * t)    # below HP cutoff — should be cut
            + 0.20 * np.sin(2 * np.pi * 300 * t)  # speech fundamental — passband
            + 0.10 * np.sin(2 * np.pi * 3000 * t) # presence region — should be boosted
        ).astype(np.float64)

        result = ae.apply_speech_eq(signal, sr)

        before_fft = np.abs(np.fft.rfft(signal))
        after_fft = np.abs(np.fft.rfft(result))
        freqs = np.fft.rfftfreq(len(signal), 1 / sr)

        # Find the FFT bin closest to 50 Hz — energy should decrease
        idx_50 = np.argmin(np.abs(freqs - 50))
        assert after_fft[idx_50] < before_fft[idx_50] * 0.5, (
            f"50 Hz bin not attenuated enough: before={before_fft[idx_50]:.1f}, after={after_fft[idx_50]:.1f}"
        )

        # Find the FFT bin closest to 3 kHz — energy should increase
        idx_3k = np.argmin(np.abs(freqs - 3000))
        assert after_fft[idx_3k] > before_fft[idx_3k] * 1.1, (
            f"3 kHz bin not boosted: before={before_fft[idx_3k]:.1f}, after={after_fft[idx_3k]:.1f}"
        )

    def test_preserves_shape_and_dtype(self, sine_440hz, sr):
        result = ae.apply_speech_eq(sine_440hz, sr)
        assert result.shape == sine_440hz.shape
        assert result.dtype == sine_440hz.dtype


# ═════════════════════════════════════════════════════════════════════════════
# 5. DYNAMIC COMPRESSION
# ═════════════════════════════════════════════════════════════════════════════

class TestCompressDynamicRange:
    """Tests for compress_dynamic_range() — 3:1 feed-forward compressor."""

    def test_reduces_dynamic_range(self, varying_amplitude_signal, sr):
        """Loud/quiet ratio should shrink after compression."""
        result = ae.compress_dynamic_range(varying_amplitude_signal, sr)

        # Measure RMS of loud vs quiet halves
        half = len(varying_amplitude_signal) // 2
        loud_rms_before = np.sqrt(np.mean(varying_amplitude_signal[:half] ** 2))
        quiet_rms_before = np.sqrt(np.mean(varying_amplitude_signal[half:] ** 2))
        ratio_before = loud_rms_before / max(quiet_rms_before, 1e-10)

        loud_rms_after = np.sqrt(np.mean(result[:half] ** 2))
        quiet_rms_after = np.sqrt(np.mean(result[half:] ** 2))
        ratio_after = loud_rms_after / max(quiet_rms_after, 1e-10)

        assert ratio_after < ratio_before, (
            f"Dynamic range not reduced: before={ratio_before:.1f}x, after={ratio_after:.1f}x"
        )

    def test_silent_audio_passthrough(self, silent_signal, sr):
        """All-zeros input passes through unchanged."""
        result = ae.compress_dynamic_range(silent_signal, sr)
        np.testing.assert_array_equal(result, silent_signal)

    def test_very_short_audio(self, sr):
        """Audio shorter than one analysis frame should pass through."""
        short = np.array([0.1, -0.1, 0.05], dtype=np.float64)
        result = ae.compress_dynamic_range(short, sr)
        assert len(result) == len(short)

    def test_output_bounded(self, varying_amplitude_signal, sr):
        """Compressed + re-normalized output must stay in [-1, 1]."""
        result = ae.compress_dynamic_range(varying_amplitude_signal, sr)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_no_nan_or_inf_on_step_signal(self, sr):
        """Compressor handles a sudden onset without NaN/Inf."""
        audio = np.zeros(sr, dtype=np.float64)
        audio[sr // 2:] = 0.8  # step at 0.5s
        result = ae.compress_dynamic_range(audio, sr)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ═════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    """Tests for compute_metrics() — LUFS, peak dBFS, SNR."""

    def test_returns_expected_keys(self, sine_440hz, sr):
        m = ae.compute_metrics(sine_440hz, sr)
        assert set(m.keys()) == {"lufs", "peak_dbfs", "snr_db"}

    def test_lufs_reasonable_for_known_signal(self, sine_440hz, sr):
        """A -20 dBFS sine should measure around -23 to -17 LUFS."""
        m = ae.compute_metrics(sine_440hz, sr)
        assert m["lufs"] is not None
        assert -25 < m["lufs"] < -15

    def test_peak_dbfs_matches_amplitude(self, sr):
        """A sine at amplitude 0.5 should have peak ~-6 dBFS."""
        t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
        m = ae.compute_metrics(audio, sr)
        assert m["peak_dbfs"] == pytest.approx(-6.0, abs=0.5)

    def test_silent_audio_metrics(self, silent_signal, sr):
        """Silent audio returns None for LUFS and peak."""
        m = ae.compute_metrics(silent_signal, sr)
        assert m["lufs"] is None
        assert m["peak_dbfs"] is None

    def test_snr_higher_when_noise_floor_is_lower(self, sr):
        """SNR metric reflects noise floor level in the first 0.5s.

        compute_metrics estimates SNR as signal_rms / noise_rms(first 0.5s).
        Signal with quiet noise-only intro → higher SNR than noisy intro.
        """
        duration = 3.0
        n = int(duration * sr)
        t = np.linspace(0, duration, n, endpoint=False)

        # High SNR: quiet noise-only first 0.5s, then speech
        rng = np.random.default_rng(42)
        high_snr = np.zeros(n, dtype=np.float64)
        high_snr[:sr // 2] = rng.standard_normal(sr // 2) * 0.005  # very low noise
        high_snr[sr:] = 0.3 * np.sin(2 * np.pi * 300 * t[sr:])    # speech after 1s

        # Low SNR: loud noise-only first 0.5s, then same speech
        low_snr = np.zeros(n, dtype=np.float64)
        low_snr[:sr // 2] = rng.standard_normal(sr // 2) * 0.1  # loud noise
        low_snr[sr:] = 0.3 * np.sin(2 * np.pi * 300 * t[sr:])  # same speech

        high_m = ae.compute_metrics(high_snr, sr)
        low_m = ae.compute_metrics(low_snr, sr)
        assert high_m["snr_db"] > low_m["snr_db"], (
            f"Expected high_snr ({high_m['snr_db']}) > low_snr ({low_m['snr_db']})"
        )

    def test_all_values_are_rounded(self, sine_440hz, sr):
        """All numeric values should be rounded to 1 decimal."""
        m = ae.compute_metrics(sine_440hz, sr)
        for key in ("lufs", "peak_dbfs", "snr_db"):
            if m[key] is not None:
                assert m[key] == round(m[key], 1)


# ═════════════════════════════════════════════════════════════════════════════
# 7. FULL PIPELINE — enhance_audio()
# ═════════════════════════════════════════════════════════════════════════════

class TestEnhanceAudio:
    """End-to-end tests for enhance_audio() — single file.

    Uses patch_reduce_noise fixture so tests pass even without torch.
    """

    def test_creates_enhanced_wav(self, wav_file, patch_reduce_noise):
        """Output file [name]_enhanced.wav is created."""
        result = ae.enhance_audio(wav_file)
        assert result.exists()
        assert result.name == "test_input_enhanced.wav"
        assert result.stat().st_size > 0

    def test_output_format_16bit_16khz_mono(self, wav_file, patch_reduce_noise):
        """Enhanced WAV is 16-bit PCM, 16 kHz, mono."""
        result = ae.enhance_audio(wav_file)
        info = sf.info(str(result))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.subtype == "PCM_16"

    def test_output_loudness_near_target(self, wav_file, sr, patch_reduce_noise):
        """Enhanced audio should be close to -16 LUFS."""
        result_path = ae.enhance_audio(wav_file)
        audio, _ = sf.read(str(result_path), dtype="float64")
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(audio)
        assert lufs == pytest.approx(ae.TARGET_LUFS, abs=3.0)

    def test_output_not_clipped(self, wav_file, patch_reduce_noise):
        """No samples should exceed [-1, 1] (within int16 precision)."""
        result_path = ae.enhance_audio(wav_file)
        audio, _ = sf.read(str(result_path), dtype="float64")
        assert np.max(np.abs(audio)) <= 1.0

    def test_returns_path_object(self, wav_file, patch_reduce_noise):
        result = ae.enhance_audio(wav_file)
        assert isinstance(result, Path)

    def test_same_directory_as_input(self, wav_file, patch_reduce_noise):
        result = ae.enhance_audio(wav_file)
        assert result.parent == wav_file.parent


# ═════════════════════════════════════════════════════════════════════════════
# 8. BATCH — enhance_audio_files()
# ═════════════════════════════════════════════════════════════════════════════

class TestEnhanceAudioFiles:
    """Tests for enhance_audio_files() — batch, caching, fallback, stem_map."""

    def test_batch_returns_correct_count(self, tmp_path, sine_440hz, sr, patch_reduce_noise):
        """Returns one enhanced file per input."""
        files = []
        for i in range(3):
            p = tmp_path / f"audio_{i}.wav"
            sf.write(str(p), sine_440hz, sr, subtype="PCM_16")
            files.append(p)

        enhanced, stem_map = ae.enhance_audio_files(files)
        assert len(enhanced) == 3
        assert all(p.exists() for p in enhanced)

    def test_stem_map_maps_to_original_names(self, tmp_path, sine_440hz, sr, patch_reduce_noise):
        """stem_map values are original stems (not _enhanced)."""
        p = tmp_path / "meeting_2024.wav"
        sf.write(str(p), sine_440hz, sr, subtype="PCM_16")

        enhanced, stem_map = ae.enhance_audio_files([p])
        enhanced_path = enhanced[0]
        assert enhanced_path.stem == "meeting_2024_enhanced"
        assert stem_map[enhanced_path] == "meeting_2024"

    def test_caching_skips_uptodate(self, tmp_path, sine_440hz, sr, patch_reduce_noise):
        """Second call skips enhancement if _enhanced.wav is newer."""
        p = tmp_path / "cached.wav"
        sf.write(str(p), sine_440hz, sr, subtype="PCM_16")

        # First run — creates _enhanced.wav
        enhanced1, _ = ae.enhance_audio_files([p])
        mtime1 = enhanced1[0].stat().st_mtime

        # Tiny delay to ensure mtime would differ if re-written
        time.sleep(0.05)

        # Second run — should skip
        enhanced2, stem_map2 = ae.enhance_audio_files([p])
        mtime2 = enhanced2[0].stat().st_mtime

        assert mtime1 == mtime2, "File should NOT have been re-created"
        assert stem_map2[enhanced2[0]] == "cached"

    def test_fallback_on_error(self, tmp_path):
        """If enhancement fails, original path is returned."""
        fake = tmp_path / "broken.wav"
        fake.write_text("not a wav")  # will fail to load

        enhanced, stem_map = ae.enhance_audio_files([fake])
        assert len(enhanced) == 1
        assert enhanced[0] == fake  # fell back to original
        assert fake not in stem_map  # no stem_map entry for fallback

    def test_already_enhanced_file_skipped(self, tmp_path, sine_440hz, sr):
        """Input named _enhanced.wav is passed through with correct stem_map."""
        p = tmp_path / "meeting_enhanced.wav"
        sf.write(str(p), sine_440hz, sr, subtype="PCM_16")

        enhanced, stem_map = ae.enhance_audio_files([p])
        assert enhanced[0] == p
        assert stem_map[p] == "meeting"  # strips _enhanced

    def test_mixed_batch_some_cached_some_new(self, tmp_path, sine_440hz, sr, patch_reduce_noise):
        """Mix of cached and new files handled correctly."""
        # File A — pre-enhanced
        a = tmp_path / "file_a.wav"
        sf.write(str(a), sine_440hz, sr, subtype="PCM_16")
        ae.enhance_audio(a)  # creates file_a_enhanced.wav

        # File B — new
        b = tmp_path / "file_b.wav"
        sf.write(str(b), sine_440hz, sr, subtype="PCM_16")

        enhanced, stem_map = ae.enhance_audio_files([a, b])
        assert len(enhanced) == 2
        assert stem_map[enhanced[0]] == "file_a"
        assert stem_map[enhanced[1]] == "file_b"


# ═════════════════════════════════════════════════════════════════════════════
# 9. RESAMPLING
# ═════════════════════════════════════════════════════════════════════════════

class TestResample:
    """Tests for _resample() helper."""

    def test_downsample_44100_to_16000(self):
        """Downsample preserves duration (within 1 sample)."""
        orig_sr = 44100
        target_sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(orig_sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)

        result = ae._resample(audio, orig_sr, target_sr)
        expected_len = int(target_sr * duration)
        assert abs(len(result) - expected_len) <= 1

    def test_frequency_content_preserved(self):
        """A 440 Hz tone remains at 440 Hz after resampling."""
        orig_sr = 44100
        target_sr = 16000
        t = np.linspace(0, 1.0, orig_sr, endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float64)

        result = ae._resample(audio, orig_sr, target_sr)
        fft = np.abs(np.fft.rfft(result))
        freqs = np.fft.rfftfreq(len(result), 1 / target_sr)
        peak_freq = freqs[np.argmax(fft)]
        assert peak_freq == pytest.approx(440, abs=5)

    def test_identity_when_same_rate(self):
        """No-op when source and target rate are equal."""
        audio = np.random.default_rng(0).standard_normal(16000).astype(np.float64)
        result = ae._resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)


# ═════════════════════════════════════════════════════════════════════════════
# 10. BIQUAD PEAK EQ
# ═════════════════════════════════════════════════════════════════════════════

class TestBiquadPeak:
    """Tests for _biquad_peak() — Audio EQ Cookbook correctness."""

    def test_zero_gain_is_identity(self, sine_440hz, sr):
        """0 dB gain should leave signal unchanged."""
        result = ae._biquad_peak(sine_440hz, sr, freq=3000.0, gain_db=0.0, Q=1.0)
        np.testing.assert_allclose(result, sine_440hz, atol=1e-10)

    def test_positive_gain_boosts(self, sr):
        """Positive gain at target frequency increases amplitude."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float64)
        result = ae._biquad_peak(tone, sr, freq=3000.0, gain_db=6.0, Q=1.0)
        assert np.sqrt(np.mean(result ** 2)) > np.sqrt(np.mean(tone ** 2))

    def test_negative_gain_cuts(self, sr):
        """Negative gain at target frequency decreases amplitude."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 3000 * t)).astype(np.float64)
        result = ae._biquad_peak(tone, sr, freq=3000.0, gain_db=-6.0, Q=1.0)
        assert np.sqrt(np.mean(result ** 2)) < np.sqrt(np.mean(tone ** 2))

    def test_off_center_frequency_less_affected(self, sr):
        """A tone far from center frequency is barely affected by the peak."""
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone_500 = (0.3 * np.sin(2 * np.pi * 500 * t)).astype(np.float64)
        result = ae._biquad_peak(tone_500, sr, freq=3000.0, gain_db=6.0, Q=1.0)
        change_db = 20 * np.log10(
            np.sqrt(np.mean(result ** 2)) / np.sqrt(np.mean(tone_500 ** 2))
        )
        assert abs(change_db) < 1.0, f"500 Hz changed by {change_db:.2f} dB, expected <1"
