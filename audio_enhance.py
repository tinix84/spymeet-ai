"""
Audio Enhancement Pipeline — preprocessing for speech transcription.

Processing chain (sequential):
  1. Load audio (soundfile for WAV/FLAC/OGG, ffmpeg fallback for M4A/MP4/MKV/WEBM)
  2. Loudness normalization (EBU R128, target -16 LUFS)
  3. Noise reduction (spectral gating, noisereduce)
  4. Speech EQ (highpass 80 Hz + presence peak +2.5 dB @ 3 kHz)
  5. Dynamic compression (3:1 ratio, adaptive threshold)
  6. Save as 16-bit PCM WAV, 16 kHz mono

Usage:
    python audio_enhance.py --input file.m4a
    python audio_enhance.py --input ./audio/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TARGET_SR = 16000
TARGET_LUFS = -16.0

# Formats that soundfile can read natively
NATIVE_FORMATS = {".wav", ".flac", ".ogg"}
# Formats that need ffmpeg conversion first
FFMPEG_FORMATS = {".m4a", ".mp4", ".mkv", ".webm", ".mp3"}
SUPPORTED_FORMATS = NATIVE_FORMATS | FFMPEG_FORMATS


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_audio(path: Path, channel: str = "mix") -> np.ndarray:
    """Load audio file to mono float64 numpy array at TARGET_SR.

    Uses soundfile for native formats, ffmpeg subprocess for others.

    Args:
        path: Audio file path.
        channel: Channel selection for stereo files:
            "mix"   = downmix to mono (default, current behavior)
            "left"  = left channel only (mic in meeting recordings)
            "right" = right channel only (system audio in meeting recordings)
    """
    import soundfile as sf

    ext = path.suffix.lower()

    if ext in NATIVE_FORMATS:
        audio, sr = sf.read(str(path), dtype="float64")
    elif ext in FFMPEG_FORMATS:
        audio, sr = _load_via_ffmpeg(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Convert to mono if stereo
    if audio.ndim > 1:
        if channel == "left":
            audio = audio[:, 0]
        elif channel == "right":
            audio = audio[:, 1]
        else:  # "mix"
            audio = audio.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)

    return audio


def _load_via_ffmpeg(path: Path) -> tuple:
    """Decode audio file to raw PCM via ffmpeg subprocess."""
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ac", "1",                  # mono
        "-ar", str(TARGET_SR),       # target sample rate
        "-f", "f64le",               # 64-bit float little-endian
        "-acodec", "pcm_f64le",
        "pipe:1"
    ]
    result = subprocess.run(
        cmd, capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:500]}")
    audio = np.frombuffer(result.stdout, dtype=np.float64)
    return audio, TARGET_SR


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy (polyphase filter, FFT fallback)."""
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    try:
        from scipy.signal import resample_poly
        return resample_poly(audio, up, down)
    except (AttributeError, TypeError):
        # scipy/torch compatibility issue — fallback to FFT-based resampling
        from scipy.signal import resample
        new_len = int(len(audio) * target_sr / orig_sr)
        return resample(audio, new_len)


# ─── Loudness normalization ──────────────────────────────────────────────────

def normalize_loudness(audio: np.ndarray, sr: int) -> np.ndarray:
    """EBU R128 loudness normalization to TARGET_LUFS, clipped to [-1, 1]."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)

    if current_lufs == float("-inf"):
        # Silent audio — skip normalization
        return audio

    audio = pyln.normalize.loudness(audio, current_lufs, TARGET_LUFS)
    return np.clip(audio, -1.0, 1.0)


# ─── Noise reduction ────────────────────────────────────────────────────────

def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Spectral gating noise reduction using first 1s as noise profile."""
    import noisereduce as nr

    noise_clip = audio[:sr]  # first 1 second
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        stationary=True,
        prop_decrease=0.8
    )


# ─── Speech EQ ──────────────────────────────────────────────────────────────

def apply_speech_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Highpass 80 Hz (Butterworth order 4) + peaking EQ +2.5 dB at 3 kHz."""
    from scipy.signal import sosfilt, butter

    # Highpass 80 Hz — remove low-frequency rumble
    sos_hp = butter(4, 80.0, btype="high", fs=sr, output="sos")
    audio = sosfilt(sos_hp, audio)

    # Peaking EQ at 3 kHz, +2.5 dB, Q=1.0 (Audio EQ Cookbook)
    audio = _biquad_peak(audio, sr, freq=3000.0, gain_db=2.5, Q=1.0)

    return audio


def _biquad_peak(audio: np.ndarray, sr: int, freq: float, gain_db: float, Q: float) -> np.ndarray:
    """Biquad peaking EQ filter (Audio EQ Cookbook by Robert Bristow-Johnson)."""
    from scipy.signal import sosfilt

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    # Normalize and pack into second-order section
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sosfilt(sos, audio)


# ─── Dynamic compression ────────────────────────────────────────────────────

def compress_dynamic_range(audio: np.ndarray, sr: int) -> np.ndarray:
    """Feed-forward compressor: 3:1 ratio, 10ms attack, 100ms release.

    Adaptive threshold = mean RMS + 6 dB.
    Makeup gain applied via loudness re-normalization.
    """
    ratio = 3.0
    attack_samples = int(0.010 * sr)   # 10 ms
    release_samples = int(0.100 * sr)  # 100 ms

    # Compute frame-level RMS for threshold estimation
    frame_len = int(0.030 * sr)  # 30 ms frames
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return audio
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_per_frame = rms_per_frame[rms_per_frame > 1e-8]  # skip silence
    if len(rms_per_frame) == 0:
        return audio

    mean_rms = np.mean(rms_per_frame)
    threshold = mean_rms * (10 ** (6.0 / 20.0))  # +6 dB above mean RMS

    # Sample-level envelope follower + gain reduction
    envelope = np.zeros(len(audio))
    env_val = 0.0
    attack_coeff = 1.0 - np.exp(-1.0 / attack_samples)
    release_coeff = 1.0 - np.exp(-1.0 / release_samples)

    abs_audio = np.abs(audio)
    for i in range(len(audio)):
        target = abs_audio[i]
        if target > env_val:
            env_val += attack_coeff * (target - env_val)
        else:
            env_val += release_coeff * (target - env_val)
        envelope[i] = env_val

    # Compute gain reduction
    gain = np.ones(len(audio))
    mask = envelope > threshold
    gain[mask] = (threshold + (envelope[mask] - threshold) / ratio) / envelope[mask]

    audio = audio * gain

    # Makeup gain via re-normalization to target LUFS
    audio = normalize_loudness(audio, sr)
    return audio


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(audio: np.ndarray, sr: int) -> dict:
    """Compute audio quality metrics."""
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio)
    peak = np.max(np.abs(audio))
    peak_dbfs = 20 * np.log10(peak) if peak > 0 else float("-inf")

    # Estimate SNR: ratio of signal RMS to noise floor RMS (first 0.5s)
    noise_samples = audio[:sr // 2]
    noise_rms = np.sqrt(np.mean(noise_samples ** 2)) if len(noise_samples) > 0 else 1e-10
    signal_rms = np.sqrt(np.mean(audio ** 2))
    snr_db = 20 * np.log10(signal_rms / max(noise_rms, 1e-10))

    return {
        "lufs": round(lufs, 1) if lufs != float("-inf") else None,
        "peak_dbfs": round(peak_dbfs, 1) if peak_dbfs != float("-inf") else None,
        "snr_db": round(snr_db, 1),
    }


def print_metrics(label: str, metrics: dict):
    """Pretty-print audio metrics."""
    lufs = f"{metrics['lufs']:.1f} LUFS" if metrics['lufs'] is not None else "silent"
    peak = f"{metrics['peak_dbfs']:.1f} dBFS" if metrics['peak_dbfs'] is not None else "silent"
    snr = f"{metrics['snr_db']:.1f} dB"
    print(f"  {label}: loudness={lufs}  peak={peak}  SNR~{snr}")


# ─── Main enhancement pipeline ──────────────────────────────────────────────

def enhance_audio(path: Path, channel: str = "mix") -> Path:
    """Enhance a single audio file. Returns path to enhanced WAV.

    Output: [name]_enhanced.wav in the same directory as the input.

    Args:
        path: Audio file path.
        channel: Channel selection passed to load_audio ("mix", "left", "right").
    """
    path = Path(path)
    output_path = path.with_name(f"{path.stem}_enhanced.wav")

    print(f"  [enhance] Loading {path.name}...")
    audio = load_audio(path, channel=channel)
    duration_s = len(audio) / TARGET_SR
    mm, ss = divmod(int(duration_s), 60)
    print(f"  [enhance] Loaded: {mm}m{ss:02d}s, {TARGET_SR} Hz mono")

    before = compute_metrics(audio, TARGET_SR)
    print_metrics("[enhance] BEFORE", before)

    print(f"  [enhance] Step 1/4: Loudness normalization (EBU R128, {TARGET_LUFS} LUFS)...", end="", flush=True)
    audio = normalize_loudness(audio, TARGET_SR)
    print(" done")

    print(f"  [enhance] Step 2/4: Noise reduction (spectral gating)...", end="", flush=True)
    audio = reduce_noise(audio, TARGET_SR)
    print(" done")

    print(f"  [enhance] Step 3/4: Speech EQ (HP 80Hz + peak 3kHz)...", end="", flush=True)
    audio = apply_speech_eq(audio, TARGET_SR)
    print(" done")

    print(f"  [enhance] Step 4/4: Dynamic compression (3:1)...", end="", flush=True)
    audio = compress_dynamic_range(audio, TARGET_SR)
    print(" done")

    after = compute_metrics(audio, TARGET_SR)
    print_metrics("[enhance] AFTER ", after)

    # Save as 16-bit PCM WAV
    import soundfile as sf
    # Clip and convert to int16 range
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(output_path), audio, TARGET_SR, subtype="PCM_16")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [enhance] Saved: {output_path.name} ({size_mb:.1f} MB)")

    return output_path


def enhance_audio_files(audio_files: list, channel: str = "mix") -> tuple:
    """Enhance a batch of audio files. Returns (enhanced_files, stem_map).

    - Skips enhancement if _enhanced.wav already exists and is newer than source.
    - Falls back to original file on error.
    - stem_map: dict mapping enhanced Path -> original stem (for transcript naming).

    Args:
        audio_files: List of audio file paths.
        channel: Channel selection passed to load_audio ("mix", "left", "right").
    """
    enhanced = []
    stem_map = {}

    for path in audio_files:
        path = Path(path)
        output_path = path.with_name(f"{path.stem}_enhanced.wav")

        # Skip already-enhanced files passed as input
        if path.stem.endswith("_enhanced"):
            enhanced.append(path)
            stem_map[path] = path.stem.replace("_enhanced", "")
            continue

        # Skip if enhanced file is up-to-date (only for default "mix" channel)
        if channel == "mix" and output_path.exists() and output_path.stat().st_mtime >= path.stat().st_mtime:
            print(f"  [enhance] Skipping {path.name} — enhanced file is up-to-date")
            enhanced.append(output_path)
            stem_map[output_path] = path.stem
            continue

        try:
            result = enhance_audio(path, channel=channel)
            enhanced.append(result)
            stem_map[result] = path.stem
        except Exception as e:
            print(f"  [enhance WARNING] {path.name}: {e}")
            print(f"  [enhance] Falling back to original file")
            enhanced.append(path)
            # No stem_map entry needed — original path keeps its own stem

    return enhanced, stem_map


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audio enhancement for speech transcription",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True,
                        help="Audio file or folder to enhance")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Not found: {input_path}")
        return

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_FORMATS:
            print(f"[ERROR] Unsupported format: {input_path.suffix}")
            print(f"  Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
            return
        files = [input_path]
    else:
        files = [f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]
        # Exclude already-enhanced files
        files = [f for f in files if not f.stem.endswith("_enhanced")]
        if not files:
            print(f"[ERROR] No audio files found in {input_path}")
            return

    print(f"[INFO] Enhancing {len(files)} file(s)")
    t0 = time.time()

    for f in sorted(files):
        print(f"\n[PROCESSING] {f.name}")
        try:
            enhance_audio(f)
        except Exception as e:
            print(f"  [ERROR] {e}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Enhancement completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
