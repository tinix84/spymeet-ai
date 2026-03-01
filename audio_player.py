"""
Audio Player — playback engine for SpyMeet Recorder widget.

Plays WAV and M4A files via PyAudioWPatch output stream on a daemon thread.
Software volume control. Separate PyAudio instance from Recorder.

Usage:
    from audio_player import AudioPlayer
    player = AudioPlayer()
    files = AudioPlayer.list_audio_files(Path("./audio"))
    player.load(files[0]["path"])
    player.play()
"""

import os
import subprocess
import threading
from pathlib import Path

import numpy as np


def _find_ffprobe() -> str:
    """Find ffprobe binary — check conda env path first, then PATH."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidate = Path(conda_prefix) / "Library" / "bin" / "ffprobe.exe"
        if candidate.exists():
            return str(candidate)
    # Also check user's home conda dir
    home = Path.home()
    for env_name in ("social_env",):
        for base in (home / ".conda" / "envs", home / "anaconda3" / "envs",
                     Path("C:/ProgramData/anaconda3/envs")):
            candidate = base / env_name / "Library" / "bin" / "ffprobe.exe"
            if candidate.exists():
                return str(candidate)
    return "ffprobe"  # fall back to PATH


def _find_ffmpeg() -> str:
    """Find ffmpeg binary — check conda env path first, then PATH."""
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidate = Path(conda_prefix) / "Library" / "bin" / "ffmpeg.exe"
        if candidate.exists():
            return str(candidate)
    home = Path.home()
    for env_name in ("social_env",):
        for base in (home / ".conda" / "envs", home / "anaconda3" / "envs",
                     Path("C:/ProgramData/anaconda3/envs")):
            candidate = base / env_name / "Library" / "bin" / "ffmpeg.exe"
            if candidate.exists():
                return str(candidate)
    return "ffmpeg"


FFPROBE = _find_ffprobe()
FFMPEG = _find_ffmpeg()

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None


PLAYBACK_RATE = 48000
FRAMES_PER_BUFFER = 1024
DTYPE = np.int16


class AudioPlayer:
    """Audio file playback with transport controls and software volume."""

    def __init__(self):
        self._pa = None
        self._stream = None
        self._thread = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially

        self._state = "idle"  # "idle" | "playing" | "paused"
        self._volume = 0.7
        self._position = 0.0  # seconds
        self._duration = 0.0
        self._loaded_path = None

        # Audio data (numpy array, int16, mono or stereo)
        self._audio_data = None
        self._sample_rate = PLAYBACK_RATE
        self._channels = 1
        self._frame_index = 0  # current position in frames

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def get_duration(path: Path) -> float:
        """Get audio duration in seconds. Supports WAV (soundfile) and M4A (ffprobe)."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".wav" and sf:
            try:
                info = sf.info(str(path))
                return info.duration
            except Exception:
                pass

        # Fallback: ffprobe for M4A and other formats
        try:
            result = subprocess.run(
                [FFPROBE, "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 str(path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return 0.0

    @staticmethod
    def list_audio_files(directory: Path) -> list:
        """Scan directory for audio files, sorted newest-first.

        Returns list of dicts:
            {"path": Path, "name": str, "duration_s": float,
             "size_mb": float, "mode": "recording"|"dictation"|"other"}
        """
        directory = Path(directory)
        if not directory.is_dir():
            return []

        extensions = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
        files = []
        for f in directory.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in extensions:
                continue
            # Skip enhanced files
            if "_enhanced" in f.stem:
                continue
            files.append(f)

        # Sort newest first by modification time
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        result = []
        for f in files:
            name = f.stem
            # Determine mode from filename
            if "_recording" in name or "recording" in name.lower():
                mode = "recording"
            elif "_dictation" in name or "dictation" in name.lower():
                mode = "dictation"
            else:
                mode = "other"

            result.append({
                "path": f,
                "name": name,
                "duration_s": AudioPlayer.get_duration(f),
                "size_mb": f.stat().st_size / (1024 * 1024),
                "mode": mode,
            })

        return result

    # ── Playback controls ─────────────────────────────────────────────────

    def load(self, path):
        """Stop current playback, load a new file, prepare for play."""
        path = Path(path)
        self.stop()

        suffix = path.suffix.lower()

        if suffix == ".wav" and sf:
            data, rate = sf.read(str(path), dtype="int16")
        elif sf:
            # Try soundfile first for other formats
            try:
                data, rate = sf.read(str(path), dtype="int16")
            except Exception:
                data, rate = self._decode_with_ffmpeg(path)
        else:
            data, rate = self._decode_with_ffmpeg(path)

        if data is None:
            raise RuntimeError(f"Cannot load audio: {path}")

        # Ensure 2D array (frames x channels)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._audio_data = data
        self._sample_rate = rate
        self._channels = data.shape[1]
        self._duration = len(data) / rate
        self._frame_index = 0
        self._position = 0.0
        self._loaded_path = path
        self._state = "idle"

    def play(self):
        """Start or resume playback."""
        if self._audio_data is None:
            return
        if self._state == "playing":
            return
        if self._state == "paused":
            self._pause_event.set()
            self._state = "playing"
            return

        # Start fresh playback
        self._stop_event.clear()
        self._pause_event.set()
        self._state = "playing"

        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def pause(self):
        """Pause playback (can resume with play)."""
        if self._state != "playing":
            return
        self._pause_event.clear()
        self._state = "paused"

    def stop(self):
        """Stop playback and reset position."""
        if self._state == "idle" and self._thread is None:
            return
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close_stream()
        self._frame_index = 0
        self._position = 0.0
        self._state = "idle"

    def seek(self, position_seconds: float):
        """Seek to absolute position in seconds."""
        if self._audio_data is None:
            return
        position_seconds = max(0.0, min(position_seconds, self._duration))
        with self._lock:
            self._frame_index = int(position_seconds * self._sample_rate)
            self._position = position_seconds

    def skip(self, delta_seconds: float):
        """Relative seek by delta seconds (positive=forward, negative=backward)."""
        self.seek(self._position + delta_seconds)

    def cleanup(self):
        """Release all resources."""
        self.stop()
        self._audio_data = None
        self._loaded_path = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def position(self) -> float:
        return self._position

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float):
        self._volume = max(0.0, min(1.0, value))

    @property
    def loaded_path(self):
        return self._loaded_path

    # ── Internal: playback thread ─────────────────────────────────────────

    def _playback_loop(self):
        """Daemon thread: read chunks, apply volume, write to output stream."""
        if not pyaudio:
            self._state = "idle"
            return

        self._pa = pyaudio.PyAudio()
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                output=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )

            total_frames = len(self._audio_data)

            while not self._stop_event.is_set():
                # Wait if paused
                self._pause_event.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                if not self._pause_event.is_set():
                    continue

                with self._lock:
                    start = self._frame_index
                    end = min(start + FRAMES_PER_BUFFER, total_frames)
                    if start >= total_frames:
                        break  # reached end
                    chunk = self._audio_data[start:end].copy()
                    self._frame_index = end
                    self._position = end / self._sample_rate

                # Apply volume
                if self._volume < 1.0:
                    chunk = (chunk.astype(np.float32) * self._volume).clip(
                        -32768, 32767
                    ).astype(np.int16)

                self._stream.write(chunk.tobytes())

        except Exception as e:
            print(f"[AudioPlayer] Playback error: {e}")
        finally:
            self._close_stream()
            if self._state == "playing":
                # Reached end naturally
                self._frame_index = 0
                self._position = 0.0
                self._state = "idle"

    def _close_stream(self):
        """Close PyAudio stream and terminate instance."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    # ── Internal: ffmpeg decode ────────────────────────────────────────────

    @staticmethod
    def _decode_with_ffmpeg(path: Path):
        """Decode audio file to numpy array via ffmpeg subprocess."""
        try:
            result = subprocess.run(
                [FFMPEG, "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
                 "-ar", str(PLAYBACK_RATE), "-ac", "1", "-"],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                return None, 0
            data = np.frombuffer(result.stdout, dtype=np.int16)
            return data, PLAYBACK_RATE
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None, 0
