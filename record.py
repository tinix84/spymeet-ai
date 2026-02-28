"""
Core Recording Engine — WASAPI loopback + mic capture.

Modes:
  meeting   : stereo WAV (L=mic, R=system audio via WASAPI loopback), 48 kHz 16-bit
  dictation : mono WAV (mic only), 48 kHz 16-bit

Usage:
    python record.py --list-devices
    python record.py --start                         # meeting mode (Ctrl+C to stop)
    python record.py --start --mode dictation        # mic-only for LLM prompts
    python record.py --start --output ./custom.wav   # custom output path

Requires: PyAudioWPatch, soundfile, numpy
Optional: scipy (for sample rate conversion when mic != loopback rate)
"""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("[ERROR] PyAudioWPatch not installed. Run: pip install PyAudioWPatch")
    sys.exit(1)


SAMPLE_RATE = 48000
CHANNELS_STEREO = 2
CHANNELS_MONO = 1
SAMPLE_FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024
DTYPE = np.int16


class Recorder:
    """Audio recorder supporting meeting (stereo) and dictation (mono) modes."""

    def __init__(self, output_dir="./audio", mode="meeting"):
        """
        Args:
            output_dir: Directory for output WAV files.
            mode: "meeting" (stereo: L=mic, R=loopback) or "dictation" (mono: mic only).
        """
        if mode not in ("meeting", "dictation"):
            raise ValueError(f"Invalid mode: {mode!r}. Use 'meeting' or 'dictation'.")
        self.output_dir = Path(output_dir)
        self.mode = mode
        self._pa = None
        self._mic_stream = None
        self._loopback_stream = None
        self._mic_queue = queue.Queue()
        self._loopback_queue = queue.Queue()
        self._mixer_thread = None
        self._stop_event = threading.Event()
        self._recording = False
        self._start_time = 0.0
        self._output_path = None
        self._wav_file = None
        self._error = None

    @staticmethod
    def list_devices() -> dict:
        """Discover default mic and loopback devices.

        Returns:
            {"mic": {device_info}, "loopback": {device_info}}

        Raises:
            RuntimeError: If WASAPI host or devices not found.
        """
        pa = pyaudio.PyAudio()
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)

            # Default mic (input)
            default_input_idx = wasapi_info["defaultInputDevice"]
            if default_input_idx < 0:
                raise RuntimeError(
                    "No default microphone found.\n"
                    "  Check Windows Sound Settings > Input."
                )
            mic_info = pa.get_device_info_by_index(default_input_idx)

            # Default output -> find its loopback
            default_output_idx = wasapi_info["defaultOutputDevice"]
            if default_output_idx < 0:
                raise RuntimeError("No default audio output device found.")
            output_info = pa.get_device_info_by_index(default_output_idx)

            loopback_info = None
            for loopback in pa.get_loopback_device_info_generator():
                if output_info["name"] in loopback["name"]:
                    loopback_info = loopback
                    break

            if loopback_info is None:
                raise RuntimeError(
                    f"No WASAPI loopback device found for '{output_info['name']}'.\n"
                    "  Run: python record.py --list-devices"
                )

            return {"mic": mic_info, "loopback": loopback_info}
        finally:
            pa.terminate()

    def start(self, output_path=None) -> Path:
        """Start recording.

        Args:
            output_path: Optional custom output path. Auto-generated if None.

        Returns:
            Path to the output WAV file (known immediately).
        """
        if self._recording:
            raise RuntimeError("Already recording.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        if output_path:
            self._output_path = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            suffix = "recording" if self.mode == "meeting" else "dictation"
            self._output_path = self.output_dir / f"{timestamp}_{suffix}.wav"

        self._pa = pyaudio.PyAudio()
        self._stop_event.clear()
        self._error = None

        # Clear queues
        for q in (self._mic_queue, self._loopback_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        try:
            devices = self._discover_devices()
        except RuntimeError as e:
            self._pa.terminate()
            self._pa = None
            raise

        mic_info = devices["mic"]
        mic_rate = int(mic_info["defaultSampleRate"])

        if self.mode == "meeting":
            loopback_info = devices["loopback"]
            loopback_rate = int(loopback_info["defaultSampleRate"])
            loopback_channels = loopback_info["maxInputChannels"]

            # Open WAV: stereo
            self._wav_file = sf.SoundFile(
                str(self._output_path), mode="w",
                samplerate=SAMPLE_RATE, channels=CHANNELS_STEREO,
                subtype="PCM_16", format="WAV"
            )

            # Open mic stream
            self._mic_stream = self._pa.open(
                format=SAMPLE_FORMAT,
                channels=1,
                rate=mic_rate,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=FRAMES_PER_BUFFER,
                stream_callback=self._mic_callback,
            )

            # Open loopback stream
            self._loopback_stream = self._pa.open(
                format=SAMPLE_FORMAT,
                channels=loopback_channels,
                rate=loopback_rate,
                input=True,
                input_device_index=loopback_info["index"],
                frames_per_buffer=FRAMES_PER_BUFFER,
                stream_callback=self._loopback_callback,
            )

            # Start mixer thread
            self._mixer_thread = threading.Thread(
                target=self._mixer_loop,
                args=(mic_rate, loopback_rate, loopback_channels),
                daemon=True,
            )
            self._mixer_thread.start()

        else:  # dictation
            # Open WAV: mono
            self._wav_file = sf.SoundFile(
                str(self._output_path), mode="w",
                samplerate=SAMPLE_RATE, channels=CHANNELS_MONO,
                subtype="PCM_16", format="WAV"
            )

            # Open mic stream only
            self._mic_stream = self._pa.open(
                format=SAMPLE_FORMAT,
                channels=1,
                rate=mic_rate,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=FRAMES_PER_BUFFER,
                stream_callback=self._dictation_callback,
            )

            # Dictation writer thread
            self._mixer_thread = threading.Thread(
                target=self._dictation_writer_loop,
                args=(mic_rate,),
                daemon=True,
            )
            self._mixer_thread.start()

        self._start_time = time.monotonic()
        self._recording = True
        return self._output_path

    def stop(self) -> Path:
        """Stop recording and finalize the WAV file.

        Returns:
            Path to the saved WAV file.
        """
        if not self._recording:
            return self._output_path

        self._stop_event.set()
        self._recording = False

        # Stop streams
        for stream in (self._mic_stream, self._loopback_stream):
            if stream and stream.is_active():
                try:
                    stream.stop_stream()
                except Exception:
                    pass
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass
        self._mic_stream = None
        self._loopback_stream = None

        # Wait for mixer thread
        if self._mixer_thread and self._mixer_thread.is_alive():
            self._mixer_thread.join(timeout=3.0)
        self._mixer_thread = None

        # Close WAV
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None

        # Terminate PyAudio
        if self._pa:
            self._pa.terminate()
            self._pa = None

        return self._output_path

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed_seconds(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def error(self) -> str | None:
        return self._error

    # ── Internal: device discovery ──────────────────────────────────────────

    def _discover_devices(self) -> dict:
        """Discover devices using the already-initialized PyAudio instance."""
        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)

        default_input_idx = wasapi_info["defaultInputDevice"]
        if default_input_idx < 0:
            raise RuntimeError("No default microphone found.")
        mic_info = self._pa.get_device_info_by_index(default_input_idx)

        result = {"mic": mic_info}

        if self.mode == "meeting":
            default_output_idx = wasapi_info["defaultOutputDevice"]
            if default_output_idx < 0:
                raise RuntimeError("No default audio output device found.")
            output_info = self._pa.get_device_info_by_index(default_output_idx)

            loopback_info = None
            for loopback in self._pa.get_loopback_device_info_generator():
                if output_info["name"] in loopback["name"]:
                    loopback_info = loopback
                    break

            if loopback_info is None:
                raise RuntimeError(
                    f"No WASAPI loopback found for '{output_info['name']}'.\n"
                    "  Try: python record.py --list-devices"
                )
            result["loopback"] = loopback_info

        return result

    # ── Internal: PyAudio callbacks ─────────────────────────────────────────

    def _mic_callback(self, in_data, frame_count, time_info, status):
        self._mic_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        self._loopback_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _dictation_callback(self, in_data, frame_count, time_info, status):
        self._mic_queue.put(in_data)
        return (None, pyaudio.paContinue)

    # ── Internal: mixer thread (meeting mode) ──────────────────────────────

    def _mixer_loop(self, mic_rate, loopback_rate, loopback_channels):
        """Read from mic + loopback queues, interleave as L/R, write to WAV."""
        need_resample_mic = (mic_rate != SAMPLE_RATE)
        need_resample_loopback = (loopback_rate != SAMPLE_RATE)

        try:
            while not self._stop_event.is_set():
                # Get mic data
                try:
                    mic_data = self._mic_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                mic_samples = np.frombuffer(mic_data, dtype=DTYPE).astype(np.float32)

                # Get loopback data (drain what's available)
                loopback_samples = np.array([], dtype=np.float32)
                try:
                    lb_data = self._loopback_queue.get(timeout=0.05)
                    lb = np.frombuffer(lb_data, dtype=DTYPE).astype(np.float32)
                    # Downmix loopback to mono if multi-channel
                    if loopback_channels > 1:
                        lb = lb.reshape(-1, loopback_channels).mean(axis=1)
                    loopback_samples = lb
                except queue.Empty:
                    pass

                # Resample if needed
                if need_resample_mic and len(mic_samples) > 0:
                    mic_samples = self._resample(mic_samples, mic_rate, SAMPLE_RATE)
                if need_resample_loopback and len(loopback_samples) > 0:
                    loopback_samples = self._resample(loopback_samples, loopback_rate, SAMPLE_RATE)

                # Match lengths (pad shorter with zeros)
                mic_len = len(mic_samples)
                lb_len = len(loopback_samples)
                target_len = max(mic_len, lb_len)
                if target_len == 0:
                    continue

                if mic_len < target_len:
                    mic_samples = np.pad(mic_samples, (0, target_len - mic_len))
                elif mic_len > target_len:
                    mic_samples = mic_samples[:target_len]

                if lb_len < target_len:
                    loopback_samples = np.pad(loopback_samples, (0, target_len - lb_len))
                elif lb_len > target_len:
                    loopback_samples = loopback_samples[:target_len]

                # Interleave: L=mic, R=loopback
                stereo = np.column_stack((mic_samples, loopback_samples))
                self._wav_file.write(stereo.astype(np.int16))

        except OSError as e:
            self._error = f"Disk write error: {e}"
        except Exception as e:
            self._error = f"Mixer error: {e}"
        finally:
            # Drain remaining data
            self._drain_and_write(mic_rate, loopback_rate, loopback_channels,
                                  need_resample_mic, need_resample_loopback)

    def _drain_and_write(self, mic_rate, loopback_rate, loopback_channels,
                         need_resample_mic, need_resample_loopback):
        """Drain remaining queued audio data after stop."""
        remaining_mic = []
        remaining_lb = []
        while not self._mic_queue.empty():
            try:
                data = self._mic_queue.get_nowait()
                samples = np.frombuffer(data, dtype=DTYPE).astype(np.float32)
                remaining_mic.append(samples)
            except queue.Empty:
                break
        while not self._loopback_queue.empty():
            try:
                data = self._loopback_queue.get_nowait()
                samples = np.frombuffer(data, dtype=DTYPE).astype(np.float32)
                if loopback_channels > 1:
                    samples = samples.reshape(-1, loopback_channels).mean(axis=1)
                remaining_lb.append(samples)
            except queue.Empty:
                break

        if remaining_mic or remaining_lb:
            mic_arr = np.concatenate(remaining_mic) if remaining_mic else np.array([], dtype=np.float32)
            lb_arr = np.concatenate(remaining_lb) if remaining_lb else np.array([], dtype=np.float32)

            if need_resample_mic and len(mic_arr) > 0:
                mic_arr = self._resample(mic_arr, mic_rate, SAMPLE_RATE)
            if need_resample_loopback and len(lb_arr) > 0:
                lb_arr = self._resample(lb_arr, loopback_rate, SAMPLE_RATE)

            target_len = max(len(mic_arr), len(lb_arr))
            if target_len > 0:
                mic_arr = np.pad(mic_arr, (0, max(0, target_len - len(mic_arr))))[:target_len]
                lb_arr = np.pad(lb_arr, (0, max(0, target_len - len(lb_arr))))[:target_len]
                stereo = np.column_stack((mic_arr, lb_arr))
                try:
                    self._wav_file.write(stereo.astype(np.int16))
                except Exception:
                    pass

    # ── Internal: dictation writer thread ───────────────────────────────────

    def _dictation_writer_loop(self, mic_rate):
        """Read from mic queue and write mono WAV (dictation mode)."""
        need_resample = (mic_rate != SAMPLE_RATE)
        try:
            while not self._stop_event.is_set():
                try:
                    data = self._mic_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                samples = np.frombuffer(data, dtype=DTYPE).astype(np.float32)
                if need_resample and len(samples) > 0:
                    samples = self._resample(samples, mic_rate, SAMPLE_RATE)
                self._wav_file.write(samples.astype(np.int16))
        except OSError as e:
            self._error = f"Disk write error: {e}"
        except Exception as e:
            self._error = f"Writer error: {e}"
        finally:
            # Drain remaining
            while not self._mic_queue.empty():
                try:
                    data = self._mic_queue.get_nowait()
                    samples = np.frombuffer(data, dtype=DTYPE).astype(np.float32)
                    if need_resample and len(samples) > 0:
                        samples = self._resample(samples, mic_rate, SAMPLE_RATE)
                    self._wav_file.write(samples.astype(np.int16))
                except (queue.Empty, Exception):
                    break

    # ── Internal: resampling ────────────────────────────────────────────────

    @staticmethod
    def _resample(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio buffer to target rate."""
        if orig_rate == target_rate or len(samples) == 0:
            return samples
        try:
            from scipy.signal import resample
            new_len = int(len(samples) * target_rate / orig_rate)
            return resample(samples, new_len).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = target_rate / orig_rate
            new_len = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, new_len)
            return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audio recorder — WASAPI loopback + mic capture",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    parser.add_argument("--start", action="store_true",
                        help="Start recording (Ctrl+C to stop)")
    parser.add_argument("--mode", default="meeting", choices=["meeting", "dictation"],
                        help="meeting  = stereo (L=mic, R=system audio)\n"
                             "dictation = mono (mic only, for LLM prompts)")
    parser.add_argument("--output", default=None,
                        help="Custom output file path")
    args = parser.parse_args()

    if args.list_devices:
        try:
            devices = Recorder.list_devices()
            print("\n[AUDIO DEVICES]")
            mic = devices["mic"]
            print(f"  Microphone:  {mic['name']}")
            print(f"    Index: {mic['index']}, Rate: {int(mic['defaultSampleRate'])} Hz, "
                  f"Channels: {mic['maxInputChannels']}")

            lb = devices["loopback"]
            print(f"  Loopback:    {lb['name']}")
            print(f"    Index: {lb['index']}, Rate: {int(lb['defaultSampleRate'])} Hz, "
                  f"Channels: {lb['maxInputChannels']}")
            print()
        except RuntimeError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)
        return

    if args.start:
        recorder = Recorder(mode=args.mode)
        try:
            path = recorder.start(output_path=args.output)
            mode_label = "Meeting (stereo: L=mic, R=system)" if args.mode == "meeting" else "Dictation (mono: mic only)"
            print(f"\n[RECORDING] {mode_label}")
            print(f"  Output: {path}")
            print(f"  Press Ctrl+C to stop...\n")

            while recorder.is_recording:
                elapsed = recorder.elapsed_seconds
                mm, ss = divmod(int(elapsed), 60)
                hh, mm = divmod(mm, 60)
                print(f"\r  Elapsed: {hh:02d}:{mm:02d}:{ss:02d}", end="", flush=True)
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n")
        except RuntimeError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)
        finally:
            path = recorder.stop()
            if path and path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                elapsed = recorder.elapsed_seconds or (time.monotonic() - recorder._start_time) if hasattr(recorder, '_start_time') else 0
                print(f"[SAVED] {path} ({size_mb:.1f} MB)")
            if recorder.error:
                print(f"[WARNING] {recorder.error}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
