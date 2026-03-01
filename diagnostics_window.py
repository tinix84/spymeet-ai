"""
Audio Diagnostics Window — device probe + live VU meters.

Separate Toplevel window (~400x350) with dark theme. Probes WASAPI
mic and loopback devices, shows info + pass/fail status, and draws
live VU meter bars (green/yellow/red) updated at ~20fps.

Usage:
    diag = DiagnosticsWindow(root)
    diag.open()       # show window, start metering
    diag.close()      # stop streams, hide window
"""

import collections
import math
import tkinter as tk
from tkinter import font as tkfont

import numpy as np


class DiagnosticsWindow:
    """Audio diagnostics panel with device info + live VU meters."""

    WIDTH = 400
    HEIGHT = 350
    BG = "#2d2d2d"
    PANEL_BG = "#252525"
    FG = "#ffffff"
    FG_DIM = "#888888"
    PASS_COLOR = "#44bb44"
    FAIL_COLOR = "#dd3333"
    BAR_BG = "#1e1e1e"
    # VU bar gradient: green (-60 to -20), yellow (-20 to -6), red (-6 to 0)
    GREEN = "#22cc44"
    YELLOW = "#cccc22"
    RED = "#dd3333"

    def __init__(self, root: tk.Tk):
        self._root = root
        self._window = None
        self._pa = None
        self._mic_stream = None
        self._loopback_stream = None
        self._mic_level = collections.deque(maxlen=1)
        self._loopback_level = collections.deque(maxlen=1)
        self._update_id = None
        self._mic_canvas = None
        self._loopback_canvas = None
        self._mic_db_label = None
        self._loopback_db_label = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self):
        """Show the diagnostics window and start metering."""
        if self._is_open and self._window:
            try:
                self._window.lift()
                return
            except tk.TclError:
                pass

        self._build_window()
        self._probe_devices()
        self._start_metering()
        self._is_open = True

    def close(self):
        """Stop metering and close the window."""
        self._stop_metering()
        if self._window:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None
        self._is_open = False

    # ── Window construction ───────────────────────────────────────────────

    def _build_window(self):
        self._window = tk.Toplevel(self._root)
        self._window.title("Audio Diagnostics")
        self._window.configure(bg=self.BG)
        self._window.resizable(False, False)
        self._window.attributes("-topmost", True)
        self._window.protocol("WM_DELETE_WINDOW", self.close)

        # Center on screen
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = (screen_w - self.WIDTH) // 2
        y = (screen_h - self.HEIGHT) // 2
        self._window.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

        try:
            title_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
            section_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
            label_font = tkfont.Font(family="Segoe UI", size=9)
            mono_font = tkfont.Font(family="Consolas", size=9)
        except Exception:
            title_font = tkfont.Font(size=11, weight="bold")
            section_font = tkfont.Font(size=10, weight="bold")
            label_font = tkfont.Font(size=9)
            mono_font = tkfont.Font(size=9)

        # Title
        tk.Label(self._window, text="Audio Diagnostics", font=title_font,
                 fg=self.FG, bg=self.BG).pack(fill=tk.X, padx=12, pady=(10, 8))

        # Mic section
        mic_frame = tk.LabelFrame(self._window, text="MICROPHONE", font=section_font,
                                  fg=self.FG, bg=self.PANEL_BG, bd=1, relief=tk.GROOVE,
                                  labelanchor="nw")
        mic_frame.pack(fill=tk.X, padx=12, pady=(0, 6))

        self._mic_name_label = tk.Label(mic_frame, text="Name: (probing...)",
                                        font=label_font, fg=self.FG_DIM, bg=self.PANEL_BG,
                                        anchor=tk.W)
        self._mic_name_label.pack(fill=tk.X, padx=8, pady=(4, 0))

        self._mic_info_label = tk.Label(mic_frame, text="Rate: — | Channels: —",
                                        font=label_font, fg=self.FG_DIM, bg=self.PANEL_BG,
                                        anchor=tk.W)
        self._mic_info_label.pack(fill=tk.X, padx=8)

        self._mic_status_label = tk.Label(mic_frame, text="Status: ...",
                                          font=label_font, fg=self.FG_DIM, bg=self.PANEL_BG,
                                          anchor=tk.W)
        self._mic_status_label.pack(fill=tk.X, padx=8)

        # Mic VU bar
        vu_row_mic = tk.Frame(mic_frame, bg=self.PANEL_BG)
        vu_row_mic.pack(fill=tk.X, padx=8, pady=(4, 8))
        tk.Label(vu_row_mic, text="Level:", font=label_font,
                 fg=self.FG_DIM, bg=self.PANEL_BG).pack(side=tk.LEFT, padx=(0, 4))
        self._mic_canvas = tk.Canvas(vu_row_mic, width=250, height=16,
                                     bg=self.BAR_BG, highlightthickness=0)
        self._mic_canvas.pack(side=tk.LEFT, padx=(0, 6))
        self._mic_db_label = tk.Label(vu_row_mic, text="—", font=mono_font,
                                      fg=self.FG_DIM, bg=self.PANEL_BG, width=5,
                                      anchor=tk.W)
        self._mic_db_label.pack(side=tk.LEFT)

        # Separator
        tk.Frame(self._window, bg="#444444", height=1).pack(fill=tk.X, padx=12)

        # Loopback section
        lb_frame = tk.LabelFrame(self._window, text="LOOPBACK", font=section_font,
                                 fg=self.FG, bg=self.PANEL_BG, bd=1, relief=tk.GROOVE,
                                 labelanchor="nw")
        lb_frame.pack(fill=tk.X, padx=12, pady=(6, 6))

        self._lb_name_label = tk.Label(lb_frame, text="Name: (probing...)",
                                       font=label_font, fg=self.FG_DIM, bg=self.PANEL_BG,
                                       anchor=tk.W)
        self._lb_name_label.pack(fill=tk.X, padx=8, pady=(4, 0))

        self._lb_info_label = tk.Label(lb_frame, text="Rate: — | Channels: —",
                                       font=label_font, fg=self.FG_DIM, bg=self.PANEL_BG,
                                       anchor=tk.W)
        self._lb_info_label.pack(fill=tk.X, padx=8)

        self._lb_status_label = tk.Label(lb_frame, text="Status: ...",
                                         font=label_font, fg=self.FG_DIM, bg=self.PANEL_BG,
                                         anchor=tk.W)
        self._lb_status_label.pack(fill=tk.X, padx=8)

        # Loopback VU bar
        vu_row_lb = tk.Frame(lb_frame, bg=self.PANEL_BG)
        vu_row_lb.pack(fill=tk.X, padx=8, pady=(4, 8))
        tk.Label(vu_row_lb, text="Level:", font=label_font,
                 fg=self.FG_DIM, bg=self.PANEL_BG).pack(side=tk.LEFT, padx=(0, 4))
        self._loopback_canvas = tk.Canvas(vu_row_lb, width=250, height=16,
                                          bg=self.BAR_BG, highlightthickness=0)
        self._loopback_canvas.pack(side=tk.LEFT, padx=(0, 6))
        self._loopback_db_label = tk.Label(vu_row_lb, text="—", font=mono_font,
                                           fg=self.FG_DIM, bg=self.PANEL_BG, width=5,
                                           anchor=tk.W)
        self._loopback_db_label.pack(side=tk.LEFT)

    # ── Device probing ────────────────────────────────────────────────────

    def _probe_devices(self):
        """Probe WASAPI mic and loopback devices, update labels."""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self._mic_name_label.config(text="Name: PyAudioWPatch not installed")
            self._mic_status_label.config(text="Status: FAIL", fg=self.FAIL_COLOR)
            self._lb_name_label.config(text="Name: PyAudioWPatch not installed")
            self._lb_status_label.config(text="Status: FAIL", fg=self.FAIL_COLOR)
            return

        pa = pyaudio.PyAudio()
        try:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)

            # Mic
            try:
                mic_idx = wasapi_info["defaultInputDevice"]
                if mic_idx < 0:
                    raise RuntimeError("No default input device")
                mic = pa.get_device_info_by_index(mic_idx)
                name = mic["name"]
                if len(name) > 40:
                    name = name[:37] + "..."
                self._mic_name_label.config(text=f"Name: {name}")
                self._mic_info_label.config(
                    text=f"Rate: {int(mic['defaultSampleRate'])} Hz  |  "
                         f"Channels: {mic['maxInputChannels']}")
                self._mic_status_label.config(text="Status: PASS", fg=self.PASS_COLOR)
                self._mic_device = mic
            except Exception as e:
                self._mic_name_label.config(text=f"Name: Error")
                self._mic_status_label.config(text=f"Status: FAIL - {e}",
                                              fg=self.FAIL_COLOR)
                self._mic_device = None

            # Loopback
            try:
                out_idx = wasapi_info["defaultOutputDevice"]
                if out_idx < 0:
                    raise RuntimeError("No default output device")
                output = pa.get_device_info_by_index(out_idx)

                loopback = None
                for lb in pa.get_loopback_device_info_generator():
                    if output["name"] in lb["name"]:
                        loopback = lb
                        break

                if loopback is None:
                    raise RuntimeError("No loopback for default output")

                name = loopback["name"]
                if len(name) > 40:
                    name = name[:37] + "..."
                self._lb_name_label.config(text=f"Name: {name}")
                self._lb_info_label.config(
                    text=f"Rate: {int(loopback['defaultSampleRate'])} Hz  |  "
                         f"Channels: {loopback['maxInputChannels']}")
                self._lb_status_label.config(text="Status: PASS", fg=self.PASS_COLOR)
                self._loopback_device = loopback
            except Exception as e:
                self._lb_name_label.config(text=f"Name: Error")
                self._lb_status_label.config(text=f"Status: FAIL - {e}",
                                             fg=self.FAIL_COLOR)
                self._loopback_device = None
        finally:
            pa.terminate()

    # ── VU metering ───────────────────────────────────────────────────────

    def _start_metering(self):
        """Open PyAudio streams in callback mode for VU metering."""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            return

        mic_dev = getattr(self, "_mic_device", None)
        lb_dev = getattr(self, "_loopback_device", None)

        if not mic_dev and not lb_dev:
            return

        self._pa = pyaudio.PyAudio()

        # Mic stream
        if mic_dev:
            try:
                self._mic_stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=int(mic_dev["defaultSampleRate"]),
                    input=True,
                    input_device_index=mic_dev["index"],
                    frames_per_buffer=2048,
                    stream_callback=self._mic_callback,
                )
            except Exception as e:
                print(f"[Diagnostics] Mic stream error: {e}")
                self._mic_stream = None

        # Loopback stream
        if lb_dev:
            try:
                self._loopback_stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=lb_dev["maxInputChannels"],
                    rate=int(lb_dev["defaultSampleRate"]),
                    input=True,
                    input_device_index=lb_dev["index"],
                    frames_per_buffer=2048,
                    stream_callback=self._loopback_callback,
                )
            except Exception as e:
                print(f"[Diagnostics] Loopback stream error: {e}")
                self._loopback_stream = None

        # Start UI update loop
        self._update_meters()

    def _stop_metering(self):
        """Stop VU streams and cancel update loop."""
        if self._update_id is not None:
            try:
                self._root.after_cancel(self._update_id)
            except Exception:
                pass
            self._update_id = None

        for stream in (self._mic_stream, self._loopback_stream):
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        self._mic_stream = None
        self._loopback_stream = None

        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for mic — compute RMS, store in deque."""
        import pyaudiowpatch as pyaudio
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
        self._mic_level.append(rms)
        return (None, pyaudio.paContinue)

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for loopback — compute RMS, store in deque."""
        import pyaudiowpatch as pyaudio
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0.0
        self._loopback_level.append(rms)
        return (None, pyaudio.paContinue)

    def _update_meters(self):
        """Update VU meter canvases at ~20fps."""
        if not self._is_open:
            return

        # Mic
        if self._mic_canvas:
            rms = self._mic_level[-1] if self._mic_level else 0.0
            db = self._rms_to_db(rms)
            self._draw_bar(self._mic_canvas, db)
            if self._mic_db_label:
                self._mic_db_label.config(text=f"{db:.0f}" if db > -60 else "—")

        # Loopback
        if self._loopback_canvas:
            rms = self._loopback_level[-1] if self._loopback_level else 0.0
            db = self._rms_to_db(rms)
            self._draw_bar(self._loopback_canvas, db)
            if self._loopback_db_label:
                self._loopback_db_label.config(text=f"{db:.0f}" if db > -60 else "—")

        self._update_id = self._root.after(50, self._update_meters)

    @staticmethod
    def _rms_to_db(rms: float) -> float:
        """Convert RMS (int16 scale) to dBFS."""
        if rms < 1.0:
            return -60.0
        db = 20 * math.log10(rms / 32768.0)
        return max(-60.0, min(0.0, db))

    def _draw_bar(self, canvas: tk.Canvas, db: float):
        """Draw a colored VU bar on the given canvas."""
        canvas.delete("all")
        w = canvas.winfo_width() or 250
        h = canvas.winfo_height() or 16

        # Map dB (-60..0) to pixels (0..w)
        ratio = (db + 60.0) / 60.0
        bar_w = int(ratio * w)

        if bar_w <= 0:
            return

        # Draw segments with color thresholds
        # -60 to -20 = green, -20 to -6 = yellow, -6 to 0 = red
        green_end = int((40.0 / 60.0) * w)   # -20 dB
        yellow_end = int((54.0 / 60.0) * w)  # -6 dB

        # Green segment
        g_w = min(bar_w, green_end)
        if g_w > 0:
            canvas.create_rectangle(0, 1, g_w, h - 1, fill=self.GREEN, outline="")

        # Yellow segment
        if bar_w > green_end:
            y_w = min(bar_w, yellow_end)
            canvas.create_rectangle(green_end, 1, y_w, h - 1,
                                    fill=self.YELLOW, outline="")

        # Red segment
        if bar_w > yellow_end:
            canvas.create_rectangle(yellow_end, 1, bar_w, h - 1,
                                    fill=self.RED, outline="")
