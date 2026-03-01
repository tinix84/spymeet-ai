"""
SpyMeet Recorder — Desktop entry point.

Launches:
  - Floating widget (tkinter, main thread) — always visible, Start/Stop buttons
  - System tray icon (pystray, daemon thread) — secondary control
  - Core recorder (Recorder, managed via callbacks)

Usage:
    python recorder_app.py                    # launch desktop app
    python recorder_app.py --mode dictation   # launch and immediately start dictation
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path

from audio_player import AudioPlayer
from diagnostics_window import DiagnosticsWindow
from pipeline_runner import PipelineRunner
from record import Recorder
from recorder_tray import RecorderTray
from recorder_widget import RecorderWidget

AUDIO_DIR = Path("./audio")


class RecorderApp:
    """Coordinates tray icon, widget, recorder, and audio player."""

    def __init__(self):
        self._recorder = Recorder()
        self._player = AudioPlayer()
        self._root = None
        self._widget = None
        self._tray = None
        self._pipeline_runner = None
        self._diagnostics = None

    def run(self, auto_start_mode=None):
        """Launch the app. Blocks on tkinter mainloop."""
        # Create hidden tkinter root (widget host)
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.title("SpyMeet")

        # Create pipeline runner
        self._pipeline_runner = PipelineRunner(
            project_dir=Path(".").resolve(),
            on_status=self._on_pipeline_status,
            on_complete=self._on_pipeline_complete,
        )

        # Create widget — visible immediately with Start buttons
        self._widget = RecorderWidget(
            on_stop=self._stop_recording,
            on_start_meeting=lambda: self._start_recording("meeting"),
            on_start_dictation=lambda: self._start_recording("dictation"),
            player=self._player,
            audio_dir=AUDIO_DIR,
            pipeline_runner=self._pipeline_runner,
            on_open_diagnostics=self._open_diagnostics,
        )
        self._widget.setup(self._root)

        # Create tray (secondary control, may be hidden in overflow on Win11)
        self._tray = RecorderTray(
            on_start_meeting=lambda: self._start_recording("meeting"),
            on_start_dictation=lambda: self._start_recording("dictation"),
            on_stop=self._stop_recording,
            on_quit=self._quit,
        )
        self._tray.start()

        # Auto-start if requested
        if auto_start_mode:
            self._root.after(500, lambda: self._start_recording(auto_start_mode))

        # Handle window close
        self._root.protocol("WM_DELETE_WINDOW", self._quit)

        print("[SpyMeet] Recorder ready. Use the floating widget or tray icon.")
        self._root.mainloop()

    def _start_recording(self, mode: str):
        """Start recording in the given mode."""
        if self._recorder.is_recording:
            return

        # Stop any active playback before recording
        self._player.stop()
        self._widget.set_recording_active(True)

        self._recorder = Recorder(mode=mode)
        try:
            path = self._recorder.start()
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            self._tray.notify("SpyMeet Error", str(e))
            return

        mode_label = "Meeting" if mode == "meeting" else "Dictation"
        print(f"[RECORDING] {mode_label} -> {path}")

        self._tray.set_recording(True)
        self._widget.show_recording(mode=mode, elapsed_getter=lambda: self._recorder.elapsed_seconds)

    def _stop_recording(self):
        """Stop the current recording."""
        if not self._recorder.is_recording:
            return

        path = self._recorder.stop()
        self._tray.set_recording(False)
        self._widget.set_recording_active(False)
        self._widget.show_idle()
        self._widget.refresh_files()

        if path and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            msg = f"Saved: {path.name} ({size_mb:.1f} MB)"
            print(f"[SAVED] {path} ({size_mb:.1f} MB)")
            self._tray.notify("SpyMeet", msg)
        else:
            print("[WARNING] No output file produced.")

        if self._recorder.error:
            print(f"[WARNING] {self._recorder.error}")

    # ── Pipeline callbacks (called from background thread) ───────────────

    def _on_pipeline_status(self, msg: str):
        if self._widget:
            self._widget.pipeline_status(msg)

    def _on_pipeline_complete(self, success: bool, msg: str):
        if self._widget:
            self._widget.pipeline_complete(success, msg)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def _open_diagnostics(self):
        """Lazy-create and open the diagnostics window."""
        if self._recorder.is_recording:
            return  # mic is in use
        if self._diagnostics is None:
            self._diagnostics = DiagnosticsWindow(self._root)
        self._diagnostics.open()

    def _quit(self):
        """Stop recording if active, tear down tray and player, exit."""
        if self._recorder.is_recording:
            self._stop_recording()
        if self._diagnostics:
            self._diagnostics.close()
        self._player.cleanup()
        self._tray.stop()
        if self._root:
            self._root.quit()
            self._root.destroy()
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="SpyMeet Recorder — desktop audio capture",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", default=None, choices=["meeting", "dictation"],
                        help="Auto-start in this mode on launch.\n"
                             "If omitted, app starts idle (use widget buttons).")
    args = parser.parse_args()

    app = RecorderApp()
    app.run(auto_start_mode=args.mode)


if __name__ == "__main__":
    main()
