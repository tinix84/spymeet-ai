"""
SpyMeet Recorder — Desktop entry point.

Launches:
  - System tray icon (pystray, daemon thread)
  - Floating recording widget (tkinter, main thread)
  - Core recorder (Recorder, managed via callbacks)

Usage:
    python recorder_app.py                    # launch desktop app
    python recorder_app.py --mode dictation   # launch and immediately start dictation
"""

import argparse
import sys
import tkinter as tk

from record import Recorder
from recorder_tray import RecorderTray
from recorder_widget import RecorderWidget


class RecorderApp:
    """Coordinates tray icon, widget, and recorder."""

    def __init__(self):
        self._recorder = Recorder()
        self._root = None
        self._widget = None
        self._tray = None

    def run(self, auto_start_mode=None):
        """Launch the app. Blocks on tkinter mainloop.

        Args:
            auto_start_mode: If set ("meeting" or "dictation"), start recording
                             immediately after launch.
        """
        # Create hidden tkinter root (widget host)
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.title("SpyMeet")

        # Create widget
        self._widget = RecorderWidget(on_stop=self._stop_recording)
        self._widget.setup(self._root)

        # Create tray
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

        print("[SpyMeet] Recorder running. Right-click tray icon to start.")
        self._root.mainloop()

    def _start_recording(self, mode: str):
        """Start recording in the given mode."""
        if self._recorder.is_recording:
            return

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
        self._widget.show(mode=mode, elapsed_getter=lambda: self._recorder.elapsed_seconds)

    def _stop_recording(self):
        """Stop the current recording."""
        if not self._recorder.is_recording:
            return

        path = self._recorder.stop()
        self._tray.set_recording(False)
        self._widget.hide()

        if path and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            msg = f"Saved: {path.name} ({size_mb:.1f} MB)"
            print(f"[SAVED] {path} ({size_mb:.1f} MB)")
            self._tray.notify("SpyMeet", msg)
        else:
            print("[WARNING] No output file produced.")

        if self._recorder.error:
            print(f"[WARNING] {self._recorder.error}")

    def _quit(self):
        """Stop recording if active, tear down tray, exit."""
        if self._recorder.is_recording:
            self._stop_recording()
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
                             "If omitted, app starts idle (use tray menu).")
    args = parser.parse_args()

    app = RecorderApp()
    app.run(auto_start_mode=args.mode)


if __name__ == "__main__":
    main()
