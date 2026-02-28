"""
System Tray Icon — pystray-based tray for SpyMeet recorder.

Two icon states: idle (gray circle) and recording (red circle).
Menu: Start Meeting / Start Dictation / Stop / Quit.

Designed to run in a daemon thread; communicates with the main thread
(tkinter widget) via callbacks.
"""

import threading

from PIL import Image, ImageDraw


def _make_icon(color="gray", size=64):
    """Generate a simple circle icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, size - 4, size - 4], fill=color)
    return img


ICON_IDLE = _make_icon("#888888")
ICON_RECORDING = _make_icon("#ff2222")


class RecorderTray:
    """System tray icon for the recorder."""

    def __init__(self, on_start_meeting=None, on_start_dictation=None,
                 on_stop=None, on_quit=None):
        """
        Args:
            on_start_meeting: Callback for "Start Meeting Recording".
            on_start_dictation: Callback for "Start Dictation".
            on_stop: Callback for "Stop Recording".
            on_quit: Callback for "Quit".
        """
        self._on_start_meeting = on_start_meeting or (lambda: None)
        self._on_start_dictation = on_start_dictation or (lambda: None)
        self._on_stop = on_stop or (lambda: None)
        self._on_quit = on_quit or (lambda: None)
        self._recording = False
        self._icon = None
        self._thread = None

    def start(self):
        """Start the tray icon in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Create and run the pystray icon (blocks in this thread)."""
        import pystray

        menu = pystray.Menu(
            pystray.MenuItem(
                "Start Meeting Recording",
                self._handle_start_meeting,
                enabled=lambda item: not self._recording,
            ),
            pystray.MenuItem(
                "Start Dictation",
                self._handle_start_dictation,
                enabled=lambda item: not self._recording,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Stop Recording",
                self._handle_stop,
                enabled=lambda item: self._recording,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._handle_quit),
        )

        self._icon = pystray.Icon(
            name="SpyMeet",
            icon=ICON_IDLE,
            title="SpyMeet Recorder",
            menu=menu,
        )
        self._icon.run()

    def set_recording(self, recording: bool):
        """Update the tray icon state."""
        self._recording = recording
        if self._icon:
            self._icon.icon = ICON_RECORDING if recording else ICON_IDLE
            self._icon.title = "SpyMeet - Recording..." if recording else "SpyMeet Recorder"

    def notify(self, title: str, message: str):
        """Show a Windows notification via pystray."""
        if self._icon:
            try:
                self._icon.notify(message, title)
            except Exception:
                pass  # Notification not supported on all platforms

    def stop(self):
        """Stop the tray icon."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass

    # ── Menu handlers ───────────────────────────────────────────────────────

    def _handle_start_meeting(self, icon, item):
        self._on_start_meeting()

    def _handle_start_dictation(self, icon, item):
        self._on_start_dictation()

    def _handle_stop(self, icon, item):
        self._on_stop()

    def _handle_quit(self, icon, item):
        self._on_quit()
