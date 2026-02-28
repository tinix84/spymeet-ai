"""
Floating Recording Widget — always-on-top tkinter overlay.

Shows recording indicator, elapsed timer, mode label, and Stop button.
Dark theme, positioned at bottom-right above taskbar.

Designed to be driven by recorder_app.py via show()/hide() and a command queue.
"""

import queue
import tkinter as tk
from tkinter import font as tkfont


class RecorderWidget:
    """Floating overlay widget for recording status."""

    WIDTH = 260
    HEIGHT = 80
    BG_COLOR = "#2d2d2d"
    FG_COLOR = "#ffffff"
    REC_COLOR = "#ff3333"
    BTN_COLOR = "#555555"
    BTN_HOVER = "#777777"

    def __init__(self, on_stop=None):
        """
        Args:
            on_stop: Callback invoked when Stop button is clicked (called in main thread).
        """
        self._on_stop = on_stop
        self._elapsed_getter = None
        self._root = None
        self._visible = False
        self._mode_label_text = None
        self._timer_label = None
        self._rec_dot = None
        self._dot_visible = True
        self._command_queue = queue.Queue()

    def setup(self, root: tk.Tk):
        """Create the widget as a Toplevel of the given root.

        Must be called from the main thread before show()/hide().
        """
        self._root = root
        self._window = tk.Toplevel(root)
        self._window.title("SpyMeet Recorder")
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        self._window.configure(bg=self.BG_COLOR)
        self._window.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        # Prevent focus stealing
        self._window.attributes("-toolwindow", True)

        # Position: bottom-right, above taskbar
        self._position_bottom_right()

        # Build UI
        self._build_ui()

        # Initially hidden
        self._window.withdraw()

        # Start polling command queue
        self._poll_commands()

    def _position_bottom_right(self):
        """Place window at bottom-right of screen, above taskbar."""
        screen_w = self._window.winfo_screenwidth()
        screen_h = self._window.winfo_screenheight()
        x = screen_w - self.WIDTH - 20
        y = screen_h - self.HEIGHT - 80  # above taskbar
        self._window.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _build_ui(self):
        """Build the widget layout."""
        # Main frame
        frame = tk.Frame(self._window, bg=self.BG_COLOR, padx=12, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)

        # Top row: dot + REC + timer + stop button
        top = tk.Frame(frame, bg=self.BG_COLOR)
        top.pack(fill=tk.X)

        # Recording dot (canvas circle)
        self._dot_canvas = tk.Canvas(
            top, width=14, height=14, bg=self.BG_COLOR,
            highlightthickness=0
        )
        self._dot_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._rec_dot = self._dot_canvas.create_oval(2, 2, 12, 12, fill=self.REC_COLOR, outline="")

        # REC label
        try:
            bold_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        except Exception:
            bold_font = tkfont.Font(size=11, weight="bold")
        rec_label = tk.Label(top, text="REC", font=bold_font, fg=self.REC_COLOR, bg=self.BG_COLOR)
        rec_label.pack(side=tk.LEFT, padx=(0, 12))

        # Timer
        try:
            timer_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        except Exception:
            timer_font = tkfont.Font(size=13, weight="bold")
        self._timer_label = tk.Label(
            top, text="00:00", font=timer_font,
            fg=self.FG_COLOR, bg=self.BG_COLOR
        )
        self._timer_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Stop button
        try:
            btn_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        except Exception:
            btn_font = tkfont.Font(size=10, weight="bold")
        self._stop_btn = tk.Button(
            top, text=" \u25a0 ", font=btn_font,
            fg=self.FG_COLOR, bg=self.BTN_COLOR,
            activeforeground=self.FG_COLOR, activebackground=self.BTN_HOVER,
            relief=tk.FLAT, bd=0, padx=8, pady=2,
            command=self._on_stop_click,
        )
        self._stop_btn.pack(side=tk.RIGHT)

        # Bottom row: mode label
        try:
            small_font = tkfont.Font(family="Segoe UI", size=9)
        except Exception:
            small_font = tkfont.Font(size=9)
        self._mode_label_text = tk.Label(
            frame, text="", font=small_font,
            fg="#aaaaaa", bg=self.BG_COLOR, anchor=tk.W
        )
        self._mode_label_text.pack(fill=tk.X, pady=(4, 0))

    def show(self, mode="meeting", elapsed_getter=None):
        """Show the widget. Thread-safe (uses command queue).

        Args:
            mode: "meeting" or "dictation" — sets the subtitle text.
            elapsed_getter: Callable returning elapsed seconds (e.g., recorder.elapsed_seconds).
        """
        self._command_queue.put(("show", mode, elapsed_getter))

    def hide(self):
        """Hide the widget. Thread-safe."""
        self._command_queue.put(("hide",))

    def _poll_commands(self):
        """Process commands from other threads via the queue."""
        try:
            while True:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == "show":
                    self._do_show(cmd[1], cmd[2] if len(cmd) > 2 else None)
                elif cmd[0] == "hide":
                    self._do_hide()
        except queue.Empty:
            pass
        if self._root:
            self._root.after(100, self._poll_commands)

    def _do_show(self, mode, elapsed_getter):
        """Actually show the widget (main thread)."""
        self._elapsed_getter = elapsed_getter
        mode_text = "Meeting recording" if mode == "meeting" else "Dictation (mic only)"
        self._mode_label_text.config(text=mode_text)
        self._timer_label.config(text="00:00")
        self._dot_visible = True
        self._window.deiconify()
        self._visible = True
        self._update_timer()

    def _do_hide(self):
        """Actually hide the widget (main thread)."""
        self._visible = False
        self._elapsed_getter = None
        self._window.withdraw()

    def _update_timer(self):
        """Update the elapsed timer display."""
        if not self._visible:
            return
        if self._elapsed_getter:
            elapsed = self._elapsed_getter()
            mm, ss = divmod(int(elapsed), 60)
            hh, mm_r = divmod(mm, 60)
            if hh > 0:
                self._timer_label.config(text=f"{hh:d}:{mm_r:02d}:{ss:02d}")
            else:
                self._timer_label.config(text=f"{mm:02d}:{ss:02d}")

        # Blink the recording dot
        self._dot_visible = not self._dot_visible
        color = self.REC_COLOR if self._dot_visible else self.BG_COLOR
        self._dot_canvas.itemconfig(self._rec_dot, fill=color)

        self._root.after(500, self._update_timer)

    def _on_stop_click(self):
        """Handle Stop button click."""
        if self._on_stop:
            self._on_stop()
