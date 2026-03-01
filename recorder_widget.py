"""
Floating Recording Widget — always-on-top tkinter overlay.

Three visual modes:
  - Idle compact (280x90): Start Meeting / Start Dictation + chevron expand + minimize
  - Expanded (~280x490): compact + file history list + audio player controls + action toolbar
  - Minimized pill (50x30): tiny "SM" button, click to restore

Dark theme, positioned at bottom-right above taskbar.
Driven by recorder_app.py via show_idle()/show_recording()/hide().
"""

import os
import queue
import re
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont


class RecorderWidget:
    """Floating overlay widget for recording status + audio player."""

    WIDTH = 280
    COMPACT_HEIGHT = 90
    EXPANDED_HEIGHT = 490
    PILL_WIDTH = 50
    PILL_HEIGHT = 30
    BG_COLOR = "#2d2d2d"
    FG_COLOR = "#ffffff"
    REC_COLOR = "#ff3333"
    IDLE_COLOR = "#888888"
    BTN_COLOR = "#555555"
    BTN_HOVER = "#777777"
    BTN_MEETING = "#2266cc"
    BTN_DICTATION = "#338833"
    LIST_BG = "#1e1e1e"
    LIST_FG = "#cccccc"
    LIST_SEL_BG = "#3a5a8a"
    PANEL_BG = "#252525"
    ACCENT = "#4a9eff"

    def __init__(self, on_stop=None, on_start_meeting=None, on_start_dictation=None,
                 player=None, audio_dir=None, pipeline_runner=None,
                 on_open_diagnostics=None):
        self._on_stop = on_stop
        self._on_start_meeting = on_start_meeting
        self._on_start_dictation = on_start_dictation
        self._player = player
        self._audio_dir = Path(audio_dir) if audio_dir else Path("./audio")
        self._pipeline_runner = pipeline_runner
        self._on_open_diagnostics = on_open_diagnostics
        self._elapsed_getter = None
        self._root = None
        self._visible = False
        self._state = "idle"  # "idle" or "recording"
        self._widget_mode = "compact"  # "compact" | "expanded" | "pill"
        self._dot_visible = True
        self._command_queue = queue.Queue()
        self._recording_active = False
        # File list data
        self._file_list_data = []
        # Widget references
        self._window = None
        self._pill_window = None
        self._idle_frame = None
        self._rec_frame = None
        self._timer_label = None
        self._mode_label_text = None
        self._dot_canvas = None
        self._rec_dot = None
        # Expand panel references
        self._expand_frame = None
        self._file_listbox = None
        self._player_frame = None
        self._chevron_label = None
        self._now_playing_label = None
        self._progress_scale = None
        self._time_label = None
        self._play_btn = None
        self._vol_scale = None
        self._player_update_id = None
        self._seeking = False
        # Action toolbar references
        self._enhance_btn = None
        self._transcribe_btn = None
        self._llm_btn = None
        self._status_label = None
        self._diag_btn = None

    def setup(self, root: tk.Tk):
        """Create the widget as a Toplevel. Must be called from main thread."""
        self._root = root
        self._window = tk.Toplevel(root)
        self._window.title("SpyMeet")
        self._window.overrideredirect(True)
        self._window.attributes("-topmost", True)
        self._window.attributes("-toolwindow", True)
        self._window.configure(bg=self.BG_COLOR)

        self._build_idle_ui()
        self._build_rec_ui()
        self._build_expand_panel()

        # Start in idle compact
        self._show_frame("idle")
        self._apply_geometry()
        self._visible = True

        # Allow dragging
        self._drag_data = {"x": 0, "y": 0}
        self._window.bind("<ButtonPress-1>", self._on_drag_start)
        self._window.bind("<B1-Motion>", self._on_drag_motion)

        # Build pill window (hidden initially)
        self._build_pill()

        # Start polling
        self._poll_commands()

    def _apply_geometry(self):
        """Set window geometry based on current widget mode."""
        screen_w = self._window.winfo_screenwidth()
        screen_h = self._window.winfo_screenheight()

        if self._widget_mode == "expanded":
            h = self.EXPANDED_HEIGHT
        else:
            h = self.COMPACT_HEIGHT

        x = screen_w - self.WIDTH - 20
        y = screen_h - h - 80
        self._window.geometry(f"{self.WIDTH}x{h}+{x}+{y}")

    def _position_bottom_right(self):
        self._apply_geometry()

    # ── Drag support ────────────────────────────────────────────────────────

    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag_motion(self, event):
        x = self._window.winfo_x() + (event.x - self._drag_data["x"])
        y = self._window.winfo_y() + (event.y - self._drag_data["y"])
        self._window.geometry(f"+{x}+{y}")

    # ── Idle UI ─────────────────────────────────────────────────────────────

    def _build_idle_ui(self):
        self._idle_frame = tk.Frame(self._window, bg=self.BG_COLOR, padx=12, pady=10)

        try:
            title_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
            btn_font = tkfont.Font(family="Segoe UI", size=9)
            icon_font = tkfont.Font(family="Segoe UI", size=10)
        except Exception:
            title_font = tkfont.Font(size=10, weight="bold")
            btn_font = tkfont.Font(size=9)
            icon_font = tkfont.Font(size=10)

        # Title row
        title_row = tk.Frame(self._idle_frame, bg=self.BG_COLOR)
        title_row.pack(fill=tk.X, pady=(0, 8))

        # Idle dot
        dot = tk.Canvas(title_row, width=12, height=12, bg=self.BG_COLOR, highlightthickness=0)
        dot.pack(side=tk.LEFT, padx=(0, 6))
        dot.create_oval(2, 2, 10, 10, fill=self.IDLE_COLOR, outline="")

        tk.Label(title_row, text="SpyMeet Recorder", font=title_font,
                 fg=self.FG_COLOR, bg=self.BG_COLOR).pack(side=tk.LEFT)

        # Close button
        close_btn = tk.Label(title_row, text="\u2715", font=icon_font,
                             fg="#888888", bg=self.BG_COLOR, cursor="hand2")
        close_btn.pack(side=tk.RIGHT, padx=(4, 0))
        close_btn.bind("<Button-1>", lambda e: self._on_close())

        # Minimize button
        min_btn = tk.Label(title_row, text="_", font=icon_font,
                           fg="#888888", bg=self.BG_COLOR, cursor="hand2")
        min_btn.pack(side=tk.RIGHT, padx=(4, 0))
        min_btn.bind("<Button-1>", lambda e: self._do_minimize())

        # Chevron expand/collapse
        self._chevron_label = tk.Label(title_row, text="\u25bc", font=icon_font,
                                        fg="#888888", bg=self.BG_COLOR, cursor="hand2")
        self._chevron_label.pack(side=tk.RIGHT, padx=(4, 0))
        self._chevron_label.bind("<Button-1>", lambda e: self._toggle_expand())

        # Button row
        btn_row = tk.Frame(self._idle_frame, bg=self.BG_COLOR)
        btn_row.pack(fill=tk.X)

        meeting_btn = tk.Button(
            btn_row, text="\u25cf Meeting", font=btn_font,
            fg=self.FG_COLOR, bg=self.BTN_MEETING,
            activeforeground=self.FG_COLOR, activebackground="#3377dd",
            relief=tk.FLAT, bd=0, padx=12, pady=4, cursor="hand2",
            command=self._on_meeting_click,
        )
        meeting_btn.pack(side=tk.LEFT, padx=(0, 8))

        dictation_btn = tk.Button(
            btn_row, text="\u25cf Dictation", font=btn_font,
            fg=self.FG_COLOR, bg=self.BTN_DICTATION,
            activeforeground=self.FG_COLOR, activebackground="#44aa44",
            relief=tk.FLAT, bd=0, padx=12, pady=4, cursor="hand2",
            command=self._on_dictation_click,
        )
        dictation_btn.pack(side=tk.LEFT)

    # ── Recording UI ────────────────────────────────────────────────────────

    def _build_rec_ui(self):
        self._rec_frame = tk.Frame(self._window, bg=self.BG_COLOR, padx=12, pady=8)

        try:
            bold_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
            timer_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
            small_font = tkfont.Font(family="Segoe UI", size=9)
            btn_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        except Exception:
            bold_font = tkfont.Font(size=11, weight="bold")
            timer_font = tkfont.Font(size=14, weight="bold")
            small_font = tkfont.Font(size=9)
            btn_font = tkfont.Font(size=10, weight="bold")

        # Top row: dot + REC + timer + stop
        top = tk.Frame(self._rec_frame, bg=self.BG_COLOR)
        top.pack(fill=tk.X)

        self._dot_canvas = tk.Canvas(top, width=14, height=14, bg=self.BG_COLOR, highlightthickness=0)
        self._dot_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._rec_dot = self._dot_canvas.create_oval(2, 2, 12, 12, fill=self.REC_COLOR, outline="")

        tk.Label(top, text="REC", font=bold_font, fg=self.REC_COLOR, bg=self.BG_COLOR).pack(side=tk.LEFT, padx=(0, 12))

        self._timer_label = tk.Label(top, text="00:00", font=timer_font, fg=self.FG_COLOR, bg=self.BG_COLOR)
        self._timer_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        stop_btn = tk.Button(
            top, text=" \u25a0 Stop ", font=btn_font,
            fg=self.FG_COLOR, bg="#cc2222",
            activeforeground=self.FG_COLOR, activebackground="#ee3333",
            relief=tk.FLAT, bd=0, padx=8, pady=2, cursor="hand2",
            command=self._on_stop_click,
        )
        stop_btn.pack(side=tk.RIGHT)

        # Bottom row: mode label
        self._mode_label_text = tk.Label(
            self._rec_frame, text="", font=small_font,
            fg="#aaaaaa", bg=self.BG_COLOR, anchor=tk.W
        )
        self._mode_label_text.pack(fill=tk.X, pady=(4, 0))

    # ── Expand Panel (file list + player) ─────────────────────────────────

    def _build_expand_panel(self):
        """Build the expandable panel with file history and player controls."""
        self._expand_frame = tk.Frame(self._window, bg=self.PANEL_BG)

        try:
            small_font = tkfont.Font(family="Segoe UI", size=8)
            label_font = tkfont.Font(family="Segoe UI", size=9)
            btn_font = tkfont.Font(family="Segoe UI", size=8)
            mono_font = tkfont.Font(family="Consolas", size=8)
        except Exception:
            small_font = tkfont.Font(size=8)
            label_font = tkfont.Font(size=9)
            btn_font = tkfont.Font(size=8)
            mono_font = tkfont.Font(size=8)

        # ── History header ──
        header = tk.Frame(self._expand_frame, bg=self.PANEL_BG)
        header.pack(fill=tk.X, padx=8, pady=(6, 2))

        tk.Label(header, text="History", font=label_font,
                 fg=self.FG_COLOR, bg=self.PANEL_BG).pack(side=tk.LEFT)

        refresh_btn = tk.Button(
            header, text="Refresh", font=small_font,
            fg="#aaaaaa", bg=self.BTN_COLOR,
            activeforeground=self.FG_COLOR, activebackground=self.BTN_HOVER,
            relief=tk.FLAT, bd=0, padx=6, pady=1, cursor="hand2",
            command=self._refresh_file_list,
        )
        refresh_btn.pack(side=tk.RIGHT)

        # ── File listbox ──
        list_frame = tk.Frame(self._expand_frame, bg=self.PANEL_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 4))

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._file_listbox = tk.Listbox(
            list_frame, font=mono_font,
            bg=self.LIST_BG, fg=self.LIST_FG,
            selectbackground=self.LIST_SEL_BG,
            selectforeground=self.FG_COLOR,
            highlightthickness=0, bd=0,
            activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        self._file_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._file_listbox.yview)
        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        # ── Separator ──
        sep = tk.Frame(self._expand_frame, bg="#444444", height=1)
        sep.pack(fill=tk.X, padx=8, pady=(0, 4))

        # ── Player controls panel ──
        self._player_frame = tk.Frame(self._expand_frame, bg=self.PANEL_BG)
        self._player_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Now playing label
        self._now_playing_label = tk.Label(
            self._player_frame, text="No file selected", font=small_font,
            fg="#888888", bg=self.PANEL_BG, anchor=tk.W,
        )
        self._now_playing_label.pack(fill=tk.X)

        # Progress bar
        progress_frame = tk.Frame(self._player_frame, bg=self.PANEL_BG)
        progress_frame.pack(fill=tk.X, pady=(2, 0))

        self._progress_scale = tk.Scale(
            progress_frame, from_=0, to=100,
            orient=tk.HORIZONTAL, showvalue=False,
            bg=self.PANEL_BG, fg=self.ACCENT,
            troughcolor=self.LIST_BG,
            highlightthickness=0, bd=0,
            sliderrelief=tk.FLAT, sliderlength=12,
            command=self._on_seek,
        )
        self._progress_scale.pack(fill=tk.X)
        self._progress_scale.bind("<ButtonPress-1>", lambda e: self._seek_start())
        self._progress_scale.bind("<ButtonRelease-1>", lambda e: self._seek_end())

        # Time label
        self._time_label = tk.Label(
            self._player_frame, text="00:00 / 00:00", font=small_font,
            fg="#888888", bg=self.PANEL_BG, anchor=tk.W,
        )
        self._time_label.pack(fill=tk.X)

        # Transport buttons row
        transport = tk.Frame(self._player_frame, bg=self.PANEL_BG)
        transport.pack(fill=tk.X, pady=(4, 0))

        btn_style = dict(
            font=btn_font, fg=self.FG_COLOR, bg=self.BTN_COLOR,
            activeforeground=self.FG_COLOR, activebackground=self.BTN_HOVER,
            relief=tk.FLAT, bd=0, pady=2, cursor="hand2",
        )

        skip_back_btn = tk.Button(transport, text="\u00ab15", padx=6,
                                  command=self._on_skip_back, **btn_style)
        skip_back_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._play_btn = tk.Button(transport, text="\u25b6", padx=10,
                                   command=self._on_play_pause, **btn_style)
        self._play_btn.pack(side=tk.LEFT, padx=(0, 4))

        skip_fwd_btn = tk.Button(transport, text="15\u00bb", padx=6,
                                 command=self._on_skip_fwd, **btn_style)
        skip_fwd_btn.pack(side=tk.LEFT, padx=(0, 4))

        stop_btn = tk.Button(transport, text="\u25a0", padx=6,
                             command=self._on_player_stop, **btn_style)
        stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Volume
        tk.Label(transport, text="Vol", font=small_font,
                 fg="#888888", bg=self.PANEL_BG).pack(side=tk.LEFT, padx=(4, 2))

        self._vol_scale = tk.Scale(
            transport, from_=0, to=100,
            orient=tk.HORIZONTAL, showvalue=False,
            length=60,
            bg=self.PANEL_BG, fg=self.ACCENT,
            troughcolor=self.LIST_BG,
            highlightthickness=0, bd=0,
            sliderrelief=tk.FLAT, sliderlength=10,
            command=self._on_volume_change,
        )
        self._vol_scale.set(70)
        self._vol_scale.pack(side=tk.LEFT, padx=(0, 4))

        # ── Action toolbar (below player) ──
        self._build_action_toolbar()

    # ── Action Toolbar ─────────────────────────────────────────────────────

    def _build_action_toolbar(self):
        """Build quick-action buttons and status label below player controls."""
        try:
            btn_font = tkfont.Font(family="Segoe UI", size=8)
            status_font = tkfont.Font(family="Segoe UI", size=8)
        except Exception:
            btn_font = tkfont.Font(size=8)
            status_font = tkfont.Font(size=8)

        # Separator
        sep = tk.Frame(self._expand_frame, bg="#444444", height=1)
        sep.pack(fill=tk.X, padx=8, pady=(4, 4))

        # Button row
        action_row = tk.Frame(self._expand_frame, bg=self.PANEL_BG)
        action_row.pack(fill=tk.X, padx=8, pady=(0, 2))

        btn_style = dict(
            font=btn_font, fg=self.FG_COLOR,
            activeforeground=self.FG_COLOR,
            relief=tk.FLAT, bd=0, pady=2, padx=4, cursor="hand2",
        )

        self._enhance_btn = tk.Button(
            action_row, text="Enhance",
            bg="#cc8822", activebackground="#dd9933",
            command=self._on_enhance_click, **btn_style)
        self._enhance_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._transcribe_btn = tk.Button(
            action_row, text="Transcribe",
            bg="#2266cc", activebackground="#3377dd",
            command=self._on_transcribe_click, **btn_style)
        self._transcribe_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._llm_btn = tk.Button(
            action_row, text="LLM",
            bg="#665599", activebackground="#7766aa",
            command=self._on_llm_click, **btn_style)
        self._llm_btn.pack(side=tk.LEFT, padx=(0, 4))

        audio_btn = tk.Button(
            action_row, text="Audio",
            bg=self.BTN_COLOR, activebackground=self.BTN_HOVER,
            command=self._on_open_audio_folder, **btn_style)
        audio_btn.pack(side=tk.LEFT, padx=(0, 4))

        transcripts_btn = tk.Button(
            action_row, text="Txts",
            bg=self.BTN_COLOR, activebackground=self.BTN_HOVER,
            command=self._on_open_transcripts_folder, **btn_style)
        transcripts_btn.pack(side=tk.LEFT)

        # Status row
        status_row = tk.Frame(self._expand_frame, bg=self.PANEL_BG)
        status_row.pack(fill=tk.X, padx=8, pady=(0, 6))

        self._status_label = tk.Label(
            status_row, text="Ready", font=status_font,
            fg=self.FG_DIM, bg=self.PANEL_BG, anchor=tk.W)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._diag_btn = tk.Button(
            status_row, text="Diag", font=btn_font,
            fg=self.FG_COLOR, bg=self.BTN_COLOR,
            activeforeground=self.FG_COLOR, activebackground=self.BTN_HOVER,
            relief=tk.FLAT, bd=0, padx=6, pady=1, cursor="hand2",
            command=self._on_diag_click)
        self._diag_btn.pack(side=tk.RIGHT)

    FG_DIM = "#888888"

    # ── Action handlers ────────────────────────────────────────────────────

    def _get_selected_audio_path(self):
        """Return the Path of the currently selected audio file, or None."""
        if not self._file_listbox:
            return None
        sel = self._file_listbox.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx >= len(self._file_list_data):
            return None
        return self._file_list_data[idx]["path"]

    def _get_transcript_path(self, audio_path: Path) -> Path:
        """Infer transcript path from audio path: ./audio/foo.wav -> ./audio/transcripts/foo.txt"""
        transcripts_dir = audio_path.parent / "transcripts"
        return transcripts_dir / (audio_path.stem + ".txt")

    def _on_enhance_click(self):
        if self._recording_active:
            return
        if not self._pipeline_runner:
            return
        if self._pipeline_runner.is_running:
            return

        audio_path = self._get_selected_audio_path()
        if not audio_path:
            self._set_status("Select an audio file first")
            return

        self._set_action_buttons_state("disabled")
        self._pipeline_runner.enhance(audio_path)

    def _on_transcribe_click(self):
        if self._recording_active:
            return
        if not self._pipeline_runner:
            return
        if self._pipeline_runner.is_running:
            return

        audio_path = self._get_selected_audio_path()
        if not audio_path:
            self._set_status("Select an audio file first")
            return

        self._set_action_buttons_state("disabled")
        self._pipeline_runner.transcribe(audio_path)

    def _on_llm_click(self):
        if self._recording_active:
            return
        if not self._pipeline_runner:
            return
        if self._pipeline_runner.is_running:
            return

        audio_path = self._get_selected_audio_path()
        if not audio_path:
            self._set_status("Select an audio file first")
            return

        transcript = self._get_transcript_path(audio_path)
        if not transcript.exists():
            self._set_status("No transcript found — transcribe first")
            return

        glossary = audio_path.parent.parent / "glossary.txt"
        self._set_action_buttons_state("disabled")
        self._pipeline_runner.run_llm(transcript,
                                       glossary=glossary if glossary.exists() else None)

    def _on_open_audio_folder(self):
        folder = self._audio_dir.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _on_open_transcripts_folder(self):
        folder = (self._audio_dir / "transcripts").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _on_diag_click(self):
        if self._on_open_diagnostics:
            self._on_open_diagnostics()

    def _set_status(self, msg: str, fg=None):
        """Update the status label text."""
        if self._status_label:
            self._status_label.config(text=msg, fg=fg or self.FG_DIM)

    def _set_action_buttons_state(self, state: str):
        """Enable or disable enhance/transcribe/LLM buttons."""
        for btn in (self._enhance_btn, self._transcribe_btn, self._llm_btn):
            if btn:
                btn.config(state=state)

    # ── Pipeline queue commands ────────────────────────────────────────────

    def pipeline_status(self, msg: str):
        """Thread-safe: push pipeline status update."""
        self._command_queue.put(("pipeline_status", msg))

    def pipeline_complete(self, success: bool, msg: str):
        """Thread-safe: push pipeline completion."""
        self._command_queue.put(("pipeline_complete", success, msg))

    # ── Pill (minimized) ──────────────────────────────────────────────────

    def _build_pill(self):
        """Build the minimized pill window."""
        self._pill_window = tk.Toplevel(self._root)
        self._pill_window.title("SM")
        self._pill_window.overrideredirect(True)
        self._pill_window.attributes("-topmost", True)
        self._pill_window.attributes("-toolwindow", True)
        self._pill_window.configure(bg=self.BG_COLOR)
        self._pill_window.withdraw()

        try:
            pill_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        except Exception:
            pill_font = tkfont.Font(size=9, weight="bold")

        pill_label = tk.Label(
            self._pill_window, text="SM", font=pill_font,
            fg=self.ACCENT, bg=self.BG_COLOR, cursor="hand2",
            padx=8, pady=2,
        )
        pill_label.pack(fill=tk.BOTH, expand=True)
        pill_label.bind("<Button-1>", lambda e: self._do_restore())

        # Pill drag support
        self._pill_drag = {"x": 0, "y": 0}
        pill_label.bind("<ButtonPress-1>", self._on_pill_drag_start)
        pill_label.bind("<B1-Motion>", self._on_pill_drag_motion)

    def _on_pill_drag_start(self, event):
        self._pill_drag["x"] = event.x
        self._pill_drag["y"] = event.y

    def _on_pill_drag_motion(self, event):
        # Only move if dragged more than 5px (otherwise treat as click)
        dx = abs(event.x - self._pill_drag["x"])
        dy = abs(event.y - self._pill_drag["y"])
        if dx > 5 or dy > 5:
            x = self._pill_window.winfo_x() + (event.x - self._pill_drag["x"])
            y = self._pill_window.winfo_y() + (event.y - self._pill_drag["y"])
            self._pill_window.geometry(f"+{x}+{y}")

    # ── Frame switching ─────────────────────────────────────────────────

    def _show_frame(self, state):
        self._state = state
        if state == "idle":
            self._rec_frame.pack_forget()
            self._idle_frame.pack(fill=tk.X)
            if self._widget_mode == "expanded":
                self._expand_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self._idle_frame.pack_forget()
            self._expand_frame.pack_forget()
            self._rec_frame.pack(fill=tk.BOTH, expand=True)

    # ── Expand / Collapse ─────────────────────────────────────────────────

    def _toggle_expand(self):
        if self._widget_mode == "expanded":
            self._do_collapse()
        else:
            self._do_expand()

    def _do_expand(self):
        self._widget_mode = "expanded"
        self._chevron_label.config(text="\u25b2")
        self._expand_frame.pack(fill=tk.BOTH, expand=True)
        self._apply_geometry()
        self._refresh_file_list()
        self._start_player_updates()

    def _do_collapse(self):
        self._widget_mode = "compact"
        self._chevron_label.config(text="\u25bc")
        self._expand_frame.pack_forget()
        self._apply_geometry()
        self._stop_player_updates()

    # ── Minimize / Restore ────────────────────────────────────────────────

    def _do_minimize(self):
        """Minimize to pill."""
        prev_mode = self._widget_mode
        self._widget_mode = "pill"

        # Position pill at bottom-right of where the window was
        wx = self._window.winfo_x()
        wy = self._window.winfo_y()
        ww = self._window.winfo_width()
        wh = self._window.winfo_height()
        pill_x = wx + ww - self.PILL_WIDTH
        pill_y = wy + wh - self.PILL_HEIGHT

        self._window.withdraw()
        self._pill_window.geometry(f"{self.PILL_WIDTH}x{self.PILL_HEIGHT}+{pill_x}+{pill_y}")
        self._pill_window.deiconify()
        # Store previous mode for restore
        self._pre_pill_mode = prev_mode

    def _do_restore(self):
        """Restore from pill."""
        self._pill_window.withdraw()
        self._widget_mode = getattr(self, "_pre_pill_mode", "compact")
        self._window.deiconify()
        self._apply_geometry()

    # ── Public API (thread-safe via queue) ────────────────────────────────

    def show_idle(self):
        """Show widget in idle state with Start buttons. Thread-safe."""
        self._command_queue.put(("show_idle",))

    def show_recording(self, mode="meeting", elapsed_getter=None):
        """Show widget in recording state. Thread-safe."""
        self._command_queue.put(("show_recording", mode, elapsed_getter))

    def hide(self):
        """Hide the widget. Thread-safe."""
        self._command_queue.put(("hide",))

    def set_recording_active(self, active: bool):
        """Enable/disable playback controls based on recording state. Thread-safe."""
        self._command_queue.put(("set_recording_active", active))

    def refresh_files(self):
        """Trigger file list refresh. Thread-safe."""
        self._command_queue.put(("refresh_files",))

    # Keep old API as alias
    def show(self, mode="meeting", elapsed_getter=None):
        self.show_recording(mode, elapsed_getter)

    # ── Command queue polling ─────────────────────────────────────────────

    def _poll_commands(self):
        try:
            while True:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == "show_idle":
                    self._do_show_idle()
                elif cmd[0] == "show_recording":
                    self._do_show_recording(cmd[1], cmd[2] if len(cmd) > 2 else None)
                elif cmd[0] == "hide":
                    self._do_hide()
                elif cmd[0] == "set_recording_active":
                    self._do_set_recording_active(cmd[1])
                elif cmd[0] == "refresh_files":
                    self._refresh_file_list()
                elif cmd[0] == "pipeline_status":
                    self._set_status(cmd[1])
                elif cmd[0] == "pipeline_complete":
                    success, msg = cmd[1], cmd[2]
                    if success:
                        self._set_status(msg, fg="#44bb44")
                    else:
                        self._set_status(msg, fg="#dd3333")
                    self._set_action_buttons_state("normal")
        except queue.Empty:
            pass
        if self._root:
            self._root.after(100, self._poll_commands)

    def _do_show_idle(self):
        self._show_frame("idle")
        if self._widget_mode == "pill":
            return  # stay minimized
        self._window.deiconify()
        self._apply_geometry()
        self._visible = True

    def _do_show_recording(self, mode, elapsed_getter):
        self._elapsed_getter = elapsed_getter
        mode_text = "Meeting recording (L=mic, R=system)" if mode == "meeting" else "Dictation (mic only)"
        self._mode_label_text.config(text=mode_text)
        self._timer_label.config(text="00:00")
        self._dot_visible = True
        self._recording_active = True
        # Collapse expanded panel during recording
        if self._widget_mode == "expanded":
            self._do_collapse()
        self._show_frame("recording")
        if self._widget_mode == "pill":
            self._do_restore()
        self._window.deiconify()
        self._apply_geometry()
        self._visible = True
        self._update_timer()

    def _do_hide(self):
        self._visible = False
        self._elapsed_getter = None
        self._window.withdraw()

    def _do_set_recording_active(self, active: bool):
        self._recording_active = active

    # ── Timer update ──────────────────────────────────────────────────────

    def _update_timer(self):
        if not self._visible or self._state != "recording":
            return
        if self._elapsed_getter:
            elapsed = self._elapsed_getter()
            mm, ss = divmod(int(elapsed), 60)
            hh, mm_r = divmod(mm, 60)
            if hh > 0:
                self._timer_label.config(text=f"{hh:d}:{mm_r:02d}:{ss:02d}")
            else:
                self._timer_label.config(text=f"{mm:02d}:{ss:02d}")

        # Blink recording dot
        self._dot_visible = not self._dot_visible
        color = self.REC_COLOR if self._dot_visible else self.BG_COLOR
        self._dot_canvas.itemconfig(self._rec_dot, fill=color)

        self._root.after(500, self._update_timer)

    # ── File list ─────────────────────────────────────────────────────────

    def _refresh_file_list(self):
        """Reload file list from audio directory."""
        if not self._player:
            return

        from audio_player import AudioPlayer
        self._file_list_data = AudioPlayer.list_audio_files(self._audio_dir)
        self._file_listbox.delete(0, tk.END)

        for entry in self._file_list_data:
            line = self._format_file_entry(entry)
            self._file_listbox.insert(tk.END, line)

        # Highlight currently loaded file
        if self._player and self._player.loaded_path:
            for i, entry in enumerate(self._file_list_data):
                if entry["path"] == self._player.loaded_path:
                    self._file_listbox.selection_set(i)
                    break

    @staticmethod
    def _format_file_entry(entry) -> str:
        """Format a file entry for the listbox."""
        path = entry["path"]
        dur = entry["duration_s"]
        size = entry["size_mb"]
        mode = entry["mode"]

        # Date from file mtime
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        date_str = mtime.strftime("%m/%d %H:%M")

        # Duration format
        if dur >= 3600:
            h, remainder = divmod(int(dur), 3600)
            m, s = divmod(remainder, 60)
            dur_str = f"{h}:{m:02d}:{s:02d}"
        elif dur > 0:
            m, s = divmod(int(dur), 60)
            dur_str = f"{m}:{s:02d}"
        else:
            dur_str = "?"

        # Size format
        if size >= 1.0:
            size_str = f"{size:.1f}M"
        else:
            size_str = f"{int(size * 1024)}K"

        # Mode label
        if mode == "recording":
            mode_str = "rec"
        elif mode == "dictation":
            mode_str = "dic"
        else:
            # Use truncated filename for non-standard files
            mode_str = entry["name"][:16]

        return f"{date_str}  {mode_str:<6s} {dur_str:>8s} {size_str:>6s}"

    def _on_file_select(self, event):
        """Handle file selection from listbox."""
        if self._recording_active:
            return
        if not self._player:
            return

        sel = self._file_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._file_list_data):
            return

        entry = self._file_list_data[idx]
        try:
            self._player.load(entry["path"])
            self._player.play()
            self._update_now_playing(entry)
        except Exception as e:
            print(f"[Widget] Cannot play: {e}")

    def _update_now_playing(self, entry):
        """Update the 'Now playing' label."""
        name = entry["name"]
        if len(name) > 30:
            name = name[:27] + "..."
        self._now_playing_label.config(text=f"Now playing: {name}", fg=self.FG_COLOR)

    # ── Player controls ───────────────────────────────────────────────────

    def _on_play_pause(self):
        if self._recording_active or not self._player:
            return
        if self._player.state == "playing":
            self._player.pause()
        elif self._player.state == "paused":
            self._player.play()
        elif self._player.loaded_path:
            self._player.play()

    def _on_player_stop(self):
        if not self._player:
            return
        self._player.stop()

    def _on_skip_back(self):
        if self._recording_active or not self._player:
            return
        self._player.skip(-15)

    def _on_skip_fwd(self):
        if self._recording_active or not self._player:
            return
        self._player.skip(15)

    def _on_volume_change(self, value):
        if self._player:
            self._player.volume = int(value) / 100.0

    def _seek_start(self):
        self._seeking = True

    def _seek_end(self):
        self._seeking = False

    def _on_seek(self, value):
        if not self._player or not self._player.loaded_path:
            return
        if self._seeking:
            pos = float(value) / 100.0 * self._player.duration
            self._player.seek(pos)

    # ── Player UI polling ─────────────────────────────────────────────────

    def _start_player_updates(self):
        if self._player_update_id is None:
            self._update_player_ui()

    def _stop_player_updates(self):
        if self._player_update_id is not None:
            self._root.after_cancel(self._player_update_id)
            self._player_update_id = None

    def _update_player_ui(self):
        """Poll player state and update UI every 250ms."""
        if self._player and self._player.loaded_path:
            pos = self._player.position
            dur = self._player.duration
            state = self._player.state

            # Update time label
            self._time_label.config(text=f"{self._fmt_time(pos)} / {self._fmt_time(dur)}")

            # Update progress bar (only if not dragging)
            if not self._seeking and dur > 0:
                pct = (pos / dur) * 100.0
                self._progress_scale.set(pct)

            # Update play/pause button text
            if state == "playing":
                self._play_btn.config(text="\u275a\u275a")  # pause symbol
            else:
                self._play_btn.config(text="\u25b6")  # play symbol

        self._player_update_id = self._root.after(250, self._update_player_ui)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        s = max(0, int(seconds))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    # ── Button handlers ───────────────────────────────────────────────────

    def _on_meeting_click(self):
        if self._on_start_meeting:
            self._on_start_meeting()

    def _on_dictation_click(self):
        if self._on_start_dictation:
            self._on_start_dictation()

    def _on_stop_click(self):
        if self._on_stop:
            self._on_stop()

    def _on_close(self):
        """Close button — quit the app."""
        if self._on_stop:
            self._on_stop()  # stop recording if active
        self._root.quit()
