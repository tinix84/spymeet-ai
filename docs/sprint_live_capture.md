# Sprint Plan: Live Audio Capture

**Feature**: System audio loopback + microphone recording for SpyMeet
**Priority**: Phase 1.5 (between audio enhancement and batch processing)
**Estimated scope**: 4 stories, ~6 files

---

## 1. Goal

Add a desktop recorder that captures **system audio** (WASAPI loopback — everything playing through speakers/headphones) and **microphone input** simultaneously, saving as a stereo WAV file (`L=mic, R=loopback`). Manual start/stop via system tray icon + floating widget.

No real-time transcription — the recorded file feeds into the existing pipeline (`run.ps1`).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                  record.py (core)                    │
│                                                      │
│  ┌──────────────┐    ┌──────────────────┐           │
│  │  Mic stream   │    │  Loopback stream  │           │
│  │  (WASAPI)     │    │  (WASAPI loopback)│           │
│  └──────┬───────┘    └────────┬─────────┘           │
│         │                     │                      │
│         ▼                     ▼                      │
│  ┌──────────────────────────────────────┐           │
│  │     Mixer thread                     │           │
│  │  Resample both to 48kHz → interleave │           │
│  │  L = mic, R = loopback               │           │
│  └──────────────┬───────────────────────┘           │
│                 ▼                                    │
│  ┌──────────────────────────────┐                   │
│  │  WAV writer (16-bit stereo)  │                   │
│  │  → ./audio/YYYY-MM-DD_HHMM_recording.wav        │
│  └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────┐   ┌────────────────────────┐
│  recorder_tray.py       │   │  recorder_widget.py    │
│  (pystray)              │   │  (tkinter)             │
│                         │   │                        │
│  System tray icon       │   │  Floating window       │
│  • Right-click menu     │   │  • always-on-top       │
│  • Start / Stop         │   │  • elapsed timer       │
│  • Red dot = recording  │   │  • Stop button         │
│  • Quit                 │   │  • appears on record   │
│  • Settings             │   │  • hides on stop       │
└─────────────────────────┘   └────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  recorder_app.py  (entry point)                      │
│  Launches tray + widget + core, coordinates events   │
└──────────────────────────────────────────────────────┘
```

**Post-recording flow (manual, existing pipeline):**
```
./audio/2026-02-28_1430_recording.wav
  → audio_enhance.py  (normalize, denoise, EQ, compress — needs mono split first)
  → transcribe.py     (Groq API / WhisperX)
  → llm_process.py    (correction + summary)
```

---

## 3. Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Audio capture lib | **PyAudioWPatch** | Only Python lib that supports WASAPI loopback. sounddevice does NOT support it. |
| Sample rate | **48000 Hz** (device default) | WASAPI requires matching the device's native rate. Most Win11 devices default to 48000. Query at runtime. |
| WASAPI mode | **Shared** (not exclusive) | Loopback only works in shared mode. |
| Output format | **Stereo WAV, 16-bit PCM** | L=mic, R=loopback. Lossless, compatible with ffmpeg/soundfile. |
| System tray | **pystray** | Lightweight, Windows-native, thread-safe. |
| Floating widget | **tkinter** | Built-in, no extra deps. `attributes('-topmost', True)` works on Win11. |
| File naming | **`YYYY-MM-DD_HHMM_recording.wav`** | Auto-generated timestamp. User can rename later. |
| Downsampling | **Deferred to pipeline** | Capture at native rate (48kHz). `audio_enhance.py` already downsamples to 16kHz. |

---

## 4. Stories

### Story 1: Core recording engine (`record.py`)

**Goal**: Capture mic + system loopback → stereo WAV.

**Tasks**:
1. Enumerate WASAPI devices: find default loopback + default mic
2. Open two concurrent input streams (PyAudioWPatch)
3. Handle sample rate mismatch: if mic ≠ loopback rate, resample mic to match loopback (scipy.signal.resample)
4. Mixer thread: read from both streams, interleave into stereo frames (L=mic, R=loopback)
5. WAV writer: `soundfile.SoundFile` in write mode, flush periodically
6. Start/stop API: `Recorder.start()`, `Recorder.stop()` → returns output path
7. Error handling: device not found, stream errors, disk full
8. CLI mode: `python record.py --list-devices`, `python record.py --start` (Ctrl+C to stop)

**Key code patterns**:
```python
import pyaudiowpatch as pyaudio

# Find loopback device
p = pyaudio.PyAudio()
wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
for loopback in p.get_loopback_device_info_generator():
    if default_output["name"] in loopback["name"]:
        loopback_device = loopback
        break
```

**Acceptance criteria**:
- [ ] `python record.py --list-devices` shows mic + loopback devices
- [ ] `python record.py --start` captures both streams to stereo WAV
- [ ] Output plays correctly: left = my voice, right = system audio
- [ ] Handles graceful shutdown (Ctrl+C, `.stop()`)
- [ ] WAV file is valid and readable by ffmpeg/soundfile

---

### Story 2: System tray icon (`recorder_tray.py`)

**Goal**: Persistent system tray icon with right-click menu.

**Tasks**:
1. Create tray icon with pystray (PIL-generated icon)
2. Two icon states: idle (gray/green) and recording (red dot)
3. Right-click menu: Start Recording / Stop Recording / Quit
4. Start Recording → calls `Recorder.start()` + swaps icon to red + shows widget
5. Stop Recording → calls `Recorder.stop()` + swaps icon to idle + hides widget
6. Windows notification on start/stop ("Recording started", "Saved to ./audio/...")
7. Quit → stops recording if active, cleans up, exits

**Acceptance criteria**:
- [ ] Tray icon visible in Windows 11 taskbar
- [ ] Menu items enable/disable correctly (can't start twice, can't stop when idle)
- [ ] Icon changes color when recording
- [ ] Notification shows output file path on stop

---

### Story 3: Floating recording widget (`recorder_widget.py`)

**Goal**: Small always-on-top window during recording.

**Tasks**:
1. tkinter window, always-on-top, small (250×80px), no resize
2. Shows: red "REC" indicator + elapsed time (MM:SS) + Stop button
3. Timer updates every second
4. Stop button → calls `Recorder.stop()`, hides window
5. Window appears when recording starts, hides when stopped
6. Position: bottom-right corner of screen (above taskbar)
7. Minimal/dark theme to not distract

**Layout**:
```
┌──────────────────────────┐
│  🔴 REC   03:45    [■]  │
└──────────────────────────┘
```

**Acceptance criteria**:
- [ ] Window stays on top of all other windows
- [ ] Timer counts up accurately
- [ ] Stop button works
- [ ] Window hides/shows correctly with recording state

---

### Story 4: App entry point + pipeline integration

**Goal**: Single entry point + make the pipeline handle stereo recordings.

**Tasks**:
1. `recorder_app.py` — launches tray + widget + recorder, coordinates events
2. Modify `audio_enhance.py` to handle stereo input:
   - Detect stereo WAV → split to L/R channels
   - Enhance each channel separately (or mix to mono first)
   - Save enhanced mono WAV (mixed or channel-selectable)
3. Update `setup.ps1` — add `PyAudioWPatch`, `pystray`, `Pillow` to deps
4. Update `CLAUDE.md` and `architecture.md` with new component
5. Add `--channel` flag to `transcribe.py`: `--channel both|left|right|mix` (default: mix)
   - `both`: process L and R separately (two transcripts)
   - `left`/`right`: process one channel only
   - `mix`: downmix to mono (current behavior, default)

**Acceptance criteria**:
- [ ] `python recorder_app.py` launches tray icon and is ready to record
- [ ] End-to-end: record → enhance → transcribe works on a stereo recording
- [ ] `--channel right` processes only the loopback (remote speaker)

---

## 5. Dependencies

### New pip packages

```
PyAudioWPatch>=0.2.12    # WASAPI loopback capture (only lib that supports it)
pystray>=0.19.0          # System tray icon
Pillow>=9.0.0            # Icon image generation for pystray
```

### Already available
- `soundfile` — WAV writing (installed in Phase 1)
- `scipy` — resampling (installed in Phase 1)
- `numpy` — audio buffer manipulation
- `tkinter` — built into Python stdlib

---

## 6. File Plan

| File | Action | Description |
|------|--------|-------------|
| `record.py` | CREATE | Core recording engine: WASAPI loopback + mic → stereo WAV |
| `recorder_tray.py` | CREATE | System tray icon with pystray |
| `recorder_widget.py` | CREATE | Floating tkinter recording widget |
| `recorder_app.py` | CREATE | Entry point: launches tray + widget + recorder |
| `audio_enhance.py` | MODIFY | Handle stereo input (split channels or downmix) |
| `transcribe.py` | MODIFY | Add `--channel` flag for stereo recordings |
| `setup.ps1` | MODIFY | Add PyAudioWPatch, pystray, Pillow |
| `CLAUDE.md` | MODIFY | Document recorder component |
| `architecture.md` | MODIFY | Add recorder to system diagram |

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| WASAPI loopback captures ALL system audio (notifications, YouTube, etc.) | Noisy recordings | Document: mute other apps during calls. Future: per-app capture via Windows Audio Session API (complex). |
| Mic and loopback have different sample rates (e.g., mic=44100, loopback=48000) | Misaligned stereo channels | Resample mic stream to match loopback rate using scipy before interleaving. |
| Long recordings → large files (48kHz stereo 16-bit ≈ 11 MB/min, 660 MB/hour) | Disk space | Warn user on start. Future: add FLAC compression option. |
| PyAudioWPatch not available for Python 3.13+ | Install failure | Has wheels for 3.7-3.13. Check compatibility. Fallback: build from source. |
| tkinter + pystray threading conflicts | UI freeze/crash | pystray runs in daemon thread, tkinter runs in main thread. Use queue for cross-thread communication. |
| Some audio devices don't expose loopback | Can't record system audio | Graceful error: "No loopback device found. Check your audio output device." List available devices. |

---

## 8. Testing Plan

### Manual tests (real hardware required)

1. **Device enumeration**: `python record.py --list-devices` shows mic + loopback
2. **Short recording**: Record 30s of a YouTube video → verify R channel has audio, L channel is silent (no mic input)
3. **Both channels**: Record yourself talking while playing audio → verify L=voice, R=system
4. **Sample rate mismatch**: Force mic to 44100, loopback at 48000 → verify resampling works
5. **Long recording**: Record for 30+ minutes → verify no gaps, memory stable
6. **Pipeline integration**: Feed stereo WAV through `audio_enhance.py` → `transcribe.py` → verify output
7. **Tray + widget**: Start/stop via tray menu, verify widget appears/hides, timer counts correctly
8. **Edge cases**: No mic connected, no audio output device, start twice, stop when not recording

### Automated tests (pytest)

- `test_record.py`: Mock PyAudioWPatch streams, verify stereo interleaving, WAV header, resampling logic
- Test `audio_enhance.py` stereo handling: stereo input → mono output, channel selection

---

## 9. Out of Scope (for this sprint)

- Per-app audio capture (Windows Audio Session API — very complex)
- Hotkey support (global keyboard shortcut to start/stop)
- Auto-detect call start/stop
- Real-time transcription during recording
- Audio level meters in the widget
- Configurable output format (FLAC, MP3)
- Multiple simultaneous recordings

These can be follow-up stories.

---

## 10. Definition of Done

- [ ] `python recorder_app.py` launches tray icon on Windows 11
- [ ] Right-click → Start Recording → captures mic + system audio
- [ ] Floating widget shows elapsed time
- [ ] Right-click → Stop Recording → saves stereo WAV to `./audio/`
- [ ] `python record.py --start` works as standalone CLI
- [ ] Recorded WAV plays back correctly (L=mic, R=system)
- [ ] Full pipeline works: recorded WAV → enhance → transcribe → summary
- [ ] New deps added to `setup.ps1`
- [ ] Docs updated (CLAUDE.md, architecture.md)
