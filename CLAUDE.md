# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpyMeet is a meeting audio transcription pipeline: audio -> (optional) audio enhancement -> speech-to-text -> LLM correction -> structured summary. Runs on Windows with PowerShell + Python (conda).

## Architecture

```
recorder_app.py           (live capture: WASAPI loopback + mic → stereo WAV)
    -> ./audio/YYYY-MM-DD_HHMM_recording.wav  (L=mic, R=system audio)

Audio files (./audio/)
    -> audio_enhance.py   (always-on: normalize, denoise, EQ, compress)
    -> [name]_enhanced.wav
    -> transcribe.py      (speech-to-text: WhisperX CPU / OpenAI API / Groq API)
    -> [name].txt + [name].json  (in ./audio/transcripts/)
    -> llm_process.py     (Claude API: correction + summary)
    -> [name]_corrected.txt + [name]_summary.md + [name]_metrics.md
```

**record.py** -- Core recording engine. `Recorder` class with two modes: `meeting` (stereo: L=mic, R=WASAPI loopback, 48kHz 16-bit) and `dictation` (mono: mic only, for LLM prompts). Threaded mixer handles sample rate mismatch. CLI: `--list-devices`, `--start`, `--mode dictation`.

**recorder_app.py** -- Entry point for desktop recorder. Coordinates system tray icon (pystray, daemon thread), floating widget (tkinter, main thread), and Recorder instance via callbacks. Launch: `python recorder_app.py` or `python recorder_app.py --mode dictation`.

**recorder_tray.py** -- System tray icon (pystray). PIL-generated icons: gray=idle, red=recording. Menu: Start Meeting / Start Dictation / Stop / Quit.

**recorder_widget.py** -- Floating tkinter widget (always-on-top, dark theme). Shows blinking REC dot, elapsed timer, mode label, and Stop button. Positioned bottom-right above taskbar.

**audio_enhance.py** -- Always-on audio preprocessing. Chain: load (soundfile/ffmpeg) -> EBU R128 normalization (-16 LUFS) -> spectral gating noise reduction -> speech EQ (HP 80Hz + peak 3kHz) -> dynamic compression (3:1). Output: 16-bit PCM WAV, 16kHz mono. Graceful fallback if deps missing (pyloudnorm, noisereduce, scipy, soundfile). Skips if `_enhanced.wav` is up-to-date. Supports `channel` parameter (`mix`/`left`/`right`) for stereo recordings.

**transcribe.py** -- Three backends: `cpu` (WhisperX local), `openai-api` (cloud), `groq-api` (free tier). Speaker diarization only available with WhisperX + HF_TOKEN. Groq API returns segments as dicts (not objects) -- use `isinstance(seg, dict)` checks. Enhancement runs automatically in `main()` before backend dispatch; `stem_map` ensures transcripts use original filenames. `--channel` flag (`mix`/`left`/`right`/`both`) for stereo recordings; `both` processes L and R separately with `_mic`/`_system` suffixes.

**llm_process.py** -- Two-stage Claude pipeline: (1) chunk transcript into ~5min segments, correct fillers/punctuation, retry warned segments up to 3x; (2) generate Markdown meeting summary. Uses `claude-haiku-4-5-20251001`.

**glossary.txt** -- One term per line, optionally with `=` descriptions. Fed to LLM for domain-specific term correction.

## Environment

- **Python env**: conda `social_env` (Python 3.13+), NOT venv
- **ffmpeg**: installed via conda (`conda install -c conda-forge ffmpeg`), path at `~/.conda/envs/social_env/Library/bin/`
- **KMP_DUPLICATE_LIB_OK=TRUE** needed for torch in conda
- **_run_transcribe.ps1**: helper script with hardcoded paths for Windows PATH/env setup (machine-specific, gitignored)

## Commands

### Recording
```powershell
python recorder_app.py                                # launch tray icon + widget (idle)
python recorder_app.py --mode meeting                 # launch and auto-start meeting recording
python recorder_app.py --mode dictation               # launch and auto-start dictation
python record.py --list-devices                       # list available audio devices
python record.py --start                              # CLI meeting recording (Ctrl+C to stop)
python record.py --start --mode dictation             # CLI dictation mode (mic only)
```

### Running the pipeline
```powershell
.\run.ps1 -Language it                                # full pipeline (transcribe + LLM)
.\run.ps1 -Backend groq-api -Language it              # Groq free tier (fast, recommended)
.\run.ps1 -Backend openai-api -Language de            # OpenAI Whisper API
.\run.ps1 -SkipLLM -Language it                       # transcribe only
.\run.ps1 -LLMOnly -Input .\audio\transcripts         # LLM-only on existing .txt
.\run.ps1 -Glossary .\glossary.txt -Language it       # with domain terminology
```

### Channel selection (stereo recordings)
```powershell
python transcribe.py --input ./audio/rec.wav --channel right    # system audio only
python transcribe.py --input ./audio/rec.wav --channel left     # mic only
python transcribe.py --input ./audio/rec.wav --channel both     # L and R separately (_mic/_system)
```

### Standalone LLM processing
```bash
python llm_process.py --input ./audio/transcripts/meeting.txt --glossary glossary.txt
```

## Environment Variables (stored in .env, loaded by _run_transcribe.ps1)

| .env key | Env var | Required for |
|----------|---------|-------------|
| `anthropic_api` | `ANTHROPIC_API_KEY` | LLM correction/summary |
| `groq_api` | `GROQ_API_KEY` | groq-api backend |
| `openai_api` | `OPENAI_API_KEY` | openai-api backend |
| `hf_token` | `HF_TOKEN` | Speaker diarization (pyannote) |

## Key Details

- Timestamp regexes use `\d{2,}:\d{2}` (not `\d{2}:\d{2}`) to support recordings >99 minutes
- `run_groq_api()` must be defined ABOVE `if __name__ == "__main__"` in transcribe.py
- Correction retry logic accumulates non-warned segments in a separate `accepted` list
- Filler words removed: Italian (allora, quindi, cioe, praticamente), German (naja, ahm, sozusagen), universal (uhm, uh)
- Output language matches transcript language for both correction and summary
- Audio files and transcripts are gitignored (large binaries / generated output)
- Audio enhancement is always-on; `enhance_audio_files()` returns `(enhanced_files, stem_map)` — `stem_map` maps enhanced paths to original stems so transcripts keep original filenames
- Enhancement deps (pyloudnorm, noisereduce, scipy, soundfile) are optional — if missing, transcribe.py falls back to raw audio with a warning
- Live capture uses PyAudioWPatch (only Python lib supporting WASAPI loopback); sounddevice does NOT support it
- WASAPI loopback captures ALL system audio — mute other apps during calls for clean recordings
- Stereo recordings: L=mic, R=loopback. Pipeline downmixes to mono by default. Use `--channel` to select specific channels
- Dictation mode records mic-only mono WAV for voice prompts / LLM input
- Recorder deps: PyAudioWPatch, pystray, Pillow (PIL). All optional — only needed for live capture
- Recording widget uses tkinter mainloop on main thread; pystray runs in daemon thread; communication via queue.Queue
