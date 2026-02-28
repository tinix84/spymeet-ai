# SpyMeet -- Product Requirements Document

## 1. Vision

SpyMeet transforms raw meeting audio into actionable, structured intelligence. It automates the full chain from audio capture to corrected transcript, speaker identification, and meeting summary -- minimizing manual effort while maximizing accuracy and traceability.

## 2. Current State (v0.1)

### What exists

| Component | Status | Description |
|-----------|--------|-------------|
| Audio transcription | Working | Three backends: WhisperX (local CPU), OpenAI Whisper API, Groq API (free tier) |
| Speaker diarization | Partial | Only available with WhisperX + HuggingFace token. API backends output single-speaker |
| LLM correction | Working | Claude Haiku corrects punctuation, removes fillers, flags unintelligible segments with retry logic |
| Meeting summary | Working | Structured Markdown: participants, decisions, action items, open questions |
| Correction metrics | Working | Confidence scores, word-level diffs, semantic preservation scores per segment |
| Glossary support | Working | Domain-specific term correction via glossary.txt |
| Windows pipeline | Working | PowerShell scripts (check_gpu, setup, run) with conda environment |

### Known limitations

- No audio preprocessing -- raw audio goes directly to transcription
- API backends (Groq, OpenAI) cannot diarize speakers
- No batch analytics across multiple meetings
- Speaker identification is manual (LLM-inferred from context, not voice-based)
- No web interface -- CLI only
- LLM step requires separate ANTHROPIC_API_KEY (not included in Claude Code Max OAuth)

## 3. Roadmap

### Phase 1: Audio Enhancement Pipeline

**Goal**: Improve transcription accuracy by preprocessing audio before speech-to-text.

**Behavior**: Always-on (runs automatically before transcription). Saves both original and enhanced audio (`[name]_enhanced.wav`).

#### Processing steps (in order)

1. **Loudness normalization (EBU R128)**
   - Target: -23 LUFS (broadcast standard) or -16 LUFS (podcast/speech)
   - Balances volume between speakers (critical for phone calls where one side is louder)
   - Library: `pyloudnorm`

2. **Noise reduction (spectral gating)**
   - Estimate noise profile from first 0.5-1s of audio (typically silence/ambient)
   - Apply spectral gating to reduce stationary background noise
   - Configurable aggressiveness (default: moderate, preserve speech naturalness)
   - Library: `noisereduce`

3. **Speech EQ**
   - High-pass filter at 80 Hz (remove rumble, HVAC, handling noise)
   - Presence boost: gentle +2-3 dB shelf around 2-4 kHz (improve speech clarity)
   - De-essing: optional, reduce sibilance if detected
   - Library: `scipy.signal` for filters

4. **Dynamic compression**
   - Reduce dynamic range so quiet speech is audible and loud parts don't clip
   - Attack: 10ms, Release: 100ms, Ratio: 3:1, Threshold: adaptive
   - Makeup gain to target loudness after compression
   - Library: custom implementation or `pydub`

#### Output

- `audio/[name]_enhanced.wav` -- enhanced audio file (16-bit PCM, 16kHz mono)
- Console log with before/after metrics: LUFS, peak dB, estimated SNR
- Transcription runs on the enhanced file

#### Dependencies

```
pyloudnorm>=0.1.0
noisereduce>=3.0.0
scipy>=1.10.0
soundfile>=0.12.0
```

### Phase 1.5: Live Audio Capture

**Goal**: Eliminate manual audio file export by capturing system audio (calls, meetings) directly from Windows.

**Behavior**: Desktop app with system tray icon. Manual start/stop. Saves stereo WAV to `./audio/`. No auto-processing — user runs `run.ps1` when ready.

#### Components

1. **Core recording engine** (`record.py`)
   - WASAPI loopback: captures all system audio (Teams, Zoom, browser, any app)
   - Microphone: captures user's voice
   - Output: stereo WAV (L=mic, R=loopback), 48kHz 16-bit
   - CLI mode: `python record.py --start` (Ctrl+C to stop)

2. **System tray icon** (`recorder_tray.py`)
   - Always-present tray icon (pystray)
   - Right-click menu: Start / Stop / Quit
   - Visual state: idle (gray) vs recording (red dot)
   - Windows notification on start/stop

3. **Floating recording widget** (`recorder_widget.py`)
   - Small always-on-top tkinter window
   - Shows: red REC indicator + elapsed time (MM:SS) + Stop button
   - Appears on record start, hides on stop

4. **App entry point** (`recorder_app.py`)
   - Launches tray + widget + recorder, coordinates events

#### Pipeline integration

- `audio_enhance.py` handles stereo input (downmix to mono or split channels)
- `transcribe.py` adds `--channel` flag: `both|left|right|mix` (default: mix)
- File naming: `YYYY-MM-DD_HHMM_recording.wav`

#### Limitations

- WASAPI loopback captures ALL system audio (notifications, music, etc.) — not per-app
- Large files: 48kHz stereo 16-bit ≈ 11 MB/min, ~660 MB/hour

#### Dependencies

```
PyAudioWPatch>=0.2.12    # WASAPI loopback (only Python lib that supports it)
pystray>=0.19.0          # system tray icon
Pillow>=9.0.0            # icon generation for pystray
```

### Phase 2: Multi-File Batch Processing + Reports

**Goal**: Process entire folders of meetings and generate cross-meeting analytics.

#### Features

1. **Batch processing**
   - Process all audio files in a folder in sequence
   - Resume interrupted batches (skip already-processed files)
   - Parallel processing option for API backends (multiple files concurrently)

2. **Cross-meeting analytics report**
   - `batch_report.md` generated after processing a folder
   - Per-meeting: duration, speaker count, word count, correction warning rate
   - Aggregate: total hours processed, most active speakers, recurring topics
   - Timeline: meeting frequency over time

3. **Topic tracking**
   - Extract key topics/themes from each meeting summary
   - Track topic evolution across meetings (first mentioned, recurring, resolved)
   - Tag meetings by project/topic for filtering

4. **Search**
   - Full-text search across all corrected transcripts
   - Filter by speaker, date range, topic
   - Output: matching segments with context and meeting source

### Phase 3: Speaker Profiles & Voice Training

**Goal**: Automatically identify known speakers by voice, eliminating manual speaker labeling.

#### Features

1. **Voice enrollment**
   - Provide 30-60s audio samples per speaker with name label
   - Extract voice embeddings (speaker fingerprint) using pyannote or resemblyzer
   - Store in `speaker_profiles.json`

2. **Auto-identification**
   - Compare diarized segments against enrolled voice profiles
   - Assign known names instead of SPEAKER_00, SPEAKER_01
   - Confidence threshold: label as "Unknown" if below threshold
   - Works with WhisperX backend (requires diarization first)

3. **Incremental learning**
   - After manual corrections ("SPEAKER_00 is actually Marco"), update voice profile
   - Improve accuracy over time as more meetings are processed

4. **Profile management**
   - CLI commands: `--enroll-speaker "Name" --sample audio.wav`
   - List profiles: `--list-speakers`
   - Delete profile: `--remove-speaker "Name"`

#### Dependencies

```
resemblyzer>=0.1.3   # or pyannote.audio for embeddings
numpy>=1.24.0
```

## 4. Non-Goals (out of scope for now)

- Real-time / live transcription (live capture records, but transcription is batch)
- Per-app audio capture (WASAPI captures all system audio)
- Web UI / dashboard
- Mobile app
- Video processing (face recognition, slide extraction)
- Translation (transcripts stay in original language)
- Cloud deployment / multi-user SaaS

## 5. Technical Constraints

- Windows-first (PowerShell + conda), Linux/macOS support is secondary
- Python 3.10+ required
- ffmpeg required for audio processing
- Free-tier friendly: Groq API for transcription, Claude Haiku for LLM
- All processing local-first where possible (privacy-sensitive meeting content)
- Audio files not committed to git (large binaries)
