# Competitive Analysis — Open-Source Meeting Transcription Tools

*Last updated: 2026-03-01*

SpyMeet is a modular Python pipeline: record (WASAPI loopback+mic) -> enhance (normalize, denoise, EQ, compress) -> transcribe (WhisperX/Groq/OpenAI) -> LLM correct+summarize (Claude). This document evaluates comparable open-source projects and assesses fork/merge potential.

---

## Table of Contents

1. [Landscape Overview](#1-landscape-overview)
2. [SpyMeet Unique Advantages](#2-spymeet-unique-advantages)
3. [Deep-Dive: Meetily](#3-deep-dive-meetily)
4. [Deep-Dive: Ecoute](#4-deep-dive-ecoute)
5. [Deep-Dive: WhisperLive](#5-deep-dive-whisperlive)
6. [Deep-Dive: Scriberr](#6-deep-dive-scriberr)
7. [Deep-Dive: noScribe](#7-deep-dive-noscribe)
8. [Deep-Dive: Hearsay](#8-deep-dive-hearsay)
9. [Fork Assessment Summary](#9-fork-assessment-summary)
10. [Recommendations](#10-recommendations)

---

## 1. Landscape Overview

12 comparable projects were evaluated. The table below covers all, with deep-dives on the 6 most relevant.

| Project | Stars | Live Capture | Audio Enhance | STT | LLM Post-Process | Desktop GUI | License | Last Updated |
|---------|------:|:---:|:---:|:---:|:---:|:---:|---------|------------|
| **SpyMeet** | -- | WASAPI+mic | Full chain | WhisperX/OpenAI/Groq | Correction+Summary | Tray+Widget | -- | Active |
| **[Meetily](https://github.com/Zackriya-Solutions/meeting-minutes)** | 10.1k | sys+mic | RNNoise+R128 | whisper.cpp/Parakeet | Summary only | Tauri native | MIT | Feb 2026 |
| **[Ecoute](https://github.com/SevaSk/ecoute)** | 6k | WASAPI+mic | -- | faster-whisper tiny.en | Removed (was GPT) | CustomTkinter | MIT | Sep 2025 |
| **[WhisperLive](https://github.com/collabora/WhisperLive)** | 3.8k | mic/streams | -- | FW/TensorRT/OpenVINO | -- | Browser ext | MIT | Feb 2026 |
| **[Scriberr](https://github.com/rishikanthc/Scriberr)** | 2.2k | browser only | minimal | WhisperX/Parakeet/Canary | Chat+Summary | Web PWA | MIT | Feb 2026 |
| **[noScribe](https://github.com/kaixxx/noScribe)** | 1.8k | -- | -- | faster-whisper | -- | CustomTkinter+Editor | GPL-3.0 | Feb 2026 |
| **[Hearsay](https://github.com/parkscloud/Hearsay)** | 0 | WASAPI+mic | -- | faster-whisper | -- | CustomTkinter | MIT | Feb 2026 |
| [Nojoin](https://github.com/Valtora/Nojoin) | 31 | sys+mic | -- | Whisper GPU | Summary+Chat | Web+Rust | AGPL-3.0 | Nov 2025 |
| [Pensieve](https://github.com/lukasbach/pensieve) | 106 | app audio | -- | bundled Whisper | Summary (Ollama) | Electron tray | -- | Sep 2025 |
| [Transcribe-Critic](https://github.com/ringger/transcribe-critic) | 12 | -- | -- | multi-model ensemble | LLM adjudication | -- | MIT | Feb 2026 |
| [Meeting Transcriber](https://github.com/jfcostello/meeting-transcriber) | 23 | -- | -- | Whisper/Faster | Summary (6+ LLMs) | -- | MIT | Aug 2024 |
| [Whisply](https://github.com/tsmdt/whisply) | 102 | -- | -- | multi-engine | YAML corrections | Gradio | MIT | Dec 2025 |
| [Meminto](https://github.com/FlorianSchepers/Meminto) | 29 | -- | -- | Whisper | Meeting minutes | -- | MIT | Feb 2025 |

---

## 2. SpyMeet Unique Advantages

No single competitor replicates SpyMeet's full pipeline. Key differentiators:

### 2.1 Audio Enhancement Pipeline (unique)
No other project has a dedicated preprocessing chain:
- EBU R128 loudness normalization (-16 LUFS)
- Spectral gating noise reduction (noisereduce)
- Speech EQ (HP 80Hz + presence peak +2.5dB @ 3kHz)
- Dynamic compression (3:1, adaptive threshold)
- Output: 16-bit PCM WAV, 16kHz mono

Meetily has real-time RNNoise + EBU R128 (-23 LUFS) in Rust, but no EQ or compression.

### 2.2 LLM Transcript Correction (unique)
Two-stage Claude pipeline with features no competitor has:
- Chunked ~5min segment correction with retry on warned segments (up to 3x)
- Filler word removal: Italian (allora, quindi, cioe), German (naja, ahm, sozusagen), universal (uhm, uh)
- Domain-specific glossary injection (`glossary.txt`)
- Structured Markdown meeting summary generation
- Quality metrics output (`_metrics.md`)

Competitors either skip LLM entirely or only do summarization (never correction).

### 2.3 Offline Postprocessing & Cache
All intermediate files are preserved on disk, enabling iterative refinement:
```
./audio/recording.wav              (original capture, 48kHz stereo)
./audio/recording_enhanced.wav     (enhanced, 16kHz mono)
./audio/transcripts/recording.txt  (raw transcript)
./audio/transcripts/recording.json (with timestamps/segments)
./audio/transcripts/recording_corrected.txt   (LLM-corrected)
./audio/transcripts/recording_summary.md      (meeting summary)
./audio/transcripts/recording_metrics.md      (quality metrics)
```
- Re-run enhancement with different parameters without re-recording
- Re-run transcription with different backends/languages without re-enhancing
- Re-run LLM correction with different glossaries until satisfied
- Compare outputs across runs

No competitor preserves the full intermediate chain. Hearsay and Ecoute discard audio entirely. Meetily and Scriberr save recordings but don't expose intermediate processing steps.

### 2.4 Stereo Channel Selection
WASAPI loopback recordings: L=mic, R=system audio. Post-hoc channel selection:
- `--channel mix` (default): downmix to mono
- `--channel left`: mic only
- `--channel right`: system audio only
- `--channel both`: process L and R separately with `_mic`/`_system` suffixes

No competitor offers per-channel processing on stereo recordings.

### 2.5 Multi-Backend Transcription
Three selectable backends via `--backend`:
- `cpu`: WhisperX (faster-whisper/CTranslate2) — fully local, speaker diarization
- `groq-api`: Groq cloud — fast, free tier, zero CPU
- `openai-api`: OpenAI Whisper API — reliable cloud fallback

Competitors typically offer only one engine (local Whisper or API, not both).

---

## 3. Deep-Dive: Meetily

**Repo**: [github.com/Zackriya-Solutions/meeting-minutes](https://github.com/Zackriya-Solutions/meeting-minutes) | 10.1k stars | MIT | Tauri (Rust + Next.js)

### Architecture
- Tauri v2 desktop app: Rust backend, Next.js/TypeScript frontend
- Audio: `cpal` library (wraps WASAPI on Windows), dual mic+system capture
- STT: whisper-rs (whisper.cpp bindings) + NVIDIA Parakeet TDT 0.6B (ONNX)
- LLM: Ollama, Claude, Groq, OpenRouter, OpenAI-compatible
- Storage: SQLite for meetings/transcripts/summaries
- Real-time audio pipeline in Rust: HP 80Hz, RNNoise denoising, EBU R128 (-23 LUFS), ring buffer mixer

### CPU Assessment
- **Parakeet (fast engine): English only** — unusable for IT/DE meetings
- **Whisper on CPU: 4-10x slower than optimal** due to missing AVX2/FMA optimizations (issue #306)
- GPU detection priority: CUDA > Metal > Vulkan > OpenBLAS > CPU fallback
- Pre-built Windows exe ships with CPU support

### Non-English
- Parakeet: English only (dealbreaker for IT/DE)
- Whisper: 99 languages, but runs on the slow/broken CPU code path
- No language-specific post-processing

### LLM
- Summarization only, NOT correction
- No filler removal, no glossary, no retry logic
- Ollama models <32B struggle with function calling (issue #25)

### Windows Issues
- Crashes on exit during recording (issue #239)
- CPU compatibility crashes without AVX2 (issue #228, fixed)
- CUDA not utilized even with RTX 4070 Mobile (issue #333)
- Long meetings caused UI freeze and data loss (issue #25, improved in v0.2.0)

### Fork Assessment
| Criterion | Score | Notes |
|-----------|-------|-------|
| IT/DE on CPU laptop | Poor | Parakeet English-only; Whisper CPU broken |
| Feature completeness | Good | Recording, real-time STT, summaries, GUI |
| Codebase extensibility | Poor | Rust/TypeScript stack, completely different from Python |
| **Verdict** | **Don't fork** | Wrong language stack, CPU issues for non-English |

### Worth Borrowing
- Silero VAD before transcription (skip silence, save API calls)
- RNNoise neural denoising concept
- Ring buffer mixer for real-time audio sync

---

## 4. Deep-Dive: Ecoute

**Repo**: [github.com/SevaSk/ecoute](https://github.com/SevaSk/ecoute) | 6k stars | MIT | Python + CustomTkinter

### Architecture
- Python, CustomTkinter GUI, PyAudioWPatch (same WASAPI lib as SpyMeet)
- faster-whisper with bundled `tiny.en` model (75MB)
- Dual-stream: mic + WASAPI loopback → separate "You" / "Speaker" transcripts
- Bundled fork of SpeechRecognition library

### CPU Assessment
- CPU-only: uses int8 quantization automatically
- Hardcoded to `tiny.en` (39M params) — English only, no model selection
- ~1x real-time on modern CPUs with tiny.en; lags on U-series mobile CPUs (issue #15, #92)

### Non-English
- Local mode: **English only** (tiny.en hardcoded)
- API mode (`--api`): OpenAI Whisper API, but multilingual support is incomplete (issue #11)
- No language parameter exists

### LLM
- **GPT responder was REMOVED** in March 2025 commit
- Previously generated conversational response suggestions (not correction)
- Currently has zero LLM integration

### Maintenance
- Effectively abandoned: 21-month gap between development bursts
- Last code commit: March 2025; last any commit: Sep 2025 (sponsorship only)
- 116 open issues, most unanswered

### Audio
- System + mic simultaneous via PyAudioWPatch (same as SpyMeet)
- **Does NOT save recordings** — audio discarded after transcription
- No enhancement, no normalization

### Fork Assessment
| Criterion | Score | Notes |
|-----------|-------|-------|
| IT/DE on CPU laptop | Poor | English-only local, broken multilingual API mode |
| Feature completeness | Poor | No recording, no LLM, no enhancement, no diarization |
| Codebase extensibility | Medium | Small codebase (~500 LOC), MIT, but abandoned |
| **Verdict** | **Don't fork** | Abandoned, English-only, missing core features |

### Worth Borrowing
- Real-time dual-stream "You" / "Speaker" display concept
- PyAudioWPatch WASAPI pattern (already used in SpyMeet)
- Notable fork: [vivekuppal/transcribe](https://github.com/vivekuppal/transcribe) (249 stars) adds multi-LLM, multilingual

---

## 5. Deep-Dive: WhisperLive

**Repo**: [github.com/collabora/WhisperLive](https://github.com/collabora/WhisperLive) | 3.8k stars | MIT | Python (client/server)

### Architecture
- Client/server over WebSockets
- Server holds Whisper model; clients stream raw audio, receive JSON transcripts
- Backends: faster-whisper (CTranslate2), TensorRT, OpenVINO
- Silero VAD v5.0 preprocesses audio (skips silence)
- Browser extensions (Chrome/Firefox) capture tab audio
- REST API with OpenAI-compatible interface
- M2M100 translation engine (100 languages)

### CPU Assessment
- CPU supported: faster-whisper int8, or OpenVINO (Intel-optimized)
- `small`/`medium` models near real-time on 8+ core CPUs
- `large-v3-turbo` int8 is the sweet spot for quality vs speed on CPU
- Windows real-time performance reported as poor (issue #172)

### Non-English
- Full multilingual support with non-`.en` models
- Auto language detection with confidence threshold
- Italian/German fully supported
- Real-time translation via M2M100

### Audio
- **No WASAPI loopback** — mic only from PyAudio
- System audio only via browser extensions (Chrome/Firefox tab capture)
- Desktop app audio requires VoiceMeeter or similar virtual audio routing (issue #258)
- Silero VAD filters silence before Whisper
- No audio enhancement

### LLM
- **None**. Pure transcription + optional translation server

### Speaker Diarization
- **Not supported** (issue #323, open since Jan 2025)

### Fork Assessment
| Criterion | Score | Notes |
|-----------|-------|-------|
| IT/DE on CPU laptop | Medium | Good multilingual, decent CPU perf, but no system audio |
| Feature completeness | Medium | Great STT streaming, but no enhancement/correction/diarization |
| Codebase extensibility | Good | Modular backends, MIT, well-maintained by Collabora |
| **Verdict** | **Don't fork, borrow patterns** | No system audio capture; server model doesn't fit desktop app |

### Worth Borrowing
- Silero VAD preprocessing (skip silence before transcription)
- Streaming faster-whisper pattern for real-time transcription
- OpenAI-compatible REST API concept
- Browser extension for tab audio capture

---

## 6. Deep-Dive: Scriberr

**Repo**: [github.com/rishikanthc/Scriberr](https://github.com/rishikanthc/Scriberr) | 2.2k stars | MIT | Go + React + Python

### Architecture
- Go 1.24 backend (Gin HTTP, GORM, SQLite) + React/TypeScript frontend (Vite, Radix UI)
- Python ML layer (WhisperX, pyannote) invoked as subprocesses from Go
- Docker-first deployment with multi-stage builds
- Folder watcher (fsnotify) for auto-processing dropped files
- SSE for real-time status, JWT auth, Swagger REST API

### CPU Assessment
- CPU works but slow: ~2 hours to transcribe+diarize 1 hour of audio on 16 cores (issue #401)
- WhisperX defaults to 4 threads — must tune `OMP_NUM_THREADS`
- Dedicated GPU Docker images per NVIDIA generation (Pascal through Blackwell)

### Non-English
- WhisperX: 90+ languages including Italian/German (same as SpyMeet)
- No language-specific post-processing

### LLM
- Ollama (local) or OpenAI-compatible API
- **Chat with transcript** — interactive Q&A over meeting content
- Summary generation
- **No transcript correction** — chat/summary only

### Audio
- Recording via browser Screen Capture API (Chromium only) — NOT WASAPI
- Minimal enhancement (ffmpeg loudnorm on MP3s only)
- Good format support (mp3, wav, flac, m4a, etc.)
- wavesurfer.js playback with word-level seek-from-text

### Speaker Diarization
- Two backends: pyannote (needs HF_TOKEN) + NVIDIA Sortformer (no token needed)
- Better options than SpyMeet (Sortformer is tokenless)

### Windows
- Docker is realistic path; native has known bugs (issue #400)
- No WASAPI integration

### Fork Assessment
| Criterion | Score | Notes |
|-----------|-------|-------|
| IT/DE on CPU laptop | Medium | Same WhisperX, but Docker overhead, no native audio |
| Feature completeness | Good | Diarization, chat, web UI, folder watcher |
| Codebase extensibility | Poor for SpyMeet | Go+React+Python polyglot stack, high maintenance cost |
| **Verdict** | **Don't fork** | Wrong architecture (server/Docker vs desktop), no WASAPI |

### Worth Borrowing
- Chat-with-transcript concept
- wavesurfer.js word-level playback sync
- Folder watcher / dropzone pattern
- NVIDIA Sortformer diarization (no HF_TOKEN needed)

---

## 7. Deep-Dive: noScribe

**Repo**: [github.com/kaixxx/noScribe](https://github.com/kaixxx/noScribe) | 1.8k stars | GPL-3.0 | Python + CustomTkinter

### Architecture
- Python, CustomTkinter GUI, faster-whisper (CTranslate2)
- Single 3,674-line monolith (`noScribe.py`) mixing GUI + business logic
- Whisper + pyannote run in spawned subprocesses (process isolation)
- HTML-based transcript format with timestamp anchors
- Separate PyQt6 editor app (noScribeEdit) with synced audio playback

### CPU Assessment
- CPU always available; GPU optional (NVIDIA CUDA 6GB+, Apple MPS)
- **"A one-hour interview can take up to three hours" on CPU** (author's statement)
- v0.6 claimed 3x speed improvement from faster-whisper optimizations
- Two models: "precise" (Large v3 Turbo full) and "fast" (Large v3 Turbo int8, ~30% faster)

### Non-English
- ~60 languages, Italian and German explicitly "best supported"
- Language-specific hotword prompts (Italian: "Ehm, sai, non e proprio, diciamo, semplice.")
- "Multilingual" mode for code-switching within single recordings
- Full UI translations (EN, DE, ES, FR, IT, JA, PT, RU, ZH)

### LLM
- **None**. Zero LLM integration

### Audio
- **Cannot record audio** — file input only
- No enhancement, no channel selection
- Converts input to 16kHz mono WAV via bundled FFmpeg

### Speaker Diarization
- pyannote v4.0 with **bundled local models** (no HF_TOKEN needed!)
- VBx clustering, configurable speaker count
- Speaker overlap detection (experimental)

### Transcript Editor (noScribeEdit)
- PyQt6 app with synced audio playback (FFplay)
- Click text to seek audio; play from cursor position
- Adjustable speed (60%-200%)
- Rich text editing, search & replace
- Export: HTML, TXT, WebVTT

### Unique Features
- **Overlapping speech detection**: marks `//double slashes//` when speakers talk simultaneously
- **Pause marking**: configurable thresholds (1s+, 2s+, 3s+), `(..)` notation, uses Silero VAD
- **Batch queue**: multiple files queued with per-job status/progress
- **Disfluency toggle**: include/exclude speech disfluencies via language-specific prompts
- **CLI mode**: `--no-gui` for headless/scripted operation

### Fork Assessment
| Criterion | Score | Notes |
|-----------|-------|-------|
| IT/DE on CPU laptop | Good | Excellent IT/DE support, CPU works (slow but reliable) |
| Feature completeness | Medium | Great STT+diarization+editor, but no recording/enhancement/LLM |
| Codebase extensibility | Poor | GPL-3.0 (copyleft!), 3.7k-line monolith, no tests |
| **Verdict** | **Don't fork** | GPL-3.0 forces open-sourcing derivatives; monolith codebase |

### Worth Borrowing (study algorithms, reimplement independently)
- Bundled pyannote models (no HF_TOKEN dependency)
- Pause marking via Silero VAD
- Overlapping speech detection logic
- Batch queue architecture
- Language-specific hotword prompts for Whisper

---

## 8. Deep-Dive: Hearsay

**Repo**: [github.com/parkscloud/Hearsay](https://github.com/parkscloud/Hearsay) | 0 stars | MIT | Python

**Architecturally the closest project to SpyMeet** — both Python, both PyAudioWPatch for WASAPI loopback.

### Architecture
- Python 3.11+, CustomTkinter GUI, pystray tray icon
- PyAudioWPatch for WASAPI loopback (identical to SpyMeet)
- sounddevice for mic-only mode
- faster-whisper (CTranslate2) for real-time local transcription
- Proper package layout: `src/hearsay/` with `audio/`, `transcription/`, `output/`, `ui/`, `utils/`
- 5-screen first-run setup wizard

### CPU Assessment
- faster-whisper with **int8 quantization by default** on CPU
- ~1x real-time with `small.en` + INT8 (README claim)
- GPU: ~8x real-time with `turbo` on 6GB+ VRAM
- Model selection: tiny through large-v3 plus turbo

### Non-English
- Multilingual models supported (tiny, base, small, medium, large-v3, turbo)
- Language config via ISO 639-1 code; auto-detect if blank
- Wizard defaults to `.en` models — users must manually switch for IT/DE

### LLM
- **None**. Zero post-processing. Raw Whisper output saved as Markdown

### Audio
- Three modes: system (loopback only), microphone (mic only), both (mixed)
- **Does NOT save audio files** — in-memory only, discarded after transcription
- No enhancement, no normalization
- 30s chunks with 1s overlap, fed directly to faster-whisper
- RMS normalization to -20 dBFS before mixing streams

### Real-Time Transcription
- `AudioRecorder` thread → audio queue → `TranscriptionPipeline` thread → transcript queue → GUI poll (250ms)
- Live transcript window with auto-scroll
- Overlap deduplication: keeps last 15 words per chunk, strips matched prefixes
- ~30-60 second display delay depending on hardware

### Windows
- Inno Setup installer (`HearsaySetup.exe`)
- Enterprise deployment: silent install via `/VERYSILENT`, SCCM/Intune compatible
- First-run setup wizard with GPU detection
- Config: `%APPDATA%\Hearsay\config.json`

### Maintenance
- Very new: created Feb 13, 2026, v1.0.2 released Feb 17, 2026
- 20 total commits, all AI-assisted (Claude co-authored)
- 0 stars, 0 issues — no community yet

### Fork Assessment
| Criterion | Score | Notes |
|-----------|-------|-------|
| IT/DE on CPU laptop | Medium | Multilingual models work; real-time INT8 viable |
| Feature completeness | Low | No recording save, no enhancement, no LLM, no diarization |
| Codebase extensibility | Medium | Clean Python, MIT, same tech stack — but fundamentally different philosophy (stream-and-discard vs record-and-process) |
| **Verdict** | **Don't fork, cherry-pick** | Same WASAPI tech, but opposite philosophy (discards audio vs preserves all intermediate files) |

### Worth Cherry-Picking
- **High value**: faster-whisper as new SpyMeet backend (alongside WhisperX/Groq/OpenAI)
- **High value**: Chunk overlap deduplication algorithm for future real-time mode
- **Medium value**: Inno Setup installer template + silent install parameters
- **Medium value**: CustomTkinter for UI polish (drop-in tkinter replacement)
- **Medium value**: Setup wizard pattern for first-run onboarding
- **Low urgency**: GPU detection logic

---

## 9. Fork Assessment Summary

### Decision Matrix

| Project | IT/DE CPU | Feature Coverage | Extensibility | License | **Fork?** |
|---------|:---------:|:----------------:|:-------------:|:-------:|:---------:|
| Meetily | Poor | Good | Poor (Rust/TS) | MIT | **No** |
| Ecoute | Poor | Poor | Medium | MIT | **No** |
| WhisperLive | Medium | Medium | Good | MIT | **No** |
| Scriberr | Medium | Good | Poor (Go+React) | MIT | **No** |
| noScribe | Good | Medium | Poor (GPL, monolith) | GPL-3.0 | **No** |
| Hearsay | Medium | Low | Medium | MIT | **No** |

### Why None Are Worth Forking

1. **Wrong architecture**: Meetily (Rust), Scriberr (Go+React), WhisperLive (server model) — all require maintaining a completely different tech stack
2. **Missing core features**: None have SpyMeet's audio enhancement + LLM correction + glossary combination
3. **Wrong philosophy**: Ecoute and Hearsay discard audio (stream-and-forget); SpyMeet preserves everything for iterative offline refinement
4. **License trap**: noScribe's GPL-3.0 forces copyleft on all derivatives
5. **Maintenance risk**: Ecoute is abandoned; Hearsay has zero community; noScribe is a solo academic project
6. **CPU/non-English gap**: Meetily's Whisper CPU path is broken; Ecoute is English-only; Hearsay defaults to English models

---

## 10. Recommendations

### Keep building SpyMeet. Cherry-pick ideas, don't fork.

SpyMeet's unique combination (enhancement + correction + glossary + offline cache + channel selection) is not replicated by any competitor. The modular Python pipeline is the right architecture for the use case (CPU laptop, IT/DE meetings, iterative offline refinement).

### Priority cherry-picks from competitors:

| Feature | Source | Value | Effort |
|---------|--------|:-----:|:------:|
| Silero VAD before transcription | WhisperLive, noScribe | High | Low |
| faster-whisper as local backend | Hearsay, noScribe | High | Medium |
| Bundled pyannote (no HF_TOKEN) | noScribe | High | Medium |
| Real-time transcript preview | Hearsay, Ecoute | High | High |
| Chat with transcript | Scriberr | Medium | High |
| Installer + setup wizard | Hearsay | Medium | Medium |
| CustomTkinter UI refresh | Hearsay, noScribe, Ecoute | Low | Low |
| Pause marking via VAD | noScribe | Low | Low |
| Overlap speech detection | noScribe | Low | Medium |
| Browser extension for tab audio | WhisperLive | Low | High |
