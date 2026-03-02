# SpyMeet

Meeting audio transcription pipeline for Windows: live capture, audio enhancement, speech-to-text, LLM correction, and structured summaries. Designed for real meetings with noisy audio, domain-specific jargon, and multi-language support.

## Features

- **Live audio capture** — WASAPI loopback + microphone recording (stereo: system audio + mic)
- **Audio enhancement** — EBU R128 normalization, spectral noise reduction, speech EQ, dynamic compression
- **3 transcription backends** — WhisperX (local CPU), OpenAI Whisper API, Groq API (free tier)
- **LLM correction** — Chunked transcript correction with filler removal, punctuation fix, retry logic
- **Meeting summaries** — Structured Markdown summaries via Claude
- **Domain glossary** — Custom terminology for accurate transcription of technical terms
- **Speaker diarization** — Via pyannote (WhisperX backend + HuggingFace token)
- **Channel selection** — Process mic, system audio, or both channels separately
- **Desktop app** — System tray icon + floating recording widget with timer and VU meters
- **Dictation mode** — Mic-only recording for voice prompts

## Quick Start

### Prerequisites

- Windows 10/11
- Python 3.13+ via [conda](https://docs.conda.io/) (environment: `social_env`)
- ffmpeg (`conda install -c conda-forge ffmpeg`)

### Setup

```powershell
# 1. Create and activate conda environment
conda create -n social_env python=3.13
conda activate social_env
conda install -c conda-forge ffmpeg

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env with your API keys

# 4. (Optional) Check GPU for WhisperX CPU backend
.\scripts\check_gpu.ps1
```

### Recording

```powershell
# Launch desktop recorder (tray icon + floating widget)
python recorder_app.py

# Auto-start meeting recording
python recorder_app.py --mode meeting

# Auto-start dictation (mic only)
python recorder_app.py --mode dictation
```

### Transcription Pipeline

```powershell
# Full pipeline: transcribe + LLM correction (Groq, recommended)
.\scripts\run.ps1 -Backend groq-api -Language it

# Transcribe only (skip LLM)
.\scripts\run.ps1 -Backend groq-api -Language it -SkipLLM

# LLM correction on existing transcripts
.\scripts\run.ps1 -LLMOnly -Input .\audio\transcripts

# With domain glossary
.\scripts\run.ps1 -Backend groq-api -Language it -Glossary .\glossary.txt
```

## Architecture

```
recorder_app.py  (live capture: WASAPI loopback + mic -> stereo WAV)
    |
    v
audio_enhance.py (normalize, denoise, EQ, compress -> _enhanced.wav)
    |
    v
transcribe.py    (WhisperX CPU / OpenAI API / Groq API -> .txt + .json)
    |
    v
llm_process.py   (Claude: correction + summary -> _corrected.txt + _summary.md)
```

## Project Structure

```
spymeet/
├── CLAUDE.md               # Claude Code instructions
├── README.md               # This file
├── .env.example            # API key template
├── requirements.txt        # Python dependencies
├── glossary.txt            # Domain terminology
├── recorder.spec           # PyInstaller build config
│
├── record.py               # Core recording engine
├── recorder_app.py         # Desktop entry point (tray + widget)
├── recorder_widget.py      # Floating tkinter widget
├── recorder_tray.py        # System tray icon
├── audio_player.py         # Audio playback
├── pipeline_runner.py      # Background subprocess executor
├── diagnostics_window.py   # Audio diagnostics window
├── audio_enhance.py        # Audio preprocessing
├── transcribe.py           # Speech-to-text (3 backends)
├── llm_process.py          # LLM correction + summary
│
├── docs/                   # Documentation
│   ├── PRD.md              # Product requirements + roadmap
│   ├── architecture.md     # System architecture
│   ├── competitive_analysis.md
│   ├── sprint_live_capture.md
│   └── README_WIN.md       # Setup guide (Italian)
│
├── scripts/                # PowerShell helper scripts
│   ├── run.ps1             # Pipeline launcher
│   ├── setup.ps1           # Automated setup
│   └── check_gpu.ps1       # GPU/CUDA detection
│
├── tests/                  # Test suite
└── audio/                  # Runtime data (gitignored)
    └── transcripts/
```

## API Keys

Copy `.env.example` to `.env` and fill in your keys:

| Key | Required for |
|-----|-------------|
| `groq_api` | Groq API transcription (free tier) |
| `anthropic_api` | LLM correction + summary |
| `openai_api` | OpenAI Whisper API (optional) |
| `hf_token` | Speaker diarization (optional) |

## Documentation

- [Product Requirements & Roadmap](docs/PRD.md)
- [System Architecture](docs/architecture.md)
- [Competitive Analysis](docs/competitive_analysis.md)
- [Setup Guide (Italian)](docs/README_WIN.md)

## License

Private project.
