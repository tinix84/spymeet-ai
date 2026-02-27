# SpyMeet -- Architecture

## System Overview

SpyMeet is a sequential pipeline with four stages. Each stage is a standalone Python module that can run independently or be chained via `run.ps1`.

```
                          +------------------+
                          |   run.ps1        |  PowerShell orchestrator
                          |  (env, keys,     |  loads .env, activates conda,
                          |   CLI routing)   |  routes to correct backend
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |              |              |
              +-----v----+  +-----v-----+  +----v------+
              | WhisperX  |  | OpenAI    |  | Groq      |
              | CPU local |  | Whisper   |  | Whisper   |
              | + pyannote|  | API       |  | API       |
              +-----+-----+  +-----+-----+  +-----+-----+
                    |              |              |
                    +--------------+--------------+
                                   |
                          +--------v---------+
                          |  transcribe.py   |  Unified output:
                          |  format_*()      |  .txt + .json
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  llm_process.py  |  Claude Haiku
                          |  correct()       |  chunk -> correct -> retry
                          |  summarize()     |  full transcript -> summary
                          +--------+---------+
                                   |
                      +------------+------------+
                      |            |            |
                +-----v---+  +----v-----+  +---v--------+
                | _corrected| | _summary |  | _metrics   |
                | .txt      | | .md      |  | .md        |
                +-----------+ +----------+  +------------+
```

## Component Details

### 1. run.ps1 (Orchestrator)

**Responsibility**: Environment setup, API key loading, CLI argument routing.

- Activates conda `social_env` environment
- Validates required API keys per backend
- Sets `KMP_DUPLICATE_LIB_OK=TRUE` for torch compatibility
- Routes to `transcribe.py` or `llm_process.py` based on flags (`-LLMOnly`, `-SkipLLM`)

**Key design choice**: PowerShell (not bash) because the primary platform is Windows. The `_run_transcribe.ps1` helper exists as a machine-specific workaround for PATH issues with conda's ffmpeg on Windows.

### 2. transcribe.py (Speech-to-Text)

**Responsibility**: Convert audio files to timestamped, speaker-labeled text segments.

**Backends**:

| Backend | Diarization | Speed | Cost | Max file |
|---------|-------------|-------|------|----------|
| `cpu` (WhisperX) | Yes (pyannote) | Slow (1-3x realtime) | Free | Unlimited |
| `openai-api` | No | Fast (~10s/file) | $0.006/min | 25MB (auto-split) |
| `groq-api` | No | Fast (~5s/file) | Free (7200 min/day) | 25MB (auto-split) |

**Data flow**:
1. Discover audio files (SUPPORTED_EXTENSIONS)
2. For large files (>25MB), split via ffmpeg into 10-min MP3 chunks
3. Transcribe each chunk, accumulate segments with time offsets
4. Output `.txt` (human-readable) and `.json` (full metadata)
5. Optionally trigger `llm_process.py`

**Output format** (.txt):
```
[SPEAKER_00]
  [00:00] First segment text...
  [00:15] Second segment text...
[SPEAKER_01]
  [00:22] Another speaker...
```

**Key gotcha**: Groq SDK returns segments as `dict`, OpenAI SDK returns objects with attributes. Code uses `isinstance(seg, dict)` to handle both.

### 3. llm_process.py (LLM Post-Processing)

**Responsibility**: Clean transcript and generate structured summary.

**Stage 1 -- Correction**:
```
Raw segments -> group into ~5min chunks -> send to Claude -> parse response
                                                |
                                    if WARNINGs: retry warned segments (up to 3x)
                                                |
                                    accumulate accepted segments
```

- System prompt instructs: fix punctuation, remove fillers, keep original language, preserve speaker/timestamp labels
- Warning mechanism flags unintelligible segments
- Retry logic isolates only warned segments and re-sends them
- Non-warned segments from each pass are accumulated in `accepted` list (not overwritten)

**Stage 2 -- Summary**:
- Sends full corrected transcript to Claude
- System prompt specifies exact Markdown structure: Participants, Key Decisions, Action Items, Open Questions
- Output language matches transcript language

**Model**: `claude-haiku-4-5-20251001` (fast, cheap, sufficient for correction/summary)

### 4. Glossary (glossary.txt)

Simple text file, one term per line. Optional `= description` format. Injected into both correction and summary prompts to improve domain-specific accuracy.

## Data Model

### Segment (raw transcription)
```python
{
    "start": float,     # seconds from audio start
    "end": float,       # seconds
    "text": str,        # raw transcribed text
    "speaker": str      # "SPEAKER_00", "SPEAKER_01", etc.
}
```

### CorrectedSegment (after LLM)
```python
@dataclass
class CorrectedSegment:
    speaker: str        # preserved from input
    timestamp: str      # "mm:ss" format
    original: str       # raw text before correction
    corrected: str      # cleaned text
    warnings: list[str] # empty if segment is clean
    retries_used: int   # 0 = corrected on first pass
```

## Planned Architecture Changes

### Phase 1: Audio Enhancement (audio_enhance.py)

New module inserted between audio input and transcription:

```
Audio -> audio_enhance.py -> [name]_enhanced.wav -> transcribe.py -> ...
```

Processing chain (sequential, always-on):
1. Load audio (soundfile/librosa)
2. Loudness normalization (EBU R128, target -16 LUFS for speech)
3. Noise reduction (spectral gating, noisereduce library)
4. Speech EQ (highpass 80Hz + presence boost 2-4kHz, scipy.signal)
5. Dynamic compression (3:1 ratio, adaptive threshold)
6. Save enhanced file (16-bit PCM, 16kHz mono WAV)
7. Log before/after metrics (LUFS, peak dB, estimated SNR)

Integration: `transcribe.py` calls `audio_enhance.py` before processing. Both original and enhanced files are preserved.

### Phase 2: Batch Processing

```
run.ps1 --batch ./meetings/
    -> transcribe.py processes each file
    -> llm_process.py processes each transcript
    -> batch_report.py generates cross-meeting analytics
       -> batch_report.md (per-meeting stats + aggregates + topic tracking)
```

New module: `batch_report.py` reads all `_summary.md` files in a folder and generates aggregate report.

### Phase 3: Speaker Profiles

```
speaker_profiles.json    <- voice embeddings per enrolled speaker
    |
    v
transcribe.py (WhisperX backend)
    -> diarize with pyannote
    -> compare embeddings against profiles
    -> assign known names to segments
```

New module: `speaker_profiles.py` handles enrollment, matching, and incremental learning. Requires WhisperX backend (API backends don't provide per-segment audio for embedding extraction).

## File Organization

```
spymeet/
  CLAUDE.md             # Claude Code guidance
  PRD.md                # Product requirements + roadmap
  architecture.md       # This file
  README_WIN.md         # User-facing setup guide (Italian)
  .gitignore
  .env                  # API keys (gitignored)
  glossary.txt          # Domain terminology
  gpu_config.json       # Machine-specific GPU config (gitignored)
  check_gpu.ps1         # Step 1: detect GPU
  setup.ps1             # Step 2: install dependencies
  run.ps1               # Step 3: run pipeline
  _run_transcribe.ps1   # Machine-specific helper (gitignored)
  transcribe.py         # Speech-to-text engine
  llm_process.py        # LLM correction + summary
  audio_enhance.py      # (planned) Audio preprocessing
  batch_report.py       # (planned) Cross-meeting analytics
  speaker_profiles.py   # (planned) Voice enrollment + matching
  audio/                # Input audio files (gitignored)
    transcripts/        # Output transcripts (gitignored)
```
