"""
Audio Transcriber — three backends
  --backend cpu        : WhisperX local, CPU mode (no GPU needed)
  --backend openai-api : OpenAI Whisper API (cloud, fast, $0.006/min)
  --backend groq-api   : Groq Whisper API (free tier, 7200 min/day)

Usage:
    python transcribe.py --input ./audio --language it --backend cpu
    python transcribe.py --input ./audio --language it --backend openai-api
    python transcribe.py --input ./audio --language it --backend groq-api
    python transcribe.py --input ./audio  # defaults to cpu, auto-detect language
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

# LLM post-processing (optional)
try:
    from llm_process import process_transcript_file
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".mkv", ".webm"}

# OpenAI API max file size: 25MB — larger files must be chunked
OPENAI_MAX_BYTES = 25 * 1024 * 1024

# ─── Shared output helpers ────────────────────────────────────────────────────

def format_transcript_txt(segments: list, output_path: Path):
    """Save human-readable transcript with speaker labels and timestamps."""
    with open(output_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER_?")
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)
            if not text:
                continue
            if speaker != current_speaker:
                f.write(f"\n[{speaker}]\n")
                current_speaker = speaker
            mm, ss = divmod(int(start), 60)
            f.write(f"  [{mm:02d}:{ss:02d}] {text}\n")
    print(f"  Saved: {output_path}")


def format_transcript_json(data, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path}")


def run_llm(audio_stem: str, output_dir: Path, glossary_path, skip_llm: bool):
    """Run LLM post-processing if conditions are met."""
    if skip_llm:
        return
    if not LLM_AVAILABLE:
        print("  [LLM] Skipped — llm_process.py not found")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [LLM] Skipped — ANTHROPIC_API_KEY not set")
        print(f"  Run manually: python llm_process.py --input {output_dir / (audio_stem + '.txt')}")
        return
    txt_path = output_dir / f"{audio_stem}.txt"
    try:
        process_transcript_file(txt_path, glossary_path, output_dir)
    except Exception as e:
        print(f"  [LLM WARNING] {e}")


# ─── BACKEND 1: WhisperX CPU ─────────────────────────────────────────────────

def run_whisperx_cpu(
    audio_files: list,
    output_dir: Path,
    language: str,
    model_name: str,
    hf_token: str,
    glossary_path,
    skip_llm: bool
):
    try:
        import whisperx
        import torch
    except ImportError:
        print("[ERROR] whisperx not installed.")
        print("  pip install whisperx torch torchaudio")
        return

    device = "cpu"
    compute_type = "int8"
    print(f"[INFO] Backend: WhisperX CPU | Model: {model_name}")
    print(f"[INFO] Note: CPU is slow. A 1h recording may take 1-3h.")

    if not hf_token:
        print("[WARNING] HF_TOKEN not set — speaker diarization will be skipped.")

    print(f"[INFO] Loading model '{model_name}'...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    diarize_model = None
    if hf_token:
        print("[INFO] Loading diarization model...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    align_models = {}

    for file_idx, audio_path in enumerate(sorted(audio_files), 1):
        file_label = f"[{file_idx}/{len(audio_files)}]" if len(audio_files) > 1 else ""
        print(f"\n[PROCESSING] {file_label} {audio_path.name}")
        t0 = time.time()

        print("  [1/4] Loading audio...", end="", flush=True)
        audio = whisperx.load_audio(str(audio_path))
        duration_secs = len(audio) / 16000  # whisperx loads at 16kHz
        mm, ss = divmod(int(duration_secs), 60)
        print(f" done ({mm}m{ss:02d}s of audio)")

        kwargs = {"batch_size": 8}  # lower batch for CPU
        if language:
            kwargs["language"] = language
        print("  [2/4] Transcribing...", end="", flush=True)
        result = model.transcribe(audio, **kwargs)
        detected_lang = result.get("language", language or "unknown")
        elapsed = time.time() - t0
        print(f" done ({elapsed:.0f}s elapsed, lang={detected_lang})")

        # Align
        print("  [3/4] Aligning timestamps...", end="", flush=True)
        if detected_lang not in align_models:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=device
            )
            align_models[detected_lang] = (align_model, metadata)
        align_model, metadata = align_models[detected_lang]
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False
        )
        print(f" done ({len(result['segments'])} segments)")

        # Diarize
        if diarize_model:
            print("  [4/4] Diarizing speakers...", end="", flush=True)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            speakers = {seg.get("speaker", "?") for seg in result["segments"]}
            print(f" done ({len(speakers)} speaker(s))")
        else:
            print("  [4/4] Diarization skipped (no HF_TOKEN)")
            for seg in result["segments"]:
                seg.setdefault("speaker", "SPEAKER_00")

        total_elapsed = time.time() - t0
        print(f"  Completed in {total_elapsed:.0f}s")

        stem = audio_path.stem
        format_transcript_txt(result["segments"], output_dir / f"{stem}.txt")
        format_transcript_json(result, output_dir / f"{stem}.json")
        run_llm(stem, output_dir, glossary_path, skip_llm)


# ─── BACKEND 2: OpenAI Whisper API ───────────────────────────────────────────

def split_audio_ffmpeg(audio_path: Path, chunk_dir: Path, chunk_secs: int = 600) -> list:
    """Split audio into chunks under 25MB using ffmpeg. Returns list of chunk paths."""
    import subprocess
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunk_dir / f"{audio_path.stem}_%03d.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_secs),
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz (Whisper native)
        "-b:a", "32k",        # low bitrate to stay under 25MB
        str(pattern)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {result.stderr}")
    chunks = sorted(chunk_dir.glob(f"{audio_path.stem}_*.mp3"))
    return chunks


def transcribe_openai_api(
    audio_path: Path,
    client,
    language: str,
    verbose: bool = True
) -> list:
    """
    Transcribe a single file via OpenAI Whisper API.
    Returns list of segment dicts compatible with format_transcript_txt.
    Handles files > 25MB by splitting with ffmpeg.
    """
    file_size = audio_path.stat().st_size
    if verbose:
        size_mb = file_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")

    if file_size > OPENAI_MAX_BYTES:
        print(f"  File > 25MB — splitting into chunks...")
        chunk_dir = audio_path.parent / f"_chunks_{audio_path.stem}"
        chunks = split_audio_ffmpeg(audio_path, chunk_dir)
        print(f"  Split into {len(chunks)} chunk(s)")
    else:
        chunks = [audio_path]

    all_segments = []
    time_offset = 0.0

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"  Transcribing chunk {i+1}/{len(chunks)}: {chunk.name}...", end="", flush=True)
        else:
            print(f"  Calling OpenAI Whisper API...", end="", flush=True)

        with open(chunk, "rb") as f:
            kwargs = {
                "model": "whisper-1",
                "file": f,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
            }
            if language:
                kwargs["language"] = language

            response = client.audio.transcriptions.create(**kwargs)

        print(" done")

        for seg in response.segments:
            all_segments.append({
                "start": seg.start + time_offset,
                "end": seg.end + time_offset,
                "text": seg.text.strip(),
                "speaker": "SPEAKER_00",  # OpenAI API has no diarization
            })

        # Advance offset by duration of this chunk
        if response.segments:
            time_offset += response.segments[-1].end

    # Cleanup temp chunks
    if len(chunks) > 1:
        for c in chunks:
            c.unlink(missing_ok=True)
        chunk_dir.rmdir()

    return all_segments


def run_openai_api(
    audio_files: list,
    output_dir: Path,
    language: str,
    glossary_path,
    skip_llm: bool
):
    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not installed.")
        print("  pip install openai")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not set.")
        print("  Set with: $env:OPENAI_API_KEY = 'sk-...'  (PowerShell)")
        return

    client = OpenAI(api_key=api_key)
    print("[INFO] Backend: OpenAI Whisper API (whisper-1)")
    print("[INFO] Note: no speaker diarization — all segments labeled SPEAKER_00")
    print(f"[INFO] Estimated cost: ${len(audio_files) * 0.006 * 60:.2f} max (at $0.006/min)")

    for audio_path in sorted(audio_files):
        print(f"\n[PROCESSING] {audio_path.name}")
        try:
            segments = transcribe_openai_api(audio_path, client, language)
            lang_label = language or "auto"
            print(f"  Language: {lang_label} | Segments: {len(segments)}")

            stem = audio_path.stem
            result = {"segments": segments, "backend": "openai-api"}
            format_transcript_txt(segments, output_dir / f"{stem}.txt")
            format_transcript_json(result, output_dir / f"{stem}.json")
            run_llm(stem, output_dir, glossary_path, skip_llm)

        except Exception as e:
            print(f"  [ERROR] {e}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audio transcription — CPU (WhisperX) or OpenAI Whisper API",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True,
                        help="Folder containing audio/video files")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: input/transcripts)")
    parser.add_argument("--language", default=None,
                        help="Language code: it, de, en (default: auto-detect)")
    parser.add_argument("--backend", default="cpu",
                        choices=["cpu", "openai-api", "groq-api"],
                        help=(
                            "cpu        = WhisperX local CPU, model selectable (slow)\n"
                            "openai-api = OpenAI Whisper API whisper-1, needs OPENAI_API_KEY\n"
                            "groq-api   = Groq whisper-large-v3-turbo, needs GROQ_API_KEY, free tier 7200 min/day"
                        ))
    parser.add_argument("--model", default="turbo",
                        help="WhisperX model (cpu backend only): tiny/medium/large-v2/large-v3/turbo (default: turbo)")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token for speaker diarization (cpu backend only)")
    parser.add_argument("--glossary", default=None,
                        help="Path to domain glossary .txt for LLM correction")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM correction + summary step")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Not found: {input_path}")
        return

    # Accept single file or folder
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"[ERROR] Unsupported format: {input_path.suffix}")
            print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            return
        audio_files = [input_path]
        default_output = input_path.parent / "transcripts"
    else:
        audio_files = [f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not audio_files:
            print(f"[ERROR] No audio files found in {input_path}")
            print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            return
        default_output = input_path / "transcripts"

    output_dir = Path(args.output) if args.output else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(audio_files)} file(s) | Backend: {args.backend}")

    glossary_path = Path(args.glossary) if args.glossary else None
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.backend == "cpu":
        run_whisperx_cpu(
            audio_files=audio_files,
            output_dir=output_dir,
            language=args.language,
            model_name=args.model,
            hf_token=hf_token,
            glossary_path=glossary_path,
            skip_llm=args.skip_llm
        )
    elif args.backend == "openai-api":
        run_openai_api(
            audio_files=audio_files,
            output_dir=output_dir,
            language=args.language,
            glossary_path=glossary_path,
            skip_llm=args.skip_llm
        )
    elif args.backend == "groq-api":
        run_groq_api(
            audio_files=audio_files,
            output_dir=output_dir,
            language=args.language,
            glossary_path=glossary_path,
            skip_llm=args.skip_llm
        )

    print(f"\n[DONE] Output folder: {output_dir}")


# ─── BACKEND 3: Groq Whisper API (turbo, free tier) ──────────────────────────

def run_groq_api(
    audio_files: list,
    output_dir: Path,
    language: str,
    glossary_path,
    skip_llm: bool
):
    try:
        from groq import Groq
    except ImportError:
        print("[ERROR] groq package not installed. Run: pip install groq")
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not set.")
        print("  Get free key: https://console.groq.com")
        print('  $env:GROQ_API_KEY = "gsk_..."')
        return

    client = Groq(api_key=api_key)
    print("[INFO] Backend: Groq whisper-large-v3-turbo (free: 7200 min/day)")
    print("[INFO] No speaker diarization — all labeled SPEAKER_00")

    for audio_path in sorted(audio_files):
        print(f"\n[PROCESSING] {audio_path.name}")
        file_size = audio_path.stat().st_size
        print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
        try:
            if file_size > OPENAI_MAX_BYTES:
                print("  File > 25MB — splitting...")
                chunk_dir = audio_path.parent / f"_chunks_{audio_path.stem}"
                chunks = split_audio_ffmpeg(audio_path, chunk_dir)
                print(f"  Split into {len(chunks)} chunk(s)")
            else:
                chunks = [audio_path]

            all_segments = []
            time_offset = 0.0

            for i, chunk in enumerate(chunks):
                label = f"Chunk {i+1}/{len(chunks)}" if len(chunks) > 1 else "Calling Groq API"
                print(f"  {label}...", end="", flush=True)
                with open(chunk, "rb") as f:
                    kwargs = {
                        "model": "whisper-large-v3-turbo",
                        "file": (chunk.name, f),
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["segment"],
                    }
                    if language:
                        kwargs["language"] = language
                    response = client.audio.transcriptions.create(**kwargs)
                print(" done")

                segments = response.segments
                for seg in segments:
                    # Groq returns dicts, OpenAI returns objects
                    s_start = seg["start"] if isinstance(seg, dict) else seg.start
                    s_end = seg["end"] if isinstance(seg, dict) else seg.end
                    s_text = seg["text"] if isinstance(seg, dict) else seg.text
                    all_segments.append({
                        "start":   s_start + time_offset,
                        "end":     s_end   + time_offset,
                        "text":    s_text.strip(),
                        "speaker": "SPEAKER_00",
                    })
                if segments:
                    last = segments[-1]
                    time_offset += last["end"] if isinstance(last, dict) else last.end

            if len(chunks) > 1:
                for c in chunks:
                    c.unlink(missing_ok=True)
                chunk_dir.rmdir()

            stem = audio_path.stem
            format_transcript_txt(all_segments, output_dir / f"{stem}.txt")
            format_transcript_json({"segments": all_segments, "backend": "groq-api"}, output_dir / f"{stem}.json")
            run_llm(stem, output_dir, glossary_path, skip_llm)

        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()

