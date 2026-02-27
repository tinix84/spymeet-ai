"""
LLM Post-Processor for Transcripts
Uses Claude (Anthropic API) to:
  1. Correct transcript: fix punctuation, sentence boundaries, remove fillers,
     flag/retry nonsensical segments (up to MAX_RETRIES per chunk)
  2. Generate structured meeting summary in Markdown

Usage (manual / standalone):
    python llm_process.py --input ./audio/transcripts/meeting.txt
    python llm_process.py --input ./audio/transcripts/ --glossary glossary.txt

Called automatically by transcribe.py after each file is processed.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 3          # max correction retries per chunk if nonsensical
CHUNK_MINUTES = 5        # group segments into ~5min chunks for correction
RETRY_DELAY = 1.0        # seconds between retries (rate limit safety)

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    speaker: str
    timestamp: str   # mm:ss
    text: str

@dataclass
class CorrectedSegment:
    speaker: str
    timestamp: str
    original: str
    corrected: str
    warnings: list[str] = field(default_factory=list)
    retries_used: int = 0

# ─── Parsing ──────────────────────────────────────────────────────────────────

def parse_transcript_txt(path: Path) -> list[Segment]:
    """Parse the .txt output from transcribe.py into Segment objects."""
    segments = []
    current_speaker = "SPEAKER_?"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            # Speaker header: [SPEAKER_00]
            speaker_match = re.match(r"^\[([A-Z_0-9]+)\]$", line)
            if speaker_match:
                current_speaker = speaker_match.group(1)
                continue
            # Segment line: "  [mm:ss] text..."
            seg_match = re.match(r"^\s+\[(\d{2,}:\d{2})\]\s+(.+)$", line)
            if seg_match:
                ts = seg_match.group(1)
                text = seg_match.group(2).strip()
                if text:
                    segments.append(Segment(
                        speaker=current_speaker,
                        timestamp=ts,
                        text=text
                    ))
    return segments


def segments_to_chunks(segments: list[Segment], chunk_minutes: int) -> list[list[Segment]]:
    """Group segments into time-based chunks of ~chunk_minutes each."""
    if not segments:
        return []

    chunks = []
    current_chunk = []
    chunk_start_secs = None

    for seg in segments:
        mm, ss = map(int, seg.timestamp.split(":"))
        secs = mm * 60 + ss
        if chunk_start_secs is None:
            chunk_start_secs = secs

        if secs - chunk_start_secs > chunk_minutes * 60 and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            chunk_start_secs = secs

        current_chunk.append(seg)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_to_text(chunk: list[Segment]) -> str:
    """Serialize a chunk of segments to plain text for the LLM prompt."""
    lines = []
    for seg in chunk:
        lines.append(f"[{seg.speaker}][{seg.timestamp}] {seg.text}")
    return "\n".join(lines)

# ─── Correction ───────────────────────────────────────────────────────────────

CORRECTION_SYSTEM = """\
You are a professional transcript editor. Your task is to clean a raw speech-to-text transcript chunk.

Rules:
1. Fix punctuation and sentence boundaries (merge broken sentences, split run-ons).
2. Remove spoken filler words: uhm, uh, allora, also, quindi, cioè, praticamente, boh, mah, \
   naja, ähm, äh, sozusagen, halt, eigentlich — and their language variants.
3. Keep each speaker's original language (do NOT translate).
4. Preserve speaker labels and timestamps exactly as given: [SPEAKER_XX][mm:ss].
5. For each segment, evaluate if the corrected text is semantically coherent and meaningful.
   If a segment is still nonsensical or unintelligible after correction, add a WARNING line \
   immediately after that segment: WARNING: [mm:ss] <short reason>
6. Do NOT invent or add content. Only fix, clean, remove fillers.
7. If a glossary is provided, use it to fix domain-specific terms.

Output format — return ONLY the corrected segments in this exact format, no preamble:
[SPEAKER_XX][mm:ss] corrected text here
WARNING: [mm:ss] reason  ← only if segment is nonsensical
[SPEAKER_XX][mm:ss] next segment...
"""

def build_correction_prompt(chunk_text: str, glossary: Optional[str], attempt: int) -> str:
    prompt = chunk_text
    if glossary:
        prompt += f"\n\n--- DOMAIN GLOSSARY ---\n{glossary}"
    if attempt > 1:
        prompt += f"\n\n[RETRY ATTEMPT {attempt}: The previous correction still had nonsensical segments. \
Please re-examine flagged segments more carefully. If audio quality is clearly the issue, keep the \
best possible interpretation and mark with WARNING.]"
    return prompt


def parse_correction_response(response_text: str, original_chunk: list[Segment]) -> list[CorrectedSegment]:
    """Parse LLM correction output back into CorrectedSegment objects."""
    corrected = []
    orig_map = {seg.timestamp: seg for seg in original_chunk}

    lines = response_text.strip().split("\n")
    warnings_for_ts: dict[str, list[str]] = {}
    seg_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        warn_match = re.match(r"^WARNING:\s+\[(\d{2,}:\d{2})\]\s+(.+)$", line)
        if warn_match:
            ts = warn_match.group(1)
            reason = warn_match.group(2)
            warnings_for_ts.setdefault(ts, []).append(reason)
        else:
            seg_lines.append(line)

    for line in seg_lines:
        seg_match = re.match(r"^\[([A-Z_0-9]+)\]\[(\d{2,}:\d{2})\]\s+(.+)$", line)
        if seg_match:
            speaker = seg_match.group(1)
            ts = seg_match.group(2)
            text = seg_match.group(3).strip()
            original = orig_map.get(ts)
            corrected.append(CorrectedSegment(
                speaker=speaker,
                timestamp=ts,
                original=original.text if original else text,
                corrected=text,
                warnings=warnings_for_ts.get(ts, [])
            ))

    return corrected


def has_warnings(corrected_segments: list[CorrectedSegment]) -> bool:
    return any(seg.warnings for seg in corrected_segments)


def correct_chunk(
    client: anthropic.Anthropic,
    chunk: list[Segment],
    glossary: Optional[str],
    max_retries: int
) -> list[CorrectedSegment]:
    """Correct a chunk, retrying warned segments up to max_retries times."""
    chunk_text = chunk_to_text(chunk)
    accepted: list[CorrectedSegment] = []  # accumulates non-warned segments across retries

    for attempt in range(1, max_retries + 1):
        prompt = build_correction_prompt(chunk_text, glossary, attempt)
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=CORRECTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.content[0].text
        corrected = parse_correction_response(result_text, chunk)

        for seg in corrected:
            seg.retries_used = attempt - 1

        if not has_warnings(corrected) or attempt == max_retries:
            accepted.extend(corrected)
            break

        # Retry only the warned segments
        warned_ts = {seg.timestamp for seg in corrected if seg.warnings}
        warned_segments = [s for s in chunk if s.timestamp in warned_ts]
        chunk = warned_segments
        chunk_text = chunk_to_text(chunk)

        # Keep non-warned segments from this pass
        accepted.extend(s for s in corrected if not s.warnings)
        time.sleep(RETRY_DELAY)

    # Sort by timestamp to restore original order
    accepted.sort(key=lambda s: tuple(map(int, s.timestamp.split(":"))))
    return accepted


def correct_transcript(
    client: anthropic.Anthropic,
    segments: list[Segment],
    glossary: Optional[str] = None,
    chunk_minutes: int = CHUNK_MINUTES,
    max_retries: int = MAX_RETRIES
) -> tuple[list[CorrectedSegment], list[str]]:
    """
    Correct all segments in chunks.
    Returns (corrected_segments, all_warnings).
    """
    chunks = segments_to_chunks(segments, chunk_minutes)
    all_corrected = []
    all_warnings = []

    print(f"  [LLM] Correcting transcript in {len(chunks)} chunk(s)...")

    for i, chunk in enumerate(chunks):
        start_ts = chunk[0].timestamp
        end_ts = chunk[-1].timestamp
        print(f"    Chunk {i+1}/{len(chunks)}: [{start_ts} → {end_ts}] ({len(chunk)} segments)", end="", flush=True)

        corrected_chunk = correct_chunk(client, chunk, glossary, max_retries)
        all_corrected.extend(corrected_chunk)

        warnings_in_chunk = [(seg.timestamp, w) for seg in corrected_chunk for w in seg.warnings]
        if warnings_in_chunk:
            print(f" ⚠ {len(warnings_in_chunk)} warning(s)")
            for ts, w in warnings_in_chunk:
                all_warnings.append(f"[{ts}] {w}")
        else:
            print(" ✓")

    return all_corrected, all_warnings

# ─── Summary ──────────────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """\
You are an expert meeting analyst. Given a corrected meeting transcript, produce a structured \
meeting summary in the SAME LANGUAGE as the transcript.

Output a Markdown document with EXACTLY these sections in order:

# Meeting Summary

## Participants & Roles
List each identified speaker with their likely role or name if inferable from context. \
If role is unknown, write "Unknown".

## Key Decisions
Bullet list of concrete decisions made during the meeting. If none, write "No decisions recorded."

## Action Items
Table with columns: | Action | Owner | Deadline |
If no deadlines were mentioned, write "TBD". If no action items, write "No action items recorded."

## Open Questions / Unresolved Topics
Bullet list of questions raised but not resolved. If none, write "None."

---
*Generated by whisper-transcriber — review before distributing.*
"""

def generate_summary(
    client: anthropic.Anthropic,
    corrected_segments: list[CorrectedSegment],
    glossary: Optional[str] = None
) -> str:
    """Generate structured meeting summary from corrected transcript."""
    print("  [LLM] Generating meeting summary...", end="", flush=True)

    # Build full transcript text
    lines = []
    current_speaker = None
    for seg in corrected_segments:
        if seg.speaker != current_speaker:
            lines.append(f"\n[{seg.speaker}]")
            current_speaker = seg.speaker
        lines.append(f"  [{seg.timestamp}] {seg.corrected}")
    transcript_text = "\n".join(lines)

    prompt = transcript_text
    if glossary:
        prompt += f"\n\n--- DOMAIN GLOSSARY ---\n{glossary}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    print(" ✓")
    return response.content[0].text

# ─── Output writers ───────────────────────────────────────────────────────────

def save_corrected_txt(corrected: list[CorrectedSegment], warnings: list[str], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        if warnings:
            f.write("# ⚠ CORRECTION WARNINGS\n")
            for w in warnings:
                f.write(f"  {w}\n")
            f.write("\n" + "─" * 60 + "\n\n")

        current_speaker = None
        for seg in corrected:
            if seg.speaker != current_speaker:
                f.write(f"\n[{seg.speaker}]\n")
                current_speaker = seg.speaker
            warn_marker = " ⚠" if seg.warnings else ""
            f.write(f"  [{seg.timestamp}] {seg.corrected}{warn_marker}\n")
            for w in seg.warnings:
                f.write(f"    ⚠ WARNING: {w}\n")

    print(f"  → Saved corrected transcript: {output_path}")


def save_summary_md(summary_text: str, warnings: list[str], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
        if warnings:
            f.write("\n\n---\n\n## ⚠ Transcript Correction Warnings\n")
            f.write("*The following segments were flagged as potentially unintelligible:*\n\n")
            for w in warnings:
                f.write(f"- `{w}`\n")
    print(f"  → Saved summary: {output_path}")

# ─── Main processing function ─────────────────────────────────────────────────

def process_transcript_file(
    txt_path: Path,
    glossary_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
):
    """Full pipeline: parse → correct → summarize → save."""
    print(f"\n[LLM PROCESSING] {txt_path.name}")

    # Resolve output dir
    if output_dir is None:
        output_dir = txt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load glossary if provided
    glossary = None
    if glossary_path and glossary_path.exists():
        glossary = glossary_path.read_text(encoding="utf-8")
        print(f"  Glossary loaded: {glossary_path.name}")
    elif glossary_path:
        print(f"  [WARNING] Glossary file not found: {glossary_path}")

    # Init Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "In WSL: export ANTHROPIC_API_KEY=your_key_here"
        )
    client = anthropic.Anthropic(api_key=api_key)

    # Parse transcript
    segments = parse_transcript_txt(txt_path)
    if not segments:
        print(f"  [ERROR] No segments found in {txt_path.name} — skipping.")
        return
    print(f"  Parsed {len(segments)} segments")

    # Step 1: Correction
    t0 = time.time()
    corrected, warnings = correct_transcript(client, segments, glossary)
    correction_time = time.time() - t0
    print(f"  Correction done in {correction_time:.0f}s")

    # Step 2: Summary
    t1 = time.time()
    summary_text = generate_summary(client, corrected, glossary)
    summary_time = time.time() - t1
    print(f"  Summary done in {summary_time:.0f}s")

    # Save outputs
    stem = txt_path.stem
    corrected_path = output_dir / f"{stem}_corrected.txt"
    summary_path = output_dir / f"{stem}_summary.md"

    save_corrected_txt(corrected, warnings, corrected_path)
    save_summary_md(summary_text, warnings, summary_path)

    total = time.time() - t0
    if warnings:
        print(f"  ⚠ {len(warnings)} segment(s) flagged — review _corrected.txt")
    else:
        print("  ✓ No warnings")
    print(f"  LLM total: {total:.0f}s")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM post-processor: correct + summarize transcript files"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a .txt transcript file OR a folder containing .txt files"
    )
    parser.add_argument(
        "--glossary", default=None,
        help="Optional path to domain glossary .txt file"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output folder (default: same folder as input)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    glossary_path = Path(args.glossary) if args.glossary else None
    output_dir = Path(args.output) if args.output else None

    if input_path.is_dir():
        txt_files = [
            f for f in input_path.iterdir()
            if f.suffix == ".txt" and not f.stem.endswith("_corrected")
        ]
        if not txt_files:
            print(f"[ERROR] No .txt transcript files found in {input_path}")
            return
        print(f"[INFO] Found {len(txt_files)} transcript file(s)")
        for f in sorted(txt_files):
            process_transcript_file(f, glossary_path, output_dir or f.parent)
    elif input_path.is_file() and input_path.suffix == ".txt":
        process_transcript_file(input_path, glossary_path, output_dir or input_path.parent)
    else:
        print(f"[ERROR] Input must be a .txt file or a folder: {input_path}")


if __name__ == "__main__":
    main()
