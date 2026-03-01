"""
Pipeline Runner — background subprocess executor for transcribe + LLM.

Runs transcribe.py and llm_process.py as subprocesses with correct
environment (.env loading, conda Python discovery, ffmpeg PATH).
Thread-safe status callbacks push to the widget's command queue.

Usage:
    runner = PipelineRunner(project_dir=Path(".").resolve(), on_status=..., on_complete=...)
    runner.transcribe(Path("./audio/rec.wav"), backend="groq-api", language="it")
    runner.run_llm(Path("./audio/transcripts/rec.txt"))
"""

import os
import subprocess
import sys
import threading
from pathlib import Path


class PipelineRunner:
    """Runs transcribe.py / llm_process.py as background subprocesses."""

    def __init__(self, project_dir: Path, on_status=None, on_complete=None):
        """
        Args:
            project_dir: Root directory of the SpyMeet project (contains .env, transcribe.py, etc.)
            on_status: Callback(msg: str) — called with status updates (thread-safe).
            on_complete: Callback(success: bool, msg: str) — called when task finishes.
        """
        self._project_dir = Path(project_dir)
        self._on_status = on_status
        self._on_complete = on_complete
        self._thread = None
        self._process = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def transcribe(self, audio_path: Path, backend="groq-api", language="it",
                   skip_llm=True):
        """Run transcribe.py on audio_path in a background thread.

        Args:
            audio_path: Path to the audio file.
            backend: Transcription backend (groq-api, openai-api, cpu).
            language: Language code (it, de, en, etc.).
            skip_llm: If True, pass --skip-llm flag.
        """
        if self.is_running:
            self._emit_status("Pipeline already running")
            return

        python = self._find_python()
        script = str(self._project_dir / "transcribe.py")
        args = [python, "-u", script,
                "--input", str(audio_path),
                "--backend", backend]
        if language:
            args += ["--language", language]
        if skip_llm:
            args.append("--skip-llm")

        label = f"Transcribing ({backend})..."
        self._run_in_thread(args, label)

    def enhance(self, audio_path: Path):
        """Run audio_enhance.py on audio_path in a background thread."""
        if self.is_running:
            self._emit_status("Pipeline already running")
            return

        python = self._find_python()
        script = str(self._project_dir / "audio_enhance.py")
        args = [python, "-u", script, "--input", str(audio_path)]

        self._run_in_thread(args, "Enhancing audio...")

    def run_llm(self, transcript_path: Path, glossary: Path | None = None):
        """Run llm_process.py on a transcript in a background thread.

        Args:
            transcript_path: Path to the .txt transcript.
            glossary: Optional path to glossary.txt.
        """
        if self.is_running:
            self._emit_status("Pipeline already running")
            return

        python = self._find_python()
        script = str(self._project_dir / "llm_process.py")
        args = [python, "-u", script, "--input", str(transcript_path)]
        if glossary and glossary.exists():
            args += ["--glossary", str(glossary)]

        self._run_in_thread(args, "Running LLM correction...")

    # ── Python discovery ──────────────────────────────────────────────────

    @staticmethod
    def _find_python() -> str:
        """Find the conda social_env Python executable."""
        # 1. Check CONDA_PREFIX if it IS social_env (not base)
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix and "social_env" in conda_prefix:
            candidate = Path(conda_prefix) / "python.exe"
            if candidate.exists():
                return str(candidate)

        # 2. Known conda locations for social_env
        home = Path.home()
        for env_name in ("social_env",):
            for base in (home / ".conda" / "envs",
                         home / "anaconda3" / "envs",
                         home / "miniconda3" / "envs",
                         Path("C:/ProgramData/anaconda3/envs")):
                candidate = base / env_name / "python.exe"
                if candidate.exists():
                    return str(candidate)

        # 3. CONDA_PREFIX (any active env)
        if conda_prefix:
            candidate = Path(conda_prefix) / "python.exe"
            if candidate.exists():
                return str(candidate)

        # 4. Current interpreter
        return sys.executable

    # ── .env loading ──────────────────────────────────────────────────────

    def _load_env(self) -> dict:
        """Read .env file and build subprocess environment dict.

        Mirrors _run_transcribe.ps1 key mapping:
            groq_api → GROQ_API_KEY
            anthropic_api → ANTHROPIC_API_KEY
            openai_api → OPENAI_API_KEY
            hf_token → HF_TOKEN

        Also handles direct uppercase keys (e.g. GROQ_API_KEY=...).
        Adds ffmpeg conda path to PATH and sets KMP_DUPLICATE_LIB_OK.
        """
        env = os.environ.copy()

        key_map = {
            "groq_api": "GROQ_API_KEY",
            "anthropic_api": "ANTHROPIC_API_KEY",
            "openai_api": "OPENAI_API_KEY",
            "hf_token": "HF_TOKEN",
        }

        env_file = self._project_dir / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]

                    # Map key
                    key_lower = key.lower()
                    env_name = key_map.get(key_lower, key.upper())
                    env[env_name] = value
            except Exception as e:
                print(f"[PipelineRunner] Warning: failed to read .env: {e}")

        # Add ffmpeg to PATH
        python_path = self._find_python()
        conda_bin = Path(python_path).parent / "Library" / "bin"
        if conda_bin.is_dir():
            env["PATH"] = str(conda_bin) + os.pathsep + env.get("PATH", "")

        # KMP_DUPLICATE_LIB_OK for torch
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        return env

    # ── Internal: background execution ────────────────────────────────────

    def _run_in_thread(self, args: list, label: str):
        """Launch subprocess in a background thread."""
        self._emit_status(label)
        self._thread = threading.Thread(
            target=self._subprocess_worker, args=(args, label), daemon=True
        )
        self._thread.start()

    def _subprocess_worker(self, args: list, label: str):
        """Worker thread: run subprocess, stream output, report completion."""
        env = self._load_env()
        try:
            # CREATE_NO_WINDOW flag on Windows
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(self._project_dir),
                creationflags=creationflags,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[Pipeline] {line}")

            self._process.wait()
            rc = self._process.returncode
            self._process = None

            if rc == 0:
                self._emit_complete(True, "Done")
            else:
                self._emit_complete(False, f"Error (exit code {rc})")

        except FileNotFoundError:
            self._emit_complete(False, f"Python not found: {args[0]}")
        except Exception as e:
            self._emit_complete(False, f"Error: {e}")
        finally:
            self._process = None

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _emit_status(self, msg: str):
        print(f"[PipelineRunner] {msg}")
        if self._on_status:
            self._on_status(msg)

    def _emit_complete(self, success: bool, msg: str):
        print(f"[PipelineRunner] Complete: {msg} (success={success})")
        if self._on_complete:
            self._on_complete(success, msg)
