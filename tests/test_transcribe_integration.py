"""Integration tests: transcribe.py ↔ audio_enhance.py.

Tests that the enhancement step is wired correctly into the transcription
pipeline without actually calling Whisper/Groq/OpenAI APIs.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


# ═════════════════════════════════════════════════════════════════════════════
# 1. IMPORT FLAG
# ═════════════════════════════════════════════════════════════════════════════

class TestEnhanceAvailableFlag:
    """Verify that ENHANCE_AVAILABLE reflects actual import success."""

    def test_flag_is_true_when_deps_present(self):
        """With audio_enhance importable, flag should be True."""
        import transcribe
        assert transcribe.ENHANCE_AVAILABLE is True

    def test_flag_is_false_when_import_fails(self):
        """Simulate missing audio_enhance → flag should be False."""
        # Temporarily remove audio_enhance from sys.modules
        saved = sys.modules.pop("audio_enhance", None)
        saved_transcribe = sys.modules.pop("transcribe", None)
        # Block the import
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "audio_enhance":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import
            import transcribe
            importlib.reload(transcribe)
            assert transcribe.ENHANCE_AVAILABLE is False
        finally:
            builtins.__import__ = real_import
            # Restore
            if saved is not None:
                sys.modules["audio_enhance"] = saved
            if saved_transcribe is not None:
                sys.modules["transcribe"] = saved_transcribe
            else:
                sys.modules.pop("transcribe", None)
            # Re-import cleanly
            import transcribe
            importlib.reload(transcribe)


# ═════════════════════════════════════════════════════════════════════════════
# 2. STEM MAP USAGE IN BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

class TestStemMapInBackends:
    """Verify that backends use stem_map for output naming."""

    def _make_segments(self):
        return [{"start": 0, "end": 5, "text": "Hello world", "speaker": "SPEAKER_00"}]

    def test_groq_uses_stem_map_for_output_name(self, tmp_path):
        """Groq backend should name output from stem_map, not the file stem."""
        import transcribe

        enhanced_path = tmp_path / "meeting_enhanced.wav"
        # Create a small valid file so stat() works
        sf.write(str(enhanced_path), np.zeros(16000, dtype=np.float64), 16000)

        stem_map = {enhanced_path: "meeting"}
        output_dir = tmp_path / "transcripts"
        output_dir.mkdir()

        # Mock the Groq client and its response
        mock_response = MagicMock()
        mock_response.segments = [
            {"start": 0, "end": 5, "text": "Hello", "id": 0, "seek": 0,
             "temperature": 0, "avg_logprob": 0, "compression_ratio": 1,
             "no_speech_prob": 0, "tokens": []}
        ]

        mock_groq_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_groq_cls.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}):
            with patch.dict("sys.modules", {"groq": MagicMock(Groq=mock_groq_cls)}):
                # Re-import to pick up mock
                from importlib import reload
                reload(transcribe)

                transcribe.run_groq_api(
                    audio_files=[enhanced_path],
                    output_dir=output_dir,
                    language="it",
                    glossary_path=None,
                    skip_llm=True,
                    stem_map=stem_map,
                )

        # Transcript should be named "meeting.txt", NOT "meeting_enhanced.txt"
        assert (output_dir / "meeting.txt").exists(), "Output should use original stem"
        assert not (output_dir / "meeting_enhanced.txt").exists()

    def test_openai_uses_stem_map_for_output_name(self, tmp_path):
        """OpenAI backend should name output from stem_map."""
        import transcribe

        enhanced_path = tmp_path / "call_enhanced.wav"
        sf.write(str(enhanced_path), np.zeros(16000, dtype=np.float64), 16000)

        stem_map = {enhanced_path: "call"}
        output_dir = tmp_path / "transcripts"
        output_dir.mkdir()

        # Mock segment with attribute access (OpenAI style)
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 5.0
        mock_seg.text = "Test"

        mock_response = MagicMock()
        mock_response.segments = [mock_seg]

        mock_openai_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake"}):
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
                from importlib import reload
                reload(transcribe)

                transcribe.run_openai_api(
                    audio_files=[enhanced_path],
                    output_dir=output_dir,
                    language="en",
                    glossary_path=None,
                    skip_llm=True,
                    stem_map=stem_map,
                )

        assert (output_dir / "call.txt").exists()
        assert not (output_dir / "call_enhanced.txt").exists()

    def test_no_stem_map_uses_file_stem(self, tmp_path):
        """Without stem_map, backend falls back to audio_path.stem."""
        import transcribe

        audio_path = tmp_path / "raw_audio.wav"
        sf.write(str(audio_path), np.zeros(16000, dtype=np.float64), 16000)

        output_dir = tmp_path / "transcripts"
        output_dir.mkdir()

        mock_response = MagicMock()
        mock_response.segments = [
            {"start": 0, "end": 5, "text": "Hi", "id": 0, "seek": 0,
             "temperature": 0, "avg_logprob": 0, "compression_ratio": 1,
             "no_speech_prob": 0, "tokens": []}
        ]

        mock_groq_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_groq_cls.return_value = mock_client

        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}):
            with patch.dict("sys.modules", {"groq": MagicMock(Groq=mock_groq_cls)}):
                from importlib import reload
                reload(transcribe)

                transcribe.run_groq_api(
                    audio_files=[audio_path],
                    output_dir=output_dir,
                    language="en",
                    glossary_path=None,
                    skip_llm=True,
                    stem_map=None,
                )

        assert (output_dir / "raw_audio.txt").exists()


# ═════════════════════════════════════════════════════════════════════════════
# 3. MAIN INTEGRATION FLOW
# ═════════════════════════════════════════════════════════════════════════════

class TestMainEnhancementFlow:
    """Test that main() calls enhance_audio_files at the right point."""

    def test_main_calls_enhance_before_backend(self, tmp_path):
        """Enhancement should run between file discovery and backend dispatch."""
        import transcribe

        audio = tmp_path / "test.wav"
        sf.write(str(audio), np.zeros(16000, dtype=np.float64), 16000)

        call_order = []

        original_enhance = transcribe.enhance_audio_files if transcribe.ENHANCE_AVAILABLE else None

        def mock_enhance(files):
            call_order.append("enhance")
            # Return files as-is with empty stem_map (no actual processing)
            return files, {}

        def mock_groq(*args, **kwargs):
            call_order.append("backend")

        with patch.object(transcribe, "enhance_audio_files", side_effect=mock_enhance):
            with patch.object(transcribe, "ENHANCE_AVAILABLE", True):
                with patch.object(transcribe, "run_groq_api", side_effect=mock_groq):
                    with patch("sys.argv", ["transcribe.py",
                                            "--input", str(audio),
                                            "--backend", "groq-api",
                                            "--language", "it",
                                            "--skip-llm"]):
                        transcribe.main()

        assert call_order == ["enhance", "backend"], (
            f"Expected enhance then backend, got: {call_order}"
        )

    def test_main_skips_enhance_when_unavailable(self, tmp_path, capsys):
        """When ENHANCE_AVAILABLE is False, main prints fallback message."""
        import transcribe

        audio = tmp_path / "test.wav"
        sf.write(str(audio), np.zeros(16000, dtype=np.float64), 16000)

        def mock_groq(*args, **kwargs):
            pass

        with patch.object(transcribe, "ENHANCE_AVAILABLE", False):
            with patch.object(transcribe, "run_groq_api", side_effect=mock_groq):
                with patch("sys.argv", ["transcribe.py",
                                        "--input", str(audio),
                                        "--backend", "groq-api",
                                        "--language", "it",
                                        "--skip-llm"]):
                    transcribe.main()

        captured = capsys.readouterr()
        assert "unavailable" in captured.out.lower()
