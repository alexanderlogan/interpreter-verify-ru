"""
Whisper Transcription Engine

Transcribes audio segments using Faster-Whisper (CTranslate2 backend).
Automatically detects whether each segment is English or Russian.
Optimized for CPU with int8 quantization.

First run will download the model (~1.5 GB). Subsequent runs use the cache.
"""
import time
import numpy as np
from dataclasses import dataclass

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError(
        "faster-whisper is required for transcription.\n"
        "Install it with: pip install faster-whisper"
    )


@dataclass
class TranscriptSegment:
    """A single transcribed segment with metadata."""
    text: str
    language: str              # "en" or "ru"
    language_confidence: float # 0.0 to 1.0
    start_time: float         # seconds from segment start
    end_time: float           # seconds from segment start
    transcription_time: float # how long Whisper took (seconds)

    @property
    def is_russian(self) -> bool:
        return self.language == "ru"

    @property
    def is_english(self) -> bool:
        return self.language == "en"

    def __str__(self) -> str:
        lang_tag = "RU" if self.is_russian else "EN"
        return f"[{lang_tag}] {self.text}"


class WhisperEngine:
    """
    Whisper-based speech-to-text engine with language detection.

    Usage:
        engine = WhisperEngine()       # Loads model (slow first time)
        segments = engine.transcribe(audio_array)
        for seg in segments:
            print(f"[{seg.language}] {seg.text}")
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 3,
    ):
        """
        Initialize the Whisper engine.

        Args:
            model_name: Whisper model to use. "distil-large-v3" recommended
                       for CPU. Options: tiny, base, small, medium, large-v3,
                       distil-large-v3
            device: "cpu" or "cuda" (GPU)
            compute_type: "int8" for CPU speed, "float16" for GPU
            beam_size: Beam search width. Higher = more accurate, slower.
        """
        self._model_name = model_name
        self._beam_size = beam_size
        self._model = None
        self._device = device
        self._compute_type = compute_type

        print(f"  [WHISPER] Loading model: {model_name} ({compute_type})...")
        print(f"  [WHISPER] First run downloads ~1.5 GB. Please wait.")

        start = time.time()
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
        )
        elapsed = time.time() - start
        print(f"  [WHISPER] Model loaded in {elapsed:.1f}s")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: int = 15,
    ) -> list[TranscriptSegment]:
        """
        Transcribe an audio segment.

        Args:
            audio: Float32 numpy array, mono, 16kHz sample rate.
            language: Force language ("en" or "ru"). None = auto-detect.
            min_speech_duration_ms: Minimum speech segment length to keep.
            max_speech_duration_s: Maximum segment length before splitting.

        Returns:
            List of TranscriptSegment objects.
        """
        if audio is None or len(audio) == 0:
            return []

        # Ensure correct dtype
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed (Whisper expects -1.0 to 1.0)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak

        # Skip near-silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.001:
            return []

        start = time.time()

        # Run transcription
        segments_gen, info = self._model.transcribe(
            audio,
            beam_size=self._beam_size,
            language=language,
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=min_speech_duration_ms,
                max_speech_duration_s=max_speech_duration_s,
                min_silence_duration_ms=800,
                speech_pad_ms=200,
            ),
        )

        # Collect results
        results = []
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue

            elapsed = time.time() - start
            results.append(TranscriptSegment(
                text=text,
                language=info.language,
                language_confidence=info.language_probability,
                start_time=seg.start,
                end_time=seg.end,
                transcription_time=elapsed,
            ))

        elapsed = time.time() - start
        if results:
            lang = results[0].language.upper()
            conf = info.language_probability
            total_text = " ".join(r.text for r in results)
            preview = total_text[:80] + "..." if len(total_text) > 80 else total_text
            print(f"  [WHISPER] [{lang} {conf:.0%}] ({elapsed:.1f}s) {preview}")
        else:
            print(f"  [WHISPER] (no speech detected, {elapsed:.1f}s)")

        return results

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Detect the language of an audio segment without full transcription.
        Faster than full transcription when you only need the language.

        Args:
            audio: Float32 numpy array, mono, 16kHz.

        Returns:
            Tuple of (language_code, confidence). e.g. ("ru", 0.95)
        """
        if audio is None or len(audio) == 0:
            return ("unknown", 0.0)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Use first 30 seconds max for language detection
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        _, info = self._model.transcribe(
            audio,
            beam_size=1,     # Fast, we only need language
            vad_filter=True,
        )

        return (info.language, info.language_probability)

    @property
    def model_name(self) -> str:
        return self._model_name
