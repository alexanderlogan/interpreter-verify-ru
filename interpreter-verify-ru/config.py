"""
Interpreter-Verify-RU Configuration
All settings for the application in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio capture settings."""
    sample_rate: int = 16000          # Whisper expects 16kHz
    channels: int = 1                  # Mono for transcription
    chunk_duration_ms: int = 30        # 30ms chunks for VAD
    device_index: int | None = None    # None = default output device
    silence_threshold_db: float = -40.0


@dataclass
class WhisperConfig:
    """Whisper transcription settings."""
    model_name: str = "distil-large-v3"   # CPU-optimized
    device: str = "cpu"
    compute_type: str = "int8"             # Quantized for speed
    beam_size: int = 5
    vad_filter: bool = True
    # Language detection: None = auto-detect per segment
    language: str | None = None
    # VAD parameters
    min_speech_duration_ms: int = 250
    max_speech_duration_s: int = 15
    min_silence_duration_ms: int = 800


@dataclass
class OllamaConfig:
    """Local LLM (Ollama) settings."""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    timeout_seconds: int = 30
    # System prompts for translation
    system_prompt_ru_to_en: str = (
        "You are a professional medical interpreter translating Russian to "
        "English. Translate the following Russian medical speech accurately "
        "into English. Preserve medical terminology precisely. If you detect "
        "a Russian medication name, include the US equivalent in parentheses "
        "if known. Output ONLY the translation, nothing else."
    )
    system_prompt_en_to_ru: str = (
        "You are a professional medical interpreter translating English to "
        "Russian. Translate the following English medical speech accurately "
        "into Russian. Preserve medical terminology precisely. If you detect "
        "a US medication name, include the Russian equivalent in parentheses "
        "if known. Output ONLY the translation, nothing else."
    )
    # System prompt for pharmaceutical audit
    system_prompt_audit: str = (
        "You are a bilingual (English/Russian) medical terminology auditor. "
        "Given a transcript segment and detected medication names, check for: "
        "1) False friends (words that sound similar in EN/RU but mean different "
        "things). 2) Drug warnings (FDA status, clinical risks). 3) Unknown "
        "terms needing clarification. Respond ONLY in JSON: "
        '{"flags": [{"term": "...", "type": "false_friend|drug_warning|unknown", '
        '"severity": "critical|high|medium|low", "message": "..."}], '
        '"clean": true/false}'
    )


@dataclass
class PharmaConfig:
    """Pharmaceutical lookup settings."""
    database_path: Path = Path("src/pharma/pharma_map.json")
    fuzzy_match_threshold: int = 80    # 0-100, higher = stricter matching
    enable_fuzzy: bool = True


@dataclass
class UIConfig:
    """Application window settings."""
    window_title: str = "Interpreter-Verify-RU"
    window_width: int = 600
    window_height: int = 800
    always_on_top: bool = True
    font_size: int = 12
    # Colors
    color_russian: str = "#2196F3"      # Blue for Russian text
    color_english: str = "#4CAF50"      # Green for English text
    color_warning: str = "#FF9800"      # Orange for warnings
    color_critical: str = "#F44336"     # Red for critical alerts
    color_drug_highlight: str = "#FFF9C4"  # Yellow background for drug names


@dataclass
class AppConfig:
    """Master configuration combining all settings."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    pharma: PharmaConfig = field(default_factory=PharmaConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Application-level settings
    app_name: str = "Interpreter-Verify-RU"
    version: str = "0.1.0"
    debug: bool = False
    log_transcripts: bool = False      # Set True to save transcripts to file
    log_directory: Path = Path("logs")


# Global config instance
config = AppConfig()
