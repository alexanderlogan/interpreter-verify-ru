"""
Ollama Translation Engine

Translates text between Russian and English using a local LLM
(Qwen 2.5 7B) running via Ollama. Includes medical terminology
awareness through system prompts.

All processing is local. No data leaves the machine.
"""
import time
import json
import requests
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """A translated text segment with metadata."""
    source_text: str
    translated_text: str
    source_language: str       # "en" or "ru"
    target_language: str       # "en" or "ru"
    translation_time: float    # seconds
    model: str

    @property
    def direction(self) -> str:
        return f"{self.source_language.upper()} -> {self.target_language.upper()}"

    def __str__(self) -> str:
        return f"[{self.direction}] {self.translated_text}"


class OllamaEngine:
    """
    Local LLM translation engine using Ollama.

    Usage:
        engine = OllamaEngine()
        result = engine.translate("У пациента высокое давление", "ru")
        print(result.translated_text)  # "The patient has high blood pressure"
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct-q4_K_M",
        timeout: int = 30,
    ):
        """
        Initialize the Ollama translation engine.

        Args:
            base_url: Ollama API URL (default localhost).
            model: Model name as installed in Ollama.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url
        self._model = model
        self._timeout = timeout

        # System prompts optimized for medical translation
        self._prompt_ru_to_en = (
            "You are a professional medical interpreter translating Russian "
            "to English. Rules:\n"
            "1. Translate accurately, preserving medical meaning.\n"
            "2. For Russian drug names, add the US equivalent in parentheses "
            "if you know it. Example: Энап (Vasotec/enalapril).\n"
            "3. For false friends, translate the CORRECT meaning. "
            "Example: ангина = tonsillitis, NOT angina pectoris.\n"
            "4. Keep the translation natural and professional.\n"
            "5. Output ONLY the English translation. No explanations, "
            "no notes, no preamble."
        )

        self._prompt_en_to_ru = (
            "You are a professional medical interpreter translating English "
            "to Russian. Rules:\n"
            "1. Translate accurately, preserving medical meaning.\n"
            "2. For US drug names, add the Russian equivalent in parentheses "
            "if you know it. Example: Augmentin (Амоксиклав).\n"
            "3. Use standard Russian medical terminology.\n"
            "4. Keep the translation natural and professional.\n"
            "5. Output ONLY the Russian translation. No explanations, "
            "no notes, no preamble."
        )

        # Verify connection
        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is running and the model is available."""
        try:
            resp = requests.get(
                f"{self._base_url}/api/tags",
                timeout=5
            )
            if resp.status_code != 200:
                raise ConnectionError("Ollama returned non-200 status.")

            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]

            # Check if our model is available (handle tag variations)
            model_found = False
            for name in model_names:
                if self._model in name or name in self._model:
                    model_found = True
                    break

            if not model_found:
                print(f"  [OLLAMA] WARNING: Model '{self._model}' not found.")
                print(f"  [OLLAMA] Available models: {model_names}")
                print(f"  [OLLAMA] Run: ollama pull {self._model}")
            else:
                print(f"  [OLLAMA] Connected. Model: {self._model}")

        except requests.ConnectionError:
            print(f"  [OLLAMA] ERROR: Cannot connect to {self._base_url}")
            print(f"  [OLLAMA] Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"  [OLLAMA] Connection check failed: {e}")

    def translate(
        self,
        text: str,
        source_language: str,
    ) -> TranslationResult | None:
        """
        Translate text between Russian and English.

        Args:
            text: Text to translate.
            source_language: "ru" or "en" (determines translation direction).

        Returns:
            TranslationResult or None if translation failed.
        """
        if not text or not text.strip():
            return None

        text = text.strip()

        # Select direction
        if source_language == "ru":
            system_prompt = self._prompt_ru_to_en
            target_language = "en"
        elif source_language == "en":
            system_prompt = self._prompt_en_to_ru
            target_language = "ru"
        else:
            print(f"  [OLLAMA] Unknown source language: {source_language}")
            return None

        # Build request
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "options": {
                "temperature": 0.1,      # Low temp for consistent translation
                "num_predict": 512,      # Max output tokens
                "top_p": 0.9,
            },
        }

        start = time.time()

        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data["message"]["content"].strip()
            elapsed = time.time() - start

            # Clean up common LLM artifacts
            translated = self._clean_output(translated)

            direction = "RU->EN" if source_language == "ru" else "EN->RU"
            preview = translated[:80] + "..." if len(translated) > 80 else translated
            print(f"  [OLLAMA] [{direction}] ({elapsed:.1f}s) {preview}")

            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_language=source_language,
                target_language=target_language,
                translation_time=elapsed,
                model=self._model,
            )

        except requests.Timeout:
            elapsed = time.time() - start
            print(f"  [OLLAMA] Timeout after {elapsed:.1f}s")
            return None
        except requests.ConnectionError:
            print(f"  [OLLAMA] Connection lost. Is Ollama still running?")
            return None
        except Exception as e:
            elapsed = time.time() - start
            print(f"  [OLLAMA] Error ({elapsed:.1f}s): {e}")
            return None

    def _clean_output(self, text: str) -> str:
        """Remove common LLM artifacts from translation output."""
        # Remove quotation marks wrapping the entire output
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Remove common preambles the LLM might add despite instructions
        preambles = [
            "Here is the translation:",
            "Translation:",
            "Here's the translation:",
            "Перевод:",
            "Вот перевод:",
        ]
        for p in preambles:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()

        return text.strip()

    def warm_up(self):
        """
        Send a short test request to pre-load the model into memory.
        First request is always slowest as the model loads from disk.
        """
        print(f"  [OLLAMA] Warming up model (first request is slow)...")
        start = time.time()
        result = self.translate("Здравствуйте", "ru")
        elapsed = time.time() - start
        if result:
            print(f"  [OLLAMA] Warm-up complete ({elapsed:.1f}s)")
        else:
            print(f"  [OLLAMA] Warm-up failed ({elapsed:.1f}s)")

    @property
    def model(self) -> str:
        return self._model
