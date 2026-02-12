"""
Phase 3 Test: Audio Capture + Transcription + Translation

This script captures system audio, transcribes it with Whisper,
detects the language, and translates it using the local LLM.

How to test:
  1. Make sure Ollama is running (it usually runs automatically)
  2. Play audio with Russian OR English speech
  3. Run: python tests/test_translation.py
  4. Watch translations appear in real time

The pipeline:
  Audio -> Whisper (transcribe + detect language) -> Ollama (translate)
  Russian speech -> English translation
  English speech -> Russian translation
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import AudioCapture
from src.transcription.whisper_engine import WhisperEngine
from src.translation.ollama_engine import OllamaEngine


def test_translation_standalone():
    """Test translation without audio, using hardcoded medical sentences."""
    print("\n" + "=" * 60)
    print("  TEST 1: Standalone Translation (no audio)")
    print("=" * 60)

    engine = OllamaEngine()

    # Warm up the model (first request loads model into RAM)
    engine.warm_up()

    # Test sentences: Russian medical -> English
    ru_sentences = [
        "У пациента высокое давление, он принимает Энап.",
        "Больной жалуется на ангину и головную боль.",
        "Назначен Амоксиклав по 500 мг три раза в день.",
        "Пациентка принимает Корвалол перед сном.",
    ]

    print(f"\n  --- Russian to English ---")
    for sentence in ru_sentences:
        print(f"\n  Source: {sentence}")
        result = engine.translate(sentence, "ru")
        if result:
            print(f"  Translation: {result.translated_text}")
            print(f"  Time: {result.translation_time:.1f}s")
        else:
            print(f"  [FAILED]")

    # Test sentences: English medical -> Russian
    en_sentences = [
        "The patient has been taking Augmentin for five days.",
        "She reports chest pain and shortness of breath.",
        "Prescribe Lisinopril 10mg once daily for hypertension.",
        "The patient has a history of stroke and diabetes.",
    ]

    print(f"\n  --- English to Russian ---")
    for sentence in en_sentences:
        print(f"\n  Source: {sentence}")
        result = engine.translate(sentence, "en")
        if result:
            print(f"  Translation: {result.translated_text}")
            print(f"  Time: {result.translation_time:.1f}s")
        else:
            print(f"  [FAILED]")

    return True


def test_live_pipeline(duration: int = 30):
    """
    Full pipeline: capture audio -> transcribe -> translate.

    Captures audio in chunks, transcribes each chunk with Whisper,
    detects the language, and translates to the other language.
    """
    print("\n" + "=" * 60)
    print(f"  TEST 2: Live Pipeline ({duration} seconds)")
    print("=" * 60)
    print()
    print("  Play audio with speech (English or Russian).")
    print("  You will see: transcription + translation for each chunk.")
    print()

    # Initialize all engines
    whisper = WhisperEngine()
    ollama = OllamaEngine()
    capture = AudioCapture()

    # Warm up Ollama
    ollama.warm_up()

    try:
        capture.start()
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        capture.close()
        return False

    segment_duration = 5  # seconds per chunk
    results = []

    print(f"\n  Listening... (processing every {segment_duration}s)\n")
    print("  " + "-" * 56)

    try:
        start_time = time.time()
        chunk_num = 0

        while time.time() - start_time < duration:
            time.sleep(segment_duration)
            chunk_num += 1

            # Grab audio
            audio = capture.get_audio_chunk(clear=True)
            if audio is None or len(audio) == 0:
                continue

            # Skip silence
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                continue

            # Transcribe
            segments = whisper.transcribe(audio)
            if not segments:
                continue

            # Translate each segment
            for seg in segments:
                # Show original
                lang_tag = "RU" if seg.is_russian else "EN"
                print(f"\n  [{lang_tag}] {seg.text}")

                # Translate
                result = ollama.translate(seg.text, seg.language)
                if result:
                    target_tag = "EN" if seg.is_russian else "RU"
                    print(f"  [{target_tag}] {result.translated_text}")
                    print(f"       (whisper: {seg.transcription_time:.1f}s + "
                          f"translate: {result.translation_time:.1f}s = "
                          f"total: {seg.transcription_time + result.translation_time:.1f}s)")
                    results.append({
                        "source": seg.text,
                        "translation": result.translated_text,
                        "direction": result.direction,
                        "whisper_time": seg.transcription_time,
                        "translate_time": result.translation_time,
                        "total_time": seg.transcription_time + result.translation_time,
                    })

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    finally:
        capture.stop()
        capture.close()

    # Summary
    print(f"\n  " + "-" * 56)
    print(f"\n  Pipeline Summary:")
    print(f"  Translations completed: {len(results)}")

    if results:
        avg_whisper = sum(r["whisper_time"] for r in results) / len(results)
        avg_translate = sum(r["translate_time"] for r in results) / len(results)
        avg_total = sum(r["total_time"] for r in results) / len(results)
        print(f"  Avg whisper time:    {avg_whisper:.1f}s")
        print(f"  Avg translation time: {avg_translate:.1f}s")
        print(f"  Avg total pipeline:   {avg_total:.1f}s")

    return len(results) > 0


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  INTERPRETER-VERIFY-RU: Phase 3 Translation Test")
    print("#" * 60)

    # Test 1: Standalone translation (no audio needed)
    print("\n  Starting standalone translation test...")
    print("  (This tests the LLM directly with medical sentences.)")
    test_translation_standalone()

    # Test 2: Live pipeline
    print("\n  Ready for live pipeline test.")
    print("  Play audio with speech (English or Russian).")
    input("  Press ENTER to start 30-second live test... ")
    success = test_live_pipeline(30)

    # Summary
    print("\n" + "#" * 60)
    if success:
        print("  PHASE 3 TEST: PASSED")
        print()
        print("  Next steps:")
        print("    1. Commit to Git:")
        print("       git add .")
        print('       git commit -m "Phase 3: Ollama translation engine"')
        print("       git push")
        print("    2. Then we move to Phase 4: Pharmaceutical Intelligence")
    else:
        print("  PHASE 3 TEST: NEEDS ATTENTION")
        print()
        print("  Check the errors above and try again.")
    print("#" * 60 + "\n")
