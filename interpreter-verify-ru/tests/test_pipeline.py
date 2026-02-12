"""
Phase 3b Test: Threaded Pipeline (coverage fix)

Tests the threaded pipeline that runs audio capture, transcription,
and translation concurrently so no speech is dropped.

How to test:
  1. Make sure Ollama is running
  2. Play audio with continuous speech (Russian or English)
  3. Run: python tests/test_pipeline.py
  4. Watch for coverage percentage at the end
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, PipelineItem


def main(duration: int = 45):
    print("\n" + "#" * 60)
    print("  INTERPRETER-VERIFY-RU: Threaded Pipeline Test")
    print("#" * 60)
    print()
    print(f"  This test runs for {duration} seconds.")
    print("  Play continuous speech (Russian or English).")
    print("  All three stages run in parallel:")
    print("    Thread 1: Audio capture (never stops)")
    print("    Thread 2: Whisper transcription")
    print("    Thread 3: Ollama translation")
    print()

    results = []

    def on_result(item: PipelineItem):
        """Collect and display each result."""
        results.append(item)

        lang = "RU" if item.transcript.is_russian else "EN"
        print(f"\n  [{lang}] {item.transcript.text}")

        if item.translation:
            target = "EN" if item.transcript.is_russian else "RU"
            print(f"  [{target}] {item.translation.translated_text}")

    input("  Press ENTER when audio is playing to start... ")

    pipeline = Pipeline(on_result=on_result, segment_duration=4.0)

    try:
        pipeline.start()
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    finally:
        pipeline.stop()

    # Results summary
    print(f"\n  Results Summary:")
    print(f"  Total translations: {len(results)}")

    if results:
        ru = sum(1 for r in results if r.transcript.is_russian)
        en = sum(1 for r in results if r.transcript.is_english)
        print(f"  Russian segments: {ru}")
        print(f"  English segments: {en}")

        translated = sum(1 for r in results if r.translation)
        print(f"  Successfully translated: {translated}/{len(results)}")

    print("\n" + "#" * 60)
    if len(results) > 0:
        print("  THREADED PIPELINE TEST: PASSED")
        print()
        print("  Compare the coverage % above with the previous test.")
        print("  It should be significantly higher (close to 100%).")
        print()
        print("  To commit:")
        print("    git add .")
        print('    git commit -m "Phase 3b: Threaded pipeline for full coverage"')
        print("    git push")
    else:
        print("  THREADED PIPELINE TEST: NEEDS ATTENTION")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main(45)
