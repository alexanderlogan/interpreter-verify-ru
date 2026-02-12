"""
Phase 2 Test: Audio Capture + Whisper Transcription

This script captures system audio and transcribes it in real time,
showing the detected language (English or Russian) for each segment.

How to test:
  1. Open a YouTube video with Russian or English speech
     (or use a telehealth test call)
  2. Run: python tests/test_whisper.py
  3. Speak or play audio for 30 seconds
  4. The script will show transcriptions with language labels

Recommended test videos:
  - English medical: search "doctor patient conversation example"
  - Russian medical: search "прием у врача диалог"
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import AudioCapture
from src.transcription.whisper_engine import WhisperEngine


def test_transcribe_file():
    """Transcribe the test_capture.wav from Phase 1 (if it exists)."""
    wav_path = Path("test_capture.wav")
    if not wav_path.exists():
        print("  [SKIP] No test_capture.wav found. Run test_audio.py first.")
        return False

    import wave
    print("\n" + "=" * 60)
    print("  TEST 1: Transcribe test_capture.wav")
    print("=" * 60)

    # Load WAV file
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        sample_rate = wf.getframerate()

    print(f"\n  Audio: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")

    # Load Whisper
    engine = WhisperEngine()

    # Transcribe
    segments = engine.transcribe(audio)

    if segments:
        print(f"\n  Results ({len(segments)} segments):")
        for seg in segments:
            lang_tag = "RU" if seg.is_russian else "EN"
            conf = seg.language_confidence
            print(f"    [{lang_tag} {conf:.0%}] {seg.text}")
        return True
    else:
        print("\n  [WARNING] No speech detected in test_capture.wav")
        print("  Was there actual speech in the recording?")
        return False


def test_live_transcription(duration: int = 30):
    """
    Capture and transcribe audio in real time.

    Captures audio in chunks, sends each chunk to Whisper,
    and prints the transcription with language detection.
    """
    print("\n" + "=" * 60)
    print(f"  TEST 2: Live Transcription ({duration} seconds)")
    print("=" * 60)
    print()
    print("  Play audio with speech (English or Russian).")
    print("  Transcriptions will appear as speech is detected.")
    print()

    # Load Whisper model (reuse if already loaded)
    engine = WhisperEngine()

    # Set up audio capture
    capture = AudioCapture()

    try:
        capture.start()
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        capture.close()
        return False

    # Capture in segments
    segment_duration = 5  # seconds per chunk
    segments_captured = 0
    all_results = []

    print(f"  Listening... (processing every {segment_duration}s)\n")

    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            # Wait for a chunk of audio
            time.sleep(segment_duration)

            # Grab audio from buffer
            audio = capture.get_audio_chunk(clear=True)
            if audio is None or len(audio) == 0:
                print(f"  ({segments_captured + 1}) [silence]")
                segments_captured += 1
                continue

            # Check audio level
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                print(f"  ({segments_captured + 1}) [silence]")
                segments_captured += 1
                continue

            # Transcribe
            segments_captured += 1
            results = engine.transcribe(audio)

            for seg in results:
                all_results.append(seg)
                lang_tag = "RU" if seg.is_russian else "EN"
                conf = seg.language_confidence
                print(f"  >> [{lang_tag} {conf:.0%}] {seg.text}")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        capture.stop()
        capture.close()

    # Summary
    print(f"\n" + "-" * 60)
    print(f"  Capture complete.")
    print(f"  Segments processed: {segments_captured}")
    print(f"  Transcriptions: {len(all_results)}")

    if all_results:
        en_count = sum(1 for r in all_results if r.is_english)
        ru_count = sum(1 for r in all_results if r.is_russian)
        avg_time = sum(r.transcription_time for r in all_results) / len(all_results)
        print(f"  English segments: {en_count}")
        print(f"  Russian segments: {ru_count}")
        print(f"  Avg transcription time: {avg_time:.1f}s")

    return len(all_results) > 0


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  INTERPRETER-VERIFY-RU: Phase 2 Whisper Test")
    print("#" * 60)

    # Test 1: Transcribe existing WAV (if available)
    test_transcribe_file()

    # Test 2: Live transcription
    print("\n  Ready for live transcription test.")
    print("  Play audio with speech (English or Russian).")
    input("  Press ENTER to start 30-second live capture... ")
    success = test_live_transcription(30)

    # Summary
    print("\n" + "#" * 60)
    if success:
        print("  PHASE 2 TEST: PASSED")
        print()
        print("  Next steps:")
        print("    1. Commit to Git:")
        print("       git add .")
        print('       git commit -m "Phase 2: Whisper transcription with language detection"')
        print("       git push")
        print("    2. Then we move to Phase 3: Translation")
    else:
        print("  PHASE 2 TEST: NEEDS ATTENTION")
        print()
        print("  Check the errors above and try again.")
    print("#" * 60 + "\n")
