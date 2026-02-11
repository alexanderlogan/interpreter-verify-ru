"""
Phase 1 Test: Audio Capture Verification

This script captures 10 seconds of system audio and saves it to a WAV file.
Play it back to verify the capture is working correctly.

How to test:
  1. Start playing audio on your computer (YouTube, Zoom, anything)
  2. Run: python tests/test_audio.py
  3. The script captures 10 seconds, then saves to test_capture.wav
  4. Open test_capture.wav and verify you hear the same audio

If no audio is captured, check:
  - Is audio actually playing through your speakers/headphones?
  - Run the device listing to see available devices
"""
import sys
import time
from pathlib import Path

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.capture import AudioCapture


def test_list_devices():
    """List all available loopback audio devices."""
    print("\n" + "=" * 60)
    print("  AVAILABLE LOOPBACK AUDIO DEVICES")
    print("=" * 60)

    capture = AudioCapture()
    devices = capture.list_devices()

    if not devices:
        print("\n  [ERROR] No loopback devices found!")
        print("  Make sure audio is playing through speakers or headphones.")
        capture.close()
        return False

    for dev in devices:
        default_tag = " << DEFAULT" if dev.is_default else ""
        print(f"\n  Device #{dev.index}: {dev.name}{default_tag}")
        print(f"    Channels: {dev.channels}")
        print(f"    Sample rate: {dev.sample_rate} Hz")

    capture.close()
    print(f"\n  Total loopback devices: {len(devices)}")
    return True


def test_capture_audio(duration: int = 10):
    """Capture system audio for the specified duration and save to WAV."""
    print("\n" + "=" * 60)
    print(f"  AUDIO CAPTURE TEST ({duration} seconds)")
    print("=" * 60)
    print()
    print("  Make sure audio is playing on your computer!")
    print("  (YouTube, music, anything that produces sound)")
    print()

    output_file = Path("test_capture.wav")

    with AudioCapture() as capture:
        # Start capturing
        try:
            capture.start()
        except RuntimeError as e:
            print(f"\n  [ERROR] {e}")
            return False

        # Countdown
        print()
        for remaining in range(duration, 0, -1):
            buf_dur = capture.buffer_duration
            print(f"  Recording... {remaining}s remaining  "
                  f"(buffer: {buf_dur:.1f}s)", end="\r")
            time.sleep(1)

        print(f"\n")

        # Stop and save
        capture.stop()
        capture.save_wav(output_file)

    if output_file.exists() and output_file.stat().st_size > 1000:
        print(f"\n  [SUCCESS] Audio captured and saved to: {output_file.absolute()}")
        print(f"  Open this file in any audio player to verify.")
        print(f"  You should hear whatever was playing on your computer.")
        return True
    else:
        print(f"\n  [WARNING] File is very small or empty.")
        print(f"  Was audio actually playing during the capture?")
        return False


def test_audio_levels():
    """Quick 3-second capture to check if audio levels are reasonable."""
    print("\n" + "=" * 60)
    print("  AUDIO LEVEL CHECK (3 seconds)")
    print("=" * 60)

    with AudioCapture() as capture:
        try:
            capture.start()
        except RuntimeError as e:
            print(f"\n  [ERROR] {e}")
            return False

        time.sleep(3)
        capture.stop()

        audio = capture.get_audio_data()
        if audio is None or len(audio) == 0:
            print("\n  [ERROR] No audio data captured.")
            return False

        # Calculate audio statistics
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio ** 2))
        duration = len(audio) / 16000

        print(f"\n  Duration: {duration:.1f}s")
        print(f"  Peak level: {peak:.4f} ({20 * np.log10(peak + 1e-10):.1f} dB)")
        print(f"  RMS level:  {rms:.4f} ({20 * np.log10(rms + 1e-10):.1f} dB)")

        if peak < 0.001:
            print("\n  [WARNING] Audio levels are very low (near silence).")
            print("  Is audio actually playing? Check your volume.")
            return False
        elif peak > 0.95:
            print("\n  [WARNING] Audio levels are very high (possible clipping).")
            print("  Consider lowering your system volume slightly.")
            return True
        else:
            print("\n  [OK] Audio levels look good.")
            return True


if __name__ == "__main__":
    import numpy as np  # needed for audio_levels test

    print("\n" + "#" * 60)
    print("  INTERPRETER-VERIFY-RU: Phase 1 Audio Test")
    print("#" * 60)

    # Step 1: List devices
    if not test_list_devices():
        sys.exit(1)

    # Step 2: Check audio levels (quick)
    print("\n  Starting audio level check in 3 seconds...")
    print("  Make sure audio is playing!")
    time.sleep(3)
    test_audio_levels()

    # Step 3: Full capture test
    print("\n  Starting 10-second capture test...")
    input("  Press ENTER when audio is playing, then wait 10 seconds... ")
    success = test_capture_audio(10)

    # Summary
    print("\n" + "#" * 60)
    if success:
        print("  PHASE 1 TEST: PASSED")
        print()
        print("  Next steps:")
        print("    1. Play test_capture.wav and verify the audio")
        print("    2. If it sounds good, commit to Git:")
        print("       git add .")
        print('       git commit -m "Phase 1: Audio capture working"')
        print("       git push")
        print("    3. Then we move to Phase 2: Transcription")
    else:
        print("  PHASE 1 TEST: NEEDS ATTENTION")
        print()
        print("  Check the errors above and try again.")
    print("#" * 60 + "\n")
