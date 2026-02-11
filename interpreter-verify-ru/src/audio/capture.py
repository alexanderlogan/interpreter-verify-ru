"""
Audio Capture Module (WASAPI Loopback)

Captures system audio output (what you hear from Zoom, Teams, etc.)
using Windows WASAPI loopback. Works with speakers, headphones, or
any audio output device.

This module is the foundation of the pipeline. It captures raw audio
and makes it available for transcription.
"""
import wave
import time
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    raise ImportError(
        "PyAudioWPatch is required for audio capture.\n"
        "Install it with: pip install PyAudioWPatch"
    )


@dataclass
class AudioDevice:
    """Represents an available audio output device."""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_loopback: bool
    is_default: bool


class AudioCapture:
    """
    Captures system audio via WASAPI loopback.

    Usage:
        capture = AudioCapture()
        capture.list_devices()          # See available devices
        capture.start()                 # Start capturing (default device)
        time.sleep(10)                  # Capture for 10 seconds
        capture.stop()                  # Stop capturing
        capture.save_wav("test.wav")    # Save to file for verification
    """

    def __init__(self, device_index: int | None = None, target_sample_rate: int = 16000):
        """
        Initialize audio capture.

        Args:
            device_index: Specific device to capture from. None = default output.
            target_sample_rate: Target sample rate for output (16000 for Whisper).
        """
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._is_capturing = False
        self._audio_buffer = []
        self._lock = threading.Lock()
        self._device_index = device_index
        self._target_sample_rate = target_sample_rate
        self._source_sample_rate = None
        self._source_channels = None

    def list_devices(self) -> list[AudioDevice]:
        """
        List all available audio output devices that support loopback capture.

        Returns:
            List of AudioDevice objects representing available devices.
        """
        devices = []

        # Get default WASAPI output device info
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output_index = wasapi_info["defaultOutputDevice"]
        except Exception:
            default_output_index = -1

        # Enumerate all devices, find loopback-capable ones
        for i in range(self._pa.get_device_count()):
            try:
                dev_info = self._pa.get_device_info_by_index(i)

                # WASAPI loopback devices have isLoopbackDevice flag
                is_loopback = dev_info.get("isLoopbackDevice", False)
                if not is_loopback:
                    continue

                devices.append(AudioDevice(
                    index=i,
                    name=dev_info["name"],
                    channels=dev_info["maxInputChannels"],
                    sample_rate=int(dev_info["defaultSampleRate"]),
                    is_loopback=True,
                    is_default=(dev_info.get("loopbackParentIndex", -1) == default_output_index)
                ))
            except Exception:
                continue

        return devices

    def get_default_loopback_device(self) -> AudioDevice | None:
        """Find the default output device's loopback counterpart."""
        devices = self.list_devices()
        # Prefer the default device
        for dev in devices:
            if dev.is_default:
                return dev
        # Fallback to first available loopback device
        return devices[0] if devices else None

    def start(self, device_index: int | None = None):
        """
        Start capturing audio.

        Args:
            device_index: Override device selection. None = use constructor value
                         or auto-detect default.
        """
        if self._is_capturing:
            print("  [!] Already capturing. Call stop() first.")
            return

        # Resolve device
        idx = device_index or self._device_index
        if idx is None:
            device = self.get_default_loopback_device()
            if device is None:
                raise RuntimeError(
                    "No loopback audio devices found. Make sure audio is "
                    "playing through speakers or headphones."
                )
            idx = device.index
            print(f"  [AUDIO] Auto-selected: {device.name}")
        else:
            device = None
            for d in self.list_devices():
                if d.index == idx:
                    device = d
                    break
            if device is None:
                raise RuntimeError(f"Device index {idx} not found or not a loopback device.")
            print(f"  [AUDIO] Selected: {device.name}")

        # Store source format
        self._source_sample_rate = device.sample_rate
        self._source_channels = device.channels

        # Clear buffer
        with self._lock:
            self._audio_buffer = []

        # Open WASAPI loopback stream
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=device.channels,
            rate=device.sample_rate,
            input=True,
            input_device_index=idx,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )

        self._is_capturing = True
        self._stream.start_stream()
        print(f"  [AUDIO] Capturing at {device.sample_rate}Hz, {device.channels}ch")

    def stop(self):
        """Stop capturing audio."""
        if not self._is_capturing:
            return

        self._is_capturing = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        duration = total_samples / (self._source_sample_rate * self._source_channels) if self._source_sample_rate else 0
        print(f"  [AUDIO] Stopped. Captured {duration:.1f} seconds of audio.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Called by PyAudio when new audio data is available."""
        if self._is_capturing:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            with self._lock:
                self._audio_buffer.append(audio_data.copy())
        return (None, pyaudio.paContinue)

    def get_audio_data(self) -> np.ndarray | None:
        """
        Get all captured audio as a numpy array, resampled to target rate
        and converted to mono.

        Returns:
            Numpy array of float32 audio samples at target_sample_rate, mono.
            None if no audio captured.
        """
        with self._lock:
            if not self._audio_buffer:
                return None
            raw = np.concatenate(self._audio_buffer)

        # Convert to mono if stereo
        if self._source_channels and self._source_channels > 1:
            raw = raw.reshape(-1, self._source_channels).mean(axis=1)

        # Resample if needed
        if self._source_sample_rate and self._source_sample_rate != self._target_sample_rate:
            try:
                import soxr
                raw = soxr.resample(
                    raw,
                    self._source_sample_rate,
                    self._target_sample_rate
                )
            except ImportError:
                # Fallback: simple linear interpolation (lower quality)
                ratio = self._target_sample_rate / self._source_sample_rate
                new_length = int(len(raw) * ratio)
                indices = np.linspace(0, len(raw) - 1, new_length)
                raw = np.interp(indices, np.arange(len(raw)), raw)
                print("  [!] soxr not installed. Using basic resampling.")
                print("      Install soxr for better quality: pip install soxr")

        return raw.astype(np.float32)

    def get_audio_chunk(self, clear: bool = True) -> np.ndarray | None:
        """
        Get current audio buffer contents and optionally clear it.
        Used in the live pipeline to grab audio segments for transcription.

        Args:
            clear: If True, clears the buffer after reading (default: True).

        Returns:
            Numpy array of float32 audio at target_sample_rate, mono.
            None if buffer is empty.
        """
        with self._lock:
            if not self._audio_buffer:
                return None
            raw = np.concatenate(self._audio_buffer)
            if clear:
                self._audio_buffer = []

        # Convert to mono
        if self._source_channels and self._source_channels > 1:
            raw = raw.reshape(-1, self._source_channels).mean(axis=1)

        # Resample
        if self._source_sample_rate and self._source_sample_rate != self._target_sample_rate:
            try:
                import soxr
                raw = soxr.resample(raw, self._source_sample_rate, self._target_sample_rate)
            except ImportError:
                ratio = self._target_sample_rate / self._source_sample_rate
                new_length = int(len(raw) * ratio)
                indices = np.linspace(0, len(raw) - 1, new_length)
                raw = np.interp(indices, np.arange(len(raw)), raw)

        return raw.astype(np.float32)

    def save_wav(self, filepath: str | Path, audio_data: np.ndarray | None = None):
        """
        Save captured audio to a WAV file for verification.

        Args:
            filepath: Path to save the WAV file.
            audio_data: Specific audio data to save. None = save entire buffer.
        """
        if audio_data is None:
            audio_data = self.get_audio_data()

        if audio_data is None or len(audio_data) == 0:
            print("  [!] No audio data to save.")
            return

        filepath = Path(filepath)

        # Convert float32 [-1.0, 1.0] to int16 for WAV
        audio_int16 = (audio_data * 32767).astype(np.int16)

        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._target_sample_rate)
            wf.writeframes(audio_int16.tobytes())

        duration = len(audio_data) / self._target_sample_rate
        size_kb = filepath.stat().st_size / 1024
        print(f"  [AUDIO] Saved {duration:.1f}s ({size_kb:.0f} KB) to {filepath}")

    def close(self):
        """Release all audio resources."""
        self.stop()
        if self._pa:
            self._pa.terminate()
            self._pa = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def is_capturing(self) -> bool:
        return self._is_capturing

    @property
    def buffer_duration(self) -> float:
        """Current buffer duration in seconds."""
        with self._lock:
            total = sum(len(c) for c in self._audio_buffer)
        if self._source_sample_rate and self._source_channels:
            return total / (self._source_sample_rate * self._source_channels)
        return 0.0
