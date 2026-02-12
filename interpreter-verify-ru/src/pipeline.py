"""
Threaded Translation Pipeline

Runs audio capture, transcription, and translation on separate threads
so that no speech is dropped. While Ollama translates segment 1,
Whisper is already transcribing segment 2, and audio capture is
recording segment 3.

Architecture:
  [Audio Capture Thread] -> audio_queue
  [Whisper Thread]       <- audio_queue -> transcript_queue
  [Ollama Thread]        <- transcript_queue -> display_queue
  [Display Thread]       <- display_queue -> screen output
"""
import time
import threading
import queue
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from src.audio.capture import AudioCapture
from src.transcription.whisper_engine import WhisperEngine, TranscriptSegment
from src.translation.ollama_engine import OllamaEngine, TranslationResult


@dataclass
class PipelineItem:
    """A single item flowing through the pipeline."""
    transcript: TranscriptSegment
    translation: TranslationResult | None = None
    pharma_flags: list = field(default_factory=list)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class Pipeline:
    """
    Threaded translation pipeline with continuous audio capture.

    Usage:
        def on_result(item: PipelineItem):
            print(f"[{item.transcript.language}] {item.transcript.text}")
            if item.translation:
                print(f"  -> {item.translation.translated_text}")

        pipeline = Pipeline(on_result=on_result)
        pipeline.start()
        # ... runs until stopped ...
        pipeline.stop()
    """

    def __init__(
        self,
        on_result: Callable[[PipelineItem], None] | None = None,
        segment_duration: float = 4.0,
        device_index: int | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            on_result: Callback fired for each completed pipeline item.
            segment_duration: Seconds of audio per chunk sent to Whisper.
            device_index: Audio device index. None = auto-detect.
        """
        self._on_result = on_result or self._default_display
        self._segment_duration = segment_duration
        self._device_index = device_index

        # Queues connecting the threads
        self._audio_queue = queue.Queue(maxsize=20)
        self._transcript_queue = queue.Queue(maxsize=20)

        # Thread control
        self._running = False
        self._threads = []

        # Engines (initialized on start)
        self._capture = None
        self._whisper = None
        self._ollama = None

        # Statistics
        self._stats = {
            "chunks_captured": 0,
            "chunks_transcribed": 0,
            "chunks_translated": 0,
            "chunks_dropped": 0,
            "total_whisper_time": 0.0,
            "total_translate_time": 0.0,
        }
        self._stats_lock = threading.Lock()

    def start(self):
        """Start the pipeline. All threads begin processing."""
        if self._running:
            print("  [PIPELINE] Already running.")
            return

        print("  [PIPELINE] Initializing engines...")

        # Initialize engines
        self._whisper = WhisperEngine()
        self._ollama = OllamaEngine()
        self._capture = AudioCapture(
            device_index=self._device_index,
            target_sample_rate=16000,
        )

        # Warm up Ollama
        self._ollama.warm_up()

        # Start audio capture
        self._capture.start()

        self._running = True

        # Launch threads
        self._threads = [
            threading.Thread(
                target=self._audio_producer,
                name="AudioProducer",
                daemon=True,
            ),
            threading.Thread(
                target=self._whisper_worker,
                name="WhisperWorker",
                daemon=True,
            ),
            threading.Thread(
                target=self._ollama_worker,
                name="OllamaWorker",
                daemon=True,
            ),
        ]

        for t in self._threads:
            t.start()

        print(f"  [PIPELINE] Running. Segment duration: {self._segment_duration}s")
        print(f"  [PIPELINE] Press Ctrl+C to stop.\n")

    def stop(self):
        """Stop the pipeline gracefully."""
        if not self._running:
            return

        self._running = False

        # Stop audio capture
        if self._capture:
            self._capture.stop()
            self._capture.close()

        # Drain queues to unblock threads
        self._drain_queue(self._audio_queue)
        self._drain_queue(self._transcript_queue)

        # Wait for threads to finish
        for t in self._threads:
            t.join(timeout=5)

        self._threads = []
        self._print_stats()

    def _audio_producer(self):
        """
        Continuously grabs audio chunks and puts them on the queue.
        Never blocks on Whisper or Ollama, so no audio is lost.
        """
        while self._running:
            time.sleep(self._segment_duration)

            if not self._running:
                break

            audio = self._capture.get_audio_chunk(clear=True)
            if audio is None or len(audio) == 0:
                continue

            # Skip silence
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.001:
                continue

            with self._stats_lock:
                self._stats["chunks_captured"] += 1

            # Put on queue. If Whisper is backed up, drop oldest chunk.
            try:
                self._audio_queue.put_nowait(audio)
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()  # Drop oldest
                    self._audio_queue.put_nowait(audio)
                    with self._stats_lock:
                        self._stats["chunks_dropped"] += 1
                except queue.Empty:
                    pass

    def _whisper_worker(self):
        """
        Takes audio chunks from the queue and transcribes them.
        Runs independently of the Ollama thread.
        """
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=1)
            except queue.Empty:
                continue

            segments = self._whisper.transcribe(audio)

            with self._stats_lock:
                self._stats["chunks_transcribed"] += 1
                if segments:
                    self._stats["total_whisper_time"] += segments[0].transcription_time

            for seg in segments:
                try:
                    self._transcript_queue.put_nowait(seg)
                except queue.Full:
                    try:
                        self._transcript_queue.get_nowait()
                        self._transcript_queue.put_nowait(seg)
                    except queue.Empty:
                        pass

    def _ollama_worker(self):
        """
        Takes transcribed segments and translates them.
        Runs independently of Whisper, so transcription is never blocked.
        """
        while self._running:
            try:
                seg = self._transcript_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Translate
            result = self._ollama.translate(seg.text, seg.language)

            with self._stats_lock:
                self._stats["chunks_translated"] += 1
                if result:
                    self._stats["total_translate_time"] += result.translation_time

            # Deliver result
            item = PipelineItem(
                transcript=seg,
                translation=result,
            )
            self._on_result(item)

    def _default_display(self, item: PipelineItem):
        """Default callback: print to console."""
        lang = "RU" if item.transcript.is_russian else "EN"
        print(f"\n  [{lang}] {item.transcript.text}")

        if item.translation:
            target = "EN" if item.transcript.is_russian else "RU"
            print(f"  [{target}] {item.translation.translated_text}")
            total = item.transcript.transcription_time + item.translation.translation_time
            print(f"       ({item.transcript.transcription_time:.1f}s + "
                  f"{item.translation.translation_time:.1f}s = {total:.1f}s)")

    def _drain_queue(self, q):
        """Empty a queue to unblock waiting threads."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _print_stats(self):
        """Print pipeline performance statistics."""
        with self._stats_lock:
            s = self._stats.copy()

        print(f"\n  " + "=" * 50)
        print(f"  Pipeline Statistics:")
        print(f"    Audio chunks captured:    {s['chunks_captured']}")
        print(f"    Chunks transcribed:       {s['chunks_transcribed']}")
        print(f"    Chunks translated:        {s['chunks_translated']}")
        print(f"    Chunks dropped (backlog): {s['chunks_dropped']}")

        if s["chunks_transcribed"] > 0:
            avg_w = s["total_whisper_time"] / s["chunks_transcribed"]
            print(f"    Avg Whisper time:         {avg_w:.1f}s")
        if s["chunks_translated"] > 0:
            avg_t = s["total_translate_time"] / s["chunks_translated"]
            print(f"    Avg translation time:     {avg_t:.1f}s")

        coverage = (s["chunks_transcribed"] / s["chunks_captured"] * 100
                    if s["chunks_captured"] > 0 else 0)
        print(f"    Coverage:                 {coverage:.0f}%")
        print(f"  " + "=" * 50)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return self._stats.copy()
