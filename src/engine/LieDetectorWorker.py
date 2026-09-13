"""Dedicated high-rate worker for the online lie-detector pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable, Generic, Optional, TypeVar

import numpy as np

from src.utils.logger import logger


T = TypeVar("T")


@dataclass(frozen=True)
class LieDetectorWorkerSnapshot(Generic[T]):
    """Latest completed result together with its source-frame metadata."""

    sequence: int
    captured_at: float
    processed_at: float
    effective_fps: float
    value: T


FrameSource = Callable[[Optional[int]], Optional[tuple[int, np.ndarray, float]]]
FrameProcessor = Callable[[np.ndarray, float], T]
ResultCallback = Callable[[LieDetectorWorkerSnapshot[T]], None]


class LieDetectorWorker(Generic[T]):
    """Process each new capture frame outside the slower gameplay loop.

    The worker always asks the capture buffer for its newest frame. If model
    inference falls behind, old frames are naturally dropped instead of
    building latency in an unbounded queue. Duplicate capture frames are never
    sent to the temporal tracker.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        processor: FrameProcessor[T],
        *,
        target_fps: float = 33.0,
        on_result: Optional[ResultCallback[T]] = None,
    ) -> None:
        self.frame_source = frame_source
        self.processor = processor
        self.target_fps = max(1.0, float(target_fps))
        self.on_result = on_result
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[LieDetectorWorkerSnapshot[T]] = None
        self._completion_times: deque[float] = deque(maxlen=34)
        self._last_slow_log_at = float("-inf")
        self._last_fps_log_at = float("-inf")

    @property
    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        if self.is_alive:
            if self._stop_event.is_set():
                self._thread.join(timeout=2.0)
            if self.is_alive:
                raise RuntimeError(
                    "previous lie-detector worker is still shutting down"
                )
        self._stop_event.clear()
        self._completion_times.clear()
        self._last_slow_log_at = float("-inf")
        self._last_fps_log_at = float("-inf")
        self._thread = threading.Thread(
            target=self._run,
            name="LieDetectorWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        if thread is not None and thread.is_alive():
            logger.warning(
                "[Lie Detector] Dedicated worker did not stop before timeout; "
                "restart will wait for the in-flight frame to finish."
            )
        else:
            self._thread = None

    def latest(self) -> Optional[LieDetectorWorkerSnapshot[T]]:
        with self._lock:
            return self._latest

    def _effective_fps(self, now: float) -> float:
        self._completion_times.append(now)
        if len(self._completion_times) < 2:
            return 0.0
        elapsed = self._completion_times[-1] - self._completion_times[0]
        return (len(self._completion_times) - 1) / max(elapsed, 1e-6)

    def _run(self) -> None:
        period = 1.0 / self.target_fps
        last_sequence: Optional[int] = None
        logger.info(
            f"[Lie Detector] Dedicated worker started; target={self.target_fps:.1f} FPS."
        )
        while not self._stop_event.is_set():
            iteration_started = time.monotonic()
            processed_frame = False
            try:
                source = self.frame_source(last_sequence)
                if source is not None:
                    sequence, frame, captured_at = source
                    if sequence != last_sequence:
                        processed_frame = True
                        value = self.processor(frame, captured_at)
                        completed_at = time.monotonic()
                        effective_fps = self._effective_fps(completed_at)
                        snapshot = LieDetectorWorkerSnapshot(
                            sequence=int(sequence),
                            captured_at=float(captured_at),
                            processed_at=completed_at,
                            effective_fps=effective_fps,
                            value=value,
                        )
                        with self._lock:
                            self._latest = snapshot
                        last_sequence = int(sequence)
                        if (
                            not self._stop_event.is_set()
                            and self.on_result is not None
                        ):
                            self.on_result(snapshot)
                        if (
                            getattr(value, "engaged", False)
                            and effective_fps > 0.0
                            and completed_at - self._last_fps_log_at >= 5.0
                        ):
                            self._last_fps_log_at = completed_at
                            logger.info(
                                "[Lie Detector] Worker throughput: "
                                f"{effective_fps:.1f}/{self.target_fps:.1f} FPS."
                            )
            except Exception as exc:
                # One corrupt frame or transient window failure must not kill
                # the detector for the rest of the challenge.
                logger.error(f"[Lie Detector] Worker frame failed: {exc}")

            elapsed = time.monotonic() - iteration_started
            if elapsed > period * 1.35:
                now = time.monotonic()
                if now - self._last_slow_log_at >= 5.0:
                    self._last_slow_log_at = now
                    logger.warning(
                        "[Lie Detector] Worker is below its target: "
                        f"last frame took {elapsed * 1000.0:.1f} ms."
                    )
            wait_seconds = (
                max(0.0, period - elapsed)
                if processed_frame
                else min(period, 0.005)
            )
            self._stop_event.wait(wait_seconds)
        logger.info("[Lie Detector] Dedicated worker stopped.")
