import threading
import time
import unittest

import numpy as np

from src.engine.LieDetectorWorker import LieDetectorWorker


class _LatestFrameSource:
    def __init__(self):
        self.lock = threading.Lock()
        self.sequence = 0
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.captured_at = time.monotonic()

    def publish(self, sequence):
        with self.lock:
            self.sequence = sequence
            self.captured_at = time.monotonic()

    def __call__(self, after_sequence=None):
        with self.lock:
            if after_sequence == self.sequence:
                return None
            return self.sequence, self.frame.copy(), self.captured_at


class LieDetectorWorkerTest(unittest.TestCase):
    def test_processes_each_capture_sequence_only_once(self):
        source = _LatestFrameSource()
        processed = []
        got_second = threading.Event()

        def process(_frame, _timestamp):
            processed.append(source.sequence)
            if source.sequence == 2:
                got_second.set()
            return source.sequence

        worker = LieDetectorWorker(source, process, target_fps=120)
        worker.start()
        try:
            deadline = time.monotonic() + 0.5
            while worker.latest() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            source.publish(1)
            time.sleep(0.04)
            source.publish(2)
            self.assertTrue(got_second.wait(0.5))
        finally:
            worker.stop()

        self.assertEqual(processed.count(0), 1)
        self.assertEqual(processed.count(1), 1)
        self.assertEqual(processed.count(2), 1)
        self.assertEqual(worker.latest().sequence, 2)
        self.assertFalse(worker.is_alive)

    def test_callback_receives_the_published_snapshot(self):
        source = _LatestFrameSource()
        callbacks = []
        called = threading.Event()

        def callback(snapshot):
            callbacks.append(snapshot)
            called.set()

        worker = LieDetectorWorker(
            source, lambda _frame, _timestamp: "tracked",
            target_fps=120, on_result=callback,
        )
        worker.start()
        try:
            self.assertTrue(called.wait(0.5))
        finally:
            worker.stop()

        self.assertEqual(callbacks[0].sequence, 0)
        self.assertEqual(callbacks[0].value, "tracked")
        self.assertGreaterEqual(callbacks[0].processed_at, callbacks[0].captured_at)

    def test_stop_suppresses_callback_for_an_inflight_frame(self):
        source = _LatestFrameSource()
        processing = threading.Event()
        release = threading.Event()
        callbacks = []

        def process(_frame, _timestamp):
            processing.set()
            release.wait(0.5)
            return "late"

        worker = LieDetectorWorker(
            source, process, target_fps=120, on_result=callbacks.append
        )
        worker.start()
        self.assertTrue(processing.wait(0.5))
        worker._stop_event.set()
        release.set()
        worker.stop()

        self.assertEqual(callbacks, [])

    def test_cleanup_runs_once_after_stop(self):
        source = _LatestFrameSource()
        processed = threading.Event()
        cleanup = []

        def process(_frame, _timestamp):
            processed.set()
            return "tracked"

        worker = LieDetectorWorker(
            source, process, on_stop=lambda: cleanup.append("closed")
        )
        worker.start()
        try:
            self.assertTrue(processed.wait(0.5))
        finally:
            worker.stop()
        worker.stop()
        self.assertEqual(cleanup, ["closed"])

    def test_timed_out_stop_defers_cleanup_until_inflight_frame_finishes(self):
        source = _LatestFrameSource()
        processing = threading.Event()
        release = threading.Event()
        cleanup = []
        callbacks = []

        def process(_frame, _timestamp):
            processing.set()
            release.wait(2.0)
            return "tracked"

        worker = LieDetectorWorker(
            source, process, on_result=callbacks.append,
            on_stop=lambda: cleanup.append("closed"),
        )
        worker.start()
        try:
            self.assertTrue(processing.wait(0.5))
            worker.stop(timeout=0.0)
            self.assertTrue(worker.is_alive)
            self.assertEqual(cleanup, [])
        finally:
            release.set()
            worker.stop()
        self.assertEqual(cleanup, ["closed"])
        self.assertEqual(callbacks, [])


if __name__ == "__main__":
    unittest.main()
