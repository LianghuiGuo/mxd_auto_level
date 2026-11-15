import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.engine import LieDetectorDiagnostics as diagnostics


class LieDetectorDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(diagnostics.wait_for_diagnostic_writes())
        self.frame = np.full((100, 120, 3), 25, np.uint8)
        self.record = {
            "frame": 2, "captured_at": 1.0, "status": 0,
            "center_x": 30, "center_y": 40, "bbox_x": 10, "bbox_y": 15,
            "bbox_width": 40, "bbox_height": 50, "confidence": 0.802,
            "sdk_ms": 47, "mouse_action": "move_requested",
        }
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.session_id = "test_session"
        self.session_output = self.output / "lie_sdk_sessions" / self.session_id
        path_patch = patch("src.engine.LieDetectorDiagnostics.Path", return_value=self.output)
        path_patch.start()
        self.addCleanup(path_patch.stop)
        self.addCleanup(diagnostics.wait_for_diagnostic_writes)

    def test_background_save_preserves_raw_input_and_draws_target(self):
        original = self.frame.copy()
        future = diagnostics.save_first_lock_snapshot(self.frame, self.record, self.session_id)
        self.assertIsNotNone(future)
        future.result(timeout=2.0)
        raw = cv2.imread(str(self.session_output / "lie_sdk_first_lock_frame.png"))
        overlay = cv2.imread(str(self.session_output / "lie_sdk_first_lock_overlay.png"))
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(self.frame, original)
        np.testing.assert_array_equal(overlay[50, 10], (0, 255, 0))
        np.testing.assert_array_equal(overlay[40, 30], (0, 0, 255))

    def test_slow_write_runs_off_thread_and_uses_frozen_copy(self):
        entered = threading.Event()
        release = threading.Event()
        copies = []
        threads = []

        def slow_write(_path, image, _params):
            threads.append(threading.current_thread().name)
            entered.set()
            release.wait(2.0)
            copies.append(image.copy())
            return True

        with patch("src.engine.LieDetectorDiagnostics.cv2.imwrite", side_effect=slow_write):
            future = diagnostics.save_input_snapshot(self.frame, self.session_id)
            try:
                self.assertTrue(entered.wait(1.0))
                self.assertFalse(future.done())
                self.frame[:] = 99
            finally:
                release.set()
            future.result(timeout=2.0)
        self.assertEqual(threads, ["LieDiagnosticWriter"])
        np.testing.assert_array_equal(copies[0], np.full((100, 120, 3), 25, np.uint8))

    def test_queue_is_bounded_and_never_blocks_tracking_submission(self):
        entered = threading.Event()
        release = threading.Event()
        jobs = []

        def slow_write(_path, _image, _params):
            entered.set()
            release.wait(2.0)
            return True

        with patch("src.engine.LieDetectorDiagnostics.cv2.imwrite", side_effect=slow_write):
            jobs.append(diagnostics.save_input_snapshot(self.frame, self.session_id))
            try:
                self.assertTrue(entered.wait(1.0))
                for _ in range(4):
                    jobs.append(diagnostics.save_input_snapshot(self.frame, self.session_id))
                self.assertIsNone(diagnostics.save_input_snapshot(self.frame, self.session_id))
                self.assertFalse(diagnostics.wait_for_diagnostic_writes(timeout=0.0))
            finally:
                release.set()
            for future in jobs:
                future.result(timeout=2.0)
        self.assertTrue(diagnostics.wait_for_diagnostic_writes())

    @patch("src.engine.LieDetectorDiagnostics.logger")
    def test_failed_background_write_is_logged_without_raising_in_caller(self, log):
        with patch("src.engine.LieDetectorDiagnostics.cv2.imwrite", return_value=False):
            future = diagnostics.save_input_snapshot(self.frame, self.session_id)
            with self.assertRaises(OSError):
                future.result(timeout=2.0)
        self.assertIn("Background diagnostic save failed", log.warning.call_args.args[0])

    def test_recent_trace_saves_csv_and_same_frame_annotations(self):
        second = dict(self.record, frame=3, captured_at=1.2, center_x=45)
        future = diagnostics.save_recent_trace(
            [self.record, second], [(2, self.frame.copy()), (3, self.frame.copy())], "close", self.session_id
        )
        future.result(timeout=2.0)
        with (self.session_output / "lie_sdk_recent_trace.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["frame"] for row in rows], ["2", "3"])
        self.assertEqual(rows[1]["center_x"], "45")
        self.assertEqual(rows[0]["mouse_action"], "move_requested")
        overlay = cv2.imread(str(self.session_output / "lie_sdk_recent_overlay.png"))
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape[1], 240)
        np.testing.assert_array_equal(overlay[40, 30], (0, 0, 255))
        np.testing.assert_array_equal(overlay[40, 120 + 45], (0, 0, 255))

    def test_session_saves_raw_samples_with_csv_links_and_metadata(self):
        records = [self.record, dict(self.record, frame=3, captured_at=1.1), dict(self.record, frame=7, captured_at=1.4)]
        images = [(2, self.frame.copy()), (7, self.frame.copy())]
        metadata = {"session_id": "test_session", "image_fps": 3.0, "truncated": False}
        future = diagnostics.save_session(records, images, metadata)
        future.result(timeout=2.0)
        output = self.output / "lie_sdk_sessions" / "test_session"
        with (output / "trace.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["frame"] for row in rows], ["2", "3", "7"])
        self.assertEqual([row["image_file"] for row in rows], ["frames/000002.png", "", "frames/000007.png"])
        raw = cv2.imread(str(output / "frames/000002.png"))
        np.testing.assert_array_equal(raw, self.frame)
        saved_metadata = json.loads((output / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_metadata, metadata)

    def test_session_submission_does_not_wait_for_slow_image_writes(self):
        entered = threading.Event()
        release = threading.Event()

        def slow_write(_path, _image, _params):
            entered.set()
            release.wait(2.0)
            return True

        metadata = {"session_id": "slow_session", "image_fps": 3.0, "truncated": False}
        with patch("src.engine.LieDetectorDiagnostics.cv2.imwrite", side_effect=slow_write):
            future = diagnostics.save_session([self.record], [(2, self.frame.copy())], metadata)
            try:
                self.assertTrue(entered.wait(1.0))
                self.assertFalse(future.done())
            finally:
                release.set()
            future.result(timeout=2.0)

    def test_session_with_memory_limit_still_saves_csv_without_images(self):
        metadata = {"session_id": "csv_session", "image_fps": 3.0, "truncated": True}
        diagnostics.save_session([self.record], [], metadata).result(timeout=2.0)
        output = self.output / "lie_sdk_sessions" / "csv_session"
        self.assertTrue((output / "trace.csv").is_file())
        self.assertEqual(list((output / "frames").iterdir()), [])

    def test_all_artifacts_share_session_directory_without_overwriting_other_sessions(self):
        for session_id, value in (("first_session", 25), ("second_session", 99)):
            with self.subTest(session_id=session_id):
                frame = np.full_like(self.frame, value)
                diagnostics.save_input_snapshot(frame, session_id).result(timeout=2.0)
                diagnostics.save_first_lock_snapshot(frame, self.record, session_id).result(timeout=2.0)
                diagnostics.save_recent_trace(
                    [self.record], [(2, frame)], "close", session_id,
                ).result(timeout=2.0)
                diagnostics.save_session(
                    [self.record], [(2, frame)],
                    {"session_id": session_id, "image_fps": 3.0, "truncated": False},
                ).result(timeout=2.0)
                output = self.output / "lie_sdk_sessions" / session_id
                self.assertEqual(
                    {path.name for path in output.iterdir()},
                    {
                        "lie_sdk_input_frame.png", "lie_sdk_first_lock_frame.png",
                        "lie_sdk_first_lock_overlay.png", "lie_sdk_recent_trace.csv",
                        "lie_sdk_recent_overlay.png", "frames", "trace.csv", "session.json",
                    },
                )
        self.assertEqual({path.name for path in self.output.iterdir()}, {"lie_sdk_sessions"})
        for session_id, value in (("first_session", 25), ("second_session", 99)):
            output = self.output / "lie_sdk_sessions" / session_id
            for filename in ("lie_sdk_input_frame.png", "lie_sdk_first_lock_frame.png", "frames/000002.png"):
                np.testing.assert_array_equal(cv2.imread(str(output / filename)), np.full_like(self.frame, value))

    def test_unlocked_recent_frame_has_no_fake_bbox(self):
        unlocked = dict(self.record, status=1, confidence=0.0, center_x=None, center_y=None)
        overlay = diagnostics._overlay(self.frame, unlocked)
        np.testing.assert_array_equal(overlay[40, 30], self.frame[40, 30])


if __name__ == "__main__":
    unittest.main()
