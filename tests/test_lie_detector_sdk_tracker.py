import ctypes
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.engine.LieDetectorSdkTracker import LieDetectorSdkTracker


class LieDetectorSdkTrackerTest(unittest.TestCase):
    def setUp(self):
        functions = (
            "TT_Version", "TT_SetForceCpu", "TT_MachineCode", "TT_AuthError",
            "TTx_Create", "TTx_LastCreateError", "TTx_Destroy", "TTx_Reset",
            "TTx_Detect", "TTx_LastError", "TTx_LastMs", "TTx_Backend",
        )
        self.lib = SimpleNamespace(**{name: Mock() for name in functions})
        self.handle = 0x123456789ABC
        self.lib.TT_Version.return_value = 220
        self.lib.TTx_Create.return_value = self.handle
        self.lib.TTx_Backend.return_value = b"DirectML"
        self.lib.TTx_Reset.return_value = 0
        self.lib.TTx_LastMs.return_value = 47
        self.lib.TTx_LastCreateError.return_value = b""
        self.lib.TT_AuthError.return_value = b""
        self.lib.TT_MachineCode.return_value = b"AAAA-BBBB-CCCC-DDDD"
        self.lib.TTx_Detect.side_effect = self._detect
        self.status = 0
        self.output = [0, 30, 40, 10, 15, 40, 50, 802]
        self.last_input = None
        self.search_directory = Mock()
        image_writer = patch("src.engine.LieDetectorSdkTracker.cv2.imwrite", return_value=True)
        image_writer.start()
        self.addCleanup(image_writer.stop)
        patches = (
            patch("src.engine.LieDetectorSdkTracker.sys.platform", "win32"),
            patch("src.engine.LieDetectorSdkTracker.Path.is_file", return_value=True),
            patch(
                "src.engine.LieDetectorSdkTracker.os.add_dll_directory",
                return_value=self.search_directory,
                create=True,
            ),
            patch(
                "src.engine.LieDetectorSdkTracker.ctypes.WinDLL",
                return_value=self.lib,
                create=True,
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _detect(self, handle, pixels, width, height, stride, fmt, out):
        self.last_input = (handle, width, height, stride, fmt)
        self.input_bytes = ctypes.string_at(pixels, stride * height)
        for index, value in enumerate(self.output):
            out[index] = value
        return self.status

    def _tracker(self, **kwargs):
        tracker = LieDetectorSdkTracker(**kwargs)
        self.addCleanup(tracker.close)
        return tracker

    def test_declares_64_bit_handles_and_keeps_search_directory_alive(self):
        tracker = self._tracker()
        self.assertIs(self.lib.TTx_Create.restype, ctypes.c_void_p)
        self.assertIs(self.lib.TTx_Destroy.argtypes[0], ctypes.c_void_p)
        self.assertIs(self.lib.TTx_Detect.argtypes[0], ctypes.c_void_p)
        self.assertIs(self.lib.TTx_Detect.restype, ctypes.c_int)
        self.lib.TTx_Create.assert_called_once_with(None)
        self.lib.TT_SetForceCpu.assert_called_once_with(0)
        self.assertEqual(tracker.backend, "DirectML")
        self.search_directory.close.assert_not_called()

    def test_locked_result_preserves_roi_coordinates_and_confidence(self):
        tracker = self._tracker()
        result = tracker.update(np.zeros((100, 120, 3), np.uint8), 1.0)
        self.assertEqual(self.last_input, (self.handle, 120, 100, 360, 0))
        self.assertTrue(result.acquired)
        self.assertTrue(result.actionable)
        self.assertEqual(result.center, (30.0, 40.0))
        self.assertAlmostEqual(result.confidence, 0.802)
        self.assertIsNone(result.target_id)
        self.assertEqual(result.tracks, [])
        self.assertEqual(tracker.last_ms, 47)

    def test_non_contiguous_input_is_copied_with_actual_stride(self):
        tracker = self._tracker()
        frame = np.arange(100 * 150 * 3, dtype=np.uint8).reshape(100, 150, 3)
        roi = frame[:, 10:130, :][:, ::-1, :]
        self.assertFalse(roi.flags.c_contiguous)
        tracker.update(roi)
        self.assertEqual(self.last_input, (self.handle, 120, 100, 360, 0))
        self.assertEqual(self.input_bytes, roi.tobytes())

    def test_unlocked_result_does_not_reuse_output_coordinates(self):
        tracker = self._tracker()
        tracker.update(np.zeros((100, 120, 3), np.uint8))
        self.status = 1
        result = tracker.update(np.zeros((100, 120, 3), np.uint8))
        self.assertFalse(result.acquired)
        self.assertFalse(result.actionable)
        self.assertIsNone(result.center)
        self.assertEqual(result.confidence, 0.0)

    def test_out_of_frame_center_is_not_actionable(self):
        tracker = self._tracker()
        self.output[1] = 120
        result = tracker.update(np.zeros((100, 120, 3), np.uint8))
        self.assertFalse(result.actionable)

    def test_sdk_error_is_decoded_and_raised(self):
        tracker = self._tracker()
        self.status = 2
        self.lib.TTx_LastError.return_value = "识别失败".encode("gbk")
        with self.assertRaisesRegex(RuntimeError, "识别失败"):
            tracker.update(np.zeros((100, 120, 3), np.uint8))

    def test_invalid_status_is_rejected(self):
        tracker = self._tracker()
        self.status = 9
        with self.assertRaisesRegex(RuntimeError, "invalid status 9"):
            tracker.update(np.zeros((100, 120, 3), np.uint8))

    def test_invalid_image_never_reaches_native_code(self):
        tracker = self._tracker()
        frames = (
            None,
            np.zeros((0, 120, 3), np.uint8),
            np.zeros((100, 120), np.uint8),
            np.zeros((100, 120, 4), np.uint8),
            np.zeros((100, 120, 3), np.float32),
        )
        for frame in frames:
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaises(ValueError):
                    tracker.update(frame)
        self.lib.TTx_Detect.assert_not_called()

    def test_dimension_change_resets_temporal_state(self):
        tracker = self._tracker()
        tracker.update(np.zeros((100, 120, 3), np.uint8))
        tracker.update(np.zeros((110, 120, 3), np.uint8))
        self.lib.TTx_Reset.assert_called_once_with(self.handle)
        tracker.reset()
        self.assertEqual(self.lib.TTx_Reset.call_count, 2)
        self.assertEqual(tracker.last_ms, 0)

    def test_close_is_idempotent_and_releases_native_handle_first(self):
        tracker = self._tracker()
        order = []
        self.lib.TTx_Destroy.side_effect = lambda _handle: order.append("destroy")
        self.search_directory.close.side_effect = lambda: order.append("directory")
        tracker.close()
        tracker.close()
        self.assertEqual(order, ["destroy", "directory"])
        self.lib.TTx_Destroy.assert_called_once_with(self.handle)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        tracker.reset()
        self.lib.TTx_Reset.assert_not_called()

    def test_authorization_failure_reports_machine_code_and_releases_directory(self):
        self.lib.TTx_Create.return_value = None
        self.lib.TTx_LastCreateError.return_value = "机器码未登记".encode("gbk")
        with self.assertRaisesRegex(RuntimeError, "机器码未登记.*AAAA-BBBB-CCCC-DDDD"):
            LieDetectorSdkTracker()
        self.search_directory.close.assert_called_once()
        self.lib.TTx_Destroy.assert_not_called()

    def test_unsupported_version_is_rejected_before_model_creation(self):
        self.lib.TT_Version.return_value = 210
        with self.assertRaisesRegex(RuntimeError, "got 210"):
            LieDetectorSdkTracker()
        self.lib.TTx_Create.assert_not_called()
        self.search_directory.close.assert_called_once()

    def test_missing_file_is_reported_before_loading(self):
        with patch("src.engine.LieDetectorSdkTracker.Path.is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "trtyoloLie.dll"):
                LieDetectorSdkTracker()
        self.lib.TTx_Create.assert_not_called()

    def test_cpu_selection_happens_before_creation(self):
        order = []
        self.lib.TT_SetForceCpu.side_effect = lambda value: order.append(("cpu", value))
        self.lib.TTx_Create.side_effect = lambda _model: order.append(("create",)) or self.handle
        self._tracker(force_cpu=True)
        self.assertEqual(order, [("cpu", 1), ("create",)])

    def test_non_windows_environment_does_not_load_dll(self):
        with patch("src.engine.LieDetectorSdkTracker.sys.platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Windows"):
                LieDetectorSdkTracker()
        self.lib.TTx_Create.assert_not_called()

    def test_32_bit_python_does_not_load_dll(self):
        with patch("src.engine.LieDetectorSdkTracker.ctypes.sizeof", return_value=4):
            with self.assertRaisesRegex(RuntimeError, "64-bit"):
                LieDetectorSdkTracker()
        self.lib.TTx_Create.assert_not_called()

    def test_relative_directory_is_resolved_from_project_root(self):
        with patch(
            "src.engine.LieDetectorSdkTracker.os.add_dll_directory",
            return_value=self.search_directory,
            create=True,
        ) as add_directory:
            self._tracker(sdk_directory="external_sdk/bin64")
        expected = Path(__file__).resolve().parents[1] / "external_sdk" / "bin64"
        add_directory.assert_called_once_with(str(expected))

    @patch("src.engine.LieDetectorSdkTracker.logger")
    def test_first_lock_and_relock_are_logged_without_per_frame_spam(self, log):
        tracker = self._tracker()
        log.reset_mock()
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=10.0):
            self.status = 1
            tracker.update(np.zeros((100, 120, 3), np.uint8))
            for _ in range(20):
                tracker.update(np.zeros((100, 120, 3), np.uint8))
            self.status = 0
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=11.1):
            self.status = 1
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=12.2):
            self.status = 0
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        messages = [call.args[0] for call in log.info.call_args_list]
        self.assertEqual(sum("First input" in text for text in messages), 1)
        transitions = [text for text in messages if "State " in text]
        self.assertEqual(len(transitions), 4)
        self.assertIn("INITIAL->NOTREADY", transitions[0])
        self.assertIn("NOTREADY->LOCKED", transitions[1])
        self.assertIn("frame=22", transitions[1])
        self.assertIn("out=[0, 30, 40", transitions[1])
        self.assertIn("LOCKED->NOTREADY", transitions[2])
        self.assertIn("NOTREADY->LOCKED", transitions[3])
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=12.2):
            tracker.close()

    @patch("src.engine.LieDetectorSdkTracker.logger")
    def test_summary_is_emitted_every_five_seconds_and_on_reset(self, log):
        tracker = self._tracker()
        log.reset_mock()
        self.status = 1
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=10.0):
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=14.9):
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        self.assertFalse(any("Summary" in c.args[0] for c in log.info.call_args_list))
        with patch("src.engine.LieDetectorSdkTracker.time.monotonic", return_value=15.0):
            tracker.update(np.zeros((100, 120, 3), np.uint8))
            tracker.update(np.zeros((100, 120, 3), np.uint8))
            tracker.reset()
        messages = [call.args[0] for call in log.info.call_args_list]
        summaries = [text for text in messages if "Summary" in text]
        self.assertEqual(len(summaries), 2)
        self.assertIn("reason=interval", summaries[0])
        self.assertIn("NOTREADY=3", summaries[0])
        self.assertIn("notready_streak=3", summaries[0])
        self.assertIn("avg_sdk_ms=47.0", summaries[0])
        self.assertIn("reason=reset", summaries[1])
        self.assertEqual(tracker._status_counts, [0, 0, 0])

    @patch("src.engine.LieDetectorSdkTracker.logger")
    def test_sdk_error_log_and_exception_include_input_context(self, log):
        tracker = self._tracker()
        self.status = 2
        self.lib.TTx_LastError.return_value = "识别失败".encode("gbk")
        with self.assertRaisesRegex(RuntimeError, "status=2.*roi_wh=\\(120,100\\).*识别失败"):
            tracker.update(np.zeros((100, 120, 3), np.uint8))
        self.assertEqual(tracker._status_counts, [0, 0, 1])
        self.assertIn("INITIAL->ERROR", log.warning.call_args.args[0])

    @patch("src.engine.LieDetectorSdkTracker.logger")
    def test_out_of_bounds_is_explicit_in_diagnostics(self, log):
        tracker = self._tracker()
        self.output[1] = 120
        result = tracker.update(np.zeros((100, 120, 3), np.uint8))
        self.assertFalse(result.actionable)
        self.assertIn("OUT_OF_BOUNDS", log.warning.call_args.args[0])

    @patch("src.engine.LieDetectorSdkTracker.cv2.imwrite", return_value=True)
    def test_first_lock_saves_exact_frame_and_bbox_without_mutating_input(self, write):
        tracker = self._tracker()
        frame = np.full((100, 120, 3), 25, dtype=np.uint8)
        original = frame.copy()
        self.status = 1
        tracker.update(frame, 1.0)
        write.assert_not_called()
        self.status = 0
        result = tracker.update(frame, 2.0)
        self.assertTrue(result.actionable)
        self.assertEqual(write.call_count, 2)
        raw_call, overlay_call = write.call_args_list
        self.assertEqual(Path(raw_call.args[0]).name, "lie_sdk_first_lock_frame.png")
        self.assertEqual(Path(overlay_call.args[0]).name, "lie_sdk_first_lock_overlay.png")
        np.testing.assert_array_equal(raw_call.args[1], original)
        np.testing.assert_array_equal(frame, original)
        overlay = overlay_call.args[1]
        self.assertIsNot(overlay, frame)
        np.testing.assert_array_equal(overlay[50, 10], (0, 255, 0))
        np.testing.assert_array_equal(overlay[40, 30], (0, 0, 255))
        tracker.update(frame, 3.0)
        self.assertEqual(write.call_count, 2)

    @patch("src.engine.LieDetectorSdkTracker.cv2.imwrite", return_value=True)
    def test_reset_allows_next_challenge_to_overwrite_first_lock_snapshots(self, write):
        tracker = self._tracker()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        tracker.update(frame)
        self.status = 1
        tracker.update(frame)
        self.status = 0
        tracker.update(frame)
        self.assertEqual(write.call_count, 2)
        first_paths = [call.args[0] for call in write.call_args_list]
        tracker.reset()
        tracker.update(frame)
        self.assertEqual(write.call_count, 4)
        self.assertEqual(first_paths, [call.args[0] for call in write.call_args_list[2:]])

    @patch("src.engine.LieDetectorSdkTracker.cv2.imwrite", return_value=False)
    @patch("src.engine.LieDetectorSdkTracker.logger")
    def test_snapshot_failure_preserves_tracking_and_is_not_retried_each_frame(self, log, write):
        tracker = self._tracker()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        result = tracker.update(frame)
        self.assertTrue(result.actionable)
        self.assertEqual(result.center, (30.0, 40.0))
        self.assertIn("Could not save first-lock snapshots", log.warning.call_args.args[0])
        tracker.update(frame)
        self.assertEqual(write.call_count, 1)


if __name__ == "__main__":
    unittest.main()
