import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.engine.LieDetectorRuntime import (
    PHASE_CONFIRM,
    PHASE_IDLE,
    PHASE_PANEL,
    LieDetectorRuntime,
    LiePanelGate,
    find_success_confirm,
)

TITLE_TEMPLATE = Path("misc/lie_detector_title_cn.png")


def _noise_frame(width=1280, height=720, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def _terrain_frame(width=1280, height=720, seed=3):
    """Sandy background approximating the map behind the dialog."""
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), (120, 170, 210), dtype=np.uint8)
    noise = rng.integers(-18, 18, size=(height, width, 3))
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _success_dialog_frame(width=1280, height=720, dialog_width=257):
    """Synthesize the NPC success dialog with the real button geometry."""
    frame = _terrain_frame(width, height)
    x = (width - dialog_width) // 2
    y = 298
    dialog_height = int(round(0.366 * dialog_width))
    # Blue message box.
    cv2.rectangle(
        frame,
        (x, y),
        (x + dialog_width, y + dialog_height),
        (233, 204, 167),
        -1,
    )
    # Gray footer that carries the 确认 button.
    cv2.rectangle(
        frame,
        (x, y + dialog_height),
        (x + dialog_width, y + dialog_height + int(0.14 * dialog_width)),
        (219, 219, 219),
        -1,
    )
    center = (x + 0.918 * dialog_width, y + 0.455 * dialog_width)
    half_w = int(round(0.062 * dialog_width))
    half_h = int(round(0.027 * dialog_width))
    cv2.rectangle(
        frame,
        (int(center[0]) - half_w, int(center[1]) - half_h),
        (int(center[0]) + half_w, int(center[1]) + half_h),
        (39, 110, 207),
        -1,
    )
    return frame, center


class _FakeGate:
    def __init__(self, presence):
        self._presence = list(presence)
        self.calls = 0
        self.local_checks = []

    def present(self, _frame, *, local=False):
        self.calls += 1
        self.local_checks.append(local)
        if not self._presence:
            return False
        if len(self._presence) == 1:
            return self._presence[0]
        return self._presence.pop(0)


class _FakeTracker:
    def __init__(self, *, actionable=True):
        self.actionable = actionable
        self.calls = 0
        self.resets = 0
        self.candidate_detector = None

    def update(self, _frame, _timestamp):
        self.calls += 1
        return SimpleNamespace(actionable=self.actionable, center=(30.0, 40.0))

    def reset(self):
        self.resets += 1


class LieDetectorRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        self.roi = (300, 120, 600, 420)

    def _runtime(self, presence, *, tracker=None, confirm=None, **config):
        settings = {
            "backend": "native",
            "panel_confirm_frames": 2,
            "panel_miss_frames": 3,
            "panel_check_interval_seconds": 0.0,
            # Existing tests focus on panel/confirm semantics. Dedicated ROI
            # locking behaviour is exercised separately below.
            "roi_lock_samples": 1,
        }
        settings.update(config)
        return LieDetectorRuntime(
            settings,
            panel_gate=_FakeGate(presence),
            roi_detector=lambda _frame: self.roi,
            confirm_detector=confirm or (lambda _frame: None),
            tracker=tracker or _FakeTracker(),
        )

    def test_gate_rejection_keeps_runtime_idle_even_if_roi_is_returned(self):
        tracker = _FakeTracker()
        runtime = self._runtime([False], tracker=tracker)

        result = runtime.update(self.frame, 1.0)

        self.assertFalse(result.engaged)
        self.assertEqual(result.phase, PHASE_IDLE)
        self.assertEqual(tracker.calls, 0)

    def test_pauses_immediately_but_moves_only_after_confirmation(self):
        tracker = _FakeTracker()
        runtime = self._runtime([True], tracker=tracker)

        first = runtime.update(self.frame, 1.0)
        second = runtime.update(self.frame, 2.0)

        self.assertTrue(first.engaged)
        self.assertFalse(first.panel_confirmed)
        self.assertFalse(first.panel_just_confirmed)
        self.assertIsNone(first.target_frame)
        self.assertTrue(second.panel_confirmed)
        self.assertTrue(second.panel_just_confirmed)
        self.assertEqual(second.target_frame, (330.0, 160.0))
        self.assertEqual(tracker.calls, 2)

    def test_panel_confirmation_event_is_emitted_once(self):
        runtime = self._runtime([True], panel_confirm_frames=2)

        first = runtime.update(self.frame, 1.0)
        second = runtime.update(self.frame, 2.0)
        third = runtime.update(self.frame, 3.0)

        self.assertFalse(first.panel_just_confirmed)
        self.assertTrue(second.panel_just_confirmed)
        self.assertFalse(third.panel_just_confirmed)
        self.assertTrue(third.panel_confirmed)

        runtime.reset()
        self.assertFalse(runtime.update(self.frame, 4.0).panel_just_confirmed)
        self.assertTrue(runtime.update(self.frame, 5.0).panel_just_confirmed)

    def test_panel_confirmation_does_not_require_roi_or_tracker(self):
        tracker = _FakeTracker()
        runtime = LieDetectorRuntime(
            {"backend": "native", "panel_confirm_frames": 2, "panel_miss_frames": 3},
            panel_gate=_FakeGate([True]),
            roi_detector=lambda _frame: None,
            confirm_detector=lambda _frame: None,
            tracker=tracker,
        )

        first = runtime.update(self.frame, 1.0)
        second = runtime.update(self.frame, 2.0)

        self.assertFalse(first.panel_just_confirmed)
        self.assertTrue(second.panel_just_confirmed)
        self.assertEqual(tracker.calls, 0)

    def test_non_actionable_track_never_moves_pointer(self):
        runtime = self._runtime(
            [True],
            tracker=_FakeTracker(actionable=False),
            panel_confirm_frames=1,
        )

        result = runtime.update(self.frame, 1.0)

        self.assertTrue(result.engaged)
        self.assertIsNone(result.target_frame)

    def test_roi_is_median_locked_before_tracking_and_then_stays_fixed(self):
        rois = iter(
            [
                (300, 100, 600, 420),
                (304, 122, 596, 399),
                (302, 104, 598, 418),
                # Must be ignored after the first three samples lock.
                (360, 180, 520, 340),
            ]
        )
        tracker = _FakeTracker()
        runtime = LieDetectorRuntime(
            {
                "panel_confirm_frames": 2,
                "panel_miss_frames": 3,
                "roi_lock_samples": 3,
                "backend": "native",
            },
            panel_gate=_FakeGate([True]),
            roi_detector=lambda _frame: next(rois),
            confirm_detector=lambda _frame: None,
            tracker=tracker,
        )

        first = runtime.update(self.frame, 1.0)
        second = runtime.update(self.frame, 2.0)
        locked = runtime.update(self.frame, 3.0)
        held = runtime.update(self.frame, 4.0)

        self.assertIsNone(first.panel_roi)
        self.assertIsNone(second.panel_roi)
        self.assertEqual(tracker.calls, 2)
        self.assertEqual(locked.panel_roi, (302, 104, 598, 418))
        self.assertEqual(held.panel_roi, locked.panel_roi)
        self.assertEqual(locked.target_frame, (332.0, 144.0))
        self.assertEqual(held.target_frame, (332.0, 144.0))

    @patch("src.engine.LieDetectorRuntime.LieDetectorTracker")
    @patch("src.engine.LieDetectorRuntime.LieShapeYoloDetector")
    def test_ranker_config_is_forwarded_to_online_tracker(
        self,
        detector_class,
        tracker_class,
    ):
        detector = detector_class.return_value
        tracker = tracker_class.return_value
        runtime = LieDetectorRuntime(
            {
                "backend": "native",
                "identity_ranker_model": "models/lie_identity_ranker.txt",
                "identity_ranker_min_margin": 2.0,
                "multi_hypothesis_identity": True,
                "stale_coast_recovery": True,
                "identity_safety": True,
                "state_aware_ranker": True,
                "motion_corroboration_model": "models/motion.txt",
                "switch_event_model": "models/switch.txt",
                "switch_event_min_probability": 0.65,
                "switch_motion_features": True,
                "preassociation_flow": True,
                "flow_association": True,
                "flow_coast": True,
            }
        )

        self.assertIs(runtime._ensure_tracker(), tracker)
        tracker_class.assert_called_once_with(
            candidate_detector=detector,
            multi_hypothesis_identity=True,
            stale_coast_recovery=True,
            identity_ranker_model="models/lie_identity_ranker.txt",
            identity_ranker_min_margin=2.0,
            identity_safety=True,
            state_aware_ranker=True,
            motion_corroboration_model="models/motion.txt",
            switch_event_model="models/switch.txt",
            switch_event_min_probability=0.65,
            switch_motion_features=True,
            preassociation_flow=True,
            flow_association=True,
            flow_coast=True,
        )

    def test_sdk_receives_fixed_roi_immediately_without_detection(self):
        tracker = _FakeTracker()
        tracker.update = Mock(wraps=tracker.update)
        gate = _FakeGate([True])
        gate.present = Mock(wraps=gate.present)
        roi_detector = Mock(side_effect=AssertionError("ROI detector must not run"))
        runtime = LieDetectorRuntime(
            {"backend": "trtyolo", "roi_lock_samples": 99},
            panel_gate=gate, roi_detector=roi_detector, tracker=tracker,
        )
        self.frame[81:549, 290:992] = (10, 20, 30)
        first = runtime.update(self.frame, 1.0)
        self.assertTrue(first.engaged)
        self.assertIsNone(first.target_frame)
        self.assertEqual(first.panel_roi, (290, 81, 702, 468))
        sdk_frame = tracker.update.call_args.args[0]
        self.assertEqual(sdk_frame.shape, (468, 702, 3))
        np.testing.assert_array_equal(sdk_frame, self.frame[81:549, 290:992])
        self.assertIs(gate.present.call_args.args[0], self.frame)
        self.assertEqual(tracker.update.call_args.args[1], 1.0)
        self.assertEqual(runtime._roi_samples, [])
        roi_detector.assert_not_called()
        second = runtime.update(self.frame, 2.0)
        self.assertEqual(second.target_frame, (320.0, 121.0))
        self.assertEqual(tracker.calls, 2)

    def test_sdk_fixed_roi_is_not_scaled_when_capture_size_changes(self):
        tracker = _FakeTracker()
        tracker.update = Mock(wraps=tracker.update)
        runtime = self._runtime(
            [True], tracker=tracker, backend="trtyolo", panel_confirm_frames=1
        )
        runtime.update(self.frame, 1.0)
        resized = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = runtime.update(resized, 2.0)
        self.assertEqual(result.panel_roi, (290, 81, 702, 468))
        self.assertEqual(result.target_frame, (320.0, 121.0))
        self.assertEqual(tracker.update.call_args.args[0].shape, (468, 702, 3))

    def test_sdk_fixed_roi_still_requires_panel_presence(self):
        tracker = _FakeTracker()
        runtime = self._runtime([False], tracker=tracker, backend="trtyolo")
        result = runtime.update(self.frame, 1.0)
        self.assertFalse(result.engaged)
        self.assertEqual(tracker.calls, 0)
        self.assertIsNone(result.panel_roi)

    def test_sdk_fixed_roi_state_is_reset_when_panel_disappears(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True, False], tracker=tracker, backend="trtyolo",
            panel_confirm_frames=1, panel_miss_frames=1,
        )
        first = runtime.update(self.frame, 1.0)
        self.assertEqual(first.target_frame, (320.0, 121.0))
        result = runtime.update(self.frame, 2.0)
        self.assertEqual(result.phase, PHASE_CONFIRM)
        self.assertIsNone(result.target_frame)
        self.assertIsNone(runtime._roi)
        self.assertEqual(tracker.resets, 1)

    @patch("src.engine.LieDetectorRuntime.LieShapeYoloDetector")
    @patch("src.engine.LieDetectorRuntime.LieDetectorSdkTracker")
    def test_sdk_is_default_and_does_not_load_native_detector(
        self, sdk_class, detector_class
    ):
        runtime = LieDetectorRuntime({})
        self.assertIs(runtime._ensure_tracker(), sdk_class.return_value)
        sdk_class.assert_called_once_with(
            "trtyoloLie_SDK_v220_x64/bin64", force_cpu=False,
            session_recording=True, session_image_fps=3.0, session_max_memory_mb=128.0,
        )
        detector_class.assert_not_called()

    @patch("src.engine.LieDetectorRuntime.LieDetectorSdkTracker")
    def test_sdk_configuration_is_forwarded(self, sdk_class):
        runtime = LieDetectorRuntime({
            "backend": "trtyolo",
            "sdk_directory": "vendor/bin64",
            "sdk_force_cpu": True,
            "sdk_session_recording": False,
            "sdk_session_image_fps": 2.0,
            "sdk_session_max_memory_mb": 64.0,
        })
        runtime._ensure_tracker()
        sdk_class.assert_called_once_with(
            "vendor/bin64", force_cpu=True, session_recording=False,
            session_image_fps=2.0, session_max_memory_mb=64.0,
        )

    @patch("src.engine.LieDetectorRuntime.LieShapeYoloDetector")
    @patch("src.engine.LieDetectorRuntime.LieDetectorSdkTracker")
    def test_sdk_creation_failure_does_not_fall_back(self, sdk_class, detector_class):
        sdk_class.side_effect = RuntimeError("not authorized")
        runtime = LieDetectorRuntime({"backend": "trtyolo"})
        with self.assertRaisesRegex(RuntimeError, "not authorized"):
            runtime._ensure_tracker()
        detector_class.assert_not_called()

    def test_sdk_custom_fixed_roi_is_cropped_without_resizing(self):
        tracker = _FakeTracker()
        tracker.update = Mock(wraps=tracker.update)
        runtime = self._runtime(
            [True], tracker=tracker, backend="trtyolo", panel_confirm_frames=1,
            sdk_roi=[200, 100, 400, 300],
        )
        result = runtime.update(self.frame, 1.0)
        self.assertEqual(result.panel_roi, (200, 100, 400, 300))
        self.assertEqual(result.target_frame, (230.0, 140.0))
        self.assertEqual(tracker.update.call_args.args[0].shape, (300, 400, 3))

    def test_fixed_roi_local_center_is_offset_once(self):
        tracker = _FakeTracker()
        tracker.update = Mock(return_value=SimpleNamespace(actionable=True, center=(350.0, 235.0)))
        runtime = self._runtime([True], tracker=tracker, backend="trtyolo", panel_confirm_frames=1)
        result = runtime.update(self.frame, 1.0)
        self.assertEqual(result.tracking.center, (350.0, 235.0))
        self.assertEqual(result.target_frame, (640.0, 316.0))
        self.assertEqual(result.panel_roi, (290, 81, 702, 468))

    def test_invalid_fixed_roi_configuration_is_rejected(self):
        for roi in ([1, 2, 3], [-1, 81, 702, 468], [290, 81, 0, 468], [290, 81, 702.5, 468]):
            with self.subTest(roi=roi):
                with self.assertRaisesRegex(ValueError, "sdk_roi"):
                    LieDetectorRuntime({"backend": "trtyolo", "sdk_roi": roi})

    def test_fixed_roi_outside_frame_does_not_call_sdk(self):
        tracker = _FakeTracker()
        runtime = self._runtime([True], tracker=tracker, backend="trtyolo", panel_confirm_frames=1)
        for frame in (np.zeros((500, 1282, 3), np.uint8), np.zeros((693, 900, 3), np.uint8)):
            with self.subTest(shape=frame.shape):
                result = runtime.update(frame, 1.0)
                self.assertTrue(result.engaged)
                self.assertIsNone(result.target_frame)
                self.assertIsNone(result.tracking)
        self.assertEqual(tracker.calls, 0)

    def test_fixed_roi_at_frame_boundary_is_not_truncated(self):
        tracker = _FakeTracker()
        tracker.update = Mock(wraps=tracker.update)
        runtime = self._runtime([True], tracker=tracker, backend="trtyolo", panel_confirm_frames=1)
        frame = np.zeros((549, 992, 3), np.uint8)
        result = runtime.update(frame, 1.0)
        self.assertEqual(result.target_frame, (320.0, 121.0))
        self.assertEqual(tracker.update.call_args.args[0].shape, (468, 702, 3))

    def test_unknown_backend_is_rejected(self):
        runtime = LieDetectorRuntime({"backend": "typo"})
        with self.assertRaisesRegex(ValueError, "Unknown lie-detector backend"):
            runtime._ensure_tracker()

    def test_tracking_failure_publishes_hold_instead_of_stale_target(self):
        tracker = _FakeTracker()
        runtime = self._runtime([True], tracker=tracker, panel_confirm_frames=1)
        self.assertIsNotNone(runtime.update(self.frame, 1.0).target_frame)
        tracker.update = Mock(side_effect=RuntimeError("SDK frame failed"))
        result = runtime.update(self.frame, 2.0)
        self.assertTrue(result.engaged)
        self.assertEqual(result.panel_roi, self.roi)
        self.assertIsNone(result.tracking)
        self.assertIsNone(result.target_frame)

    def test_close_releases_tracker_once_and_clears_runtime_state(self):
        tracker = _FakeTracker()
        tracker.close = Mock()
        runtime = self._runtime([True], tracker=tracker, panel_confirm_frames=1)
        runtime.update(self.frame, 1.0)
        runtime.close()
        runtime.close()
        tracker.close.assert_called_once()
        self.assertIsNone(runtime._tracker)
        self.assertEqual(runtime.phase, PHASE_IDLE)
        self.assertIsNone(runtime._roi)

    @patch("src.engine.LieDetectorRuntime.logger")
    def test_waiting_for_roi_logs_that_tracker_has_not_received_frames(self, log):
        runtime = LieDetectorRuntime(
            {"backend": "native"}, panel_gate=_FakeGate([True]),
            roi_detector=lambda _frame: None, tracker=_FakeTracker(),
        )
        for timestamp in (1.0, 2.0, 5.9, 6.0):
            runtime.update(self.frame, timestamp)
        messages = [call.args[0] for call in log.info.call_args_list]
        waiting = [text for text in messages if "Waiting for ROI" in text]
        self.assertEqual(len(waiting), 2)
        self.assertIn("tracker has not received an input frame yet", waiting[0])
        self.assertEqual(runtime._tracker.calls, 0)

    def test_confirmed_panel_checks_title_less_often_but_tracks_every_frame(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True], tracker=tracker, panel_check_interval_seconds=0.2,
        )
        first = runtime.update(self.frame, 0.0)
        second = runtime.update(self.frame, 0.01)
        self.assertIsNone(first.target_frame)
        self.assertIsNotNone(second.target_frame)
        for timestamp in (0.05, 0.10, 0.19):
            result = runtime.update(self.frame, timestamp)
            self.assertIsNotNone(result.target_frame)
            self.assertFalse(result.panel_just_confirmed)
        self.assertEqual(runtime._panel_gate.calls, 2)
        self.assertEqual(tracker.calls, 5)
        self.assertEqual(runtime._seen_streak, 2)
        runtime.update(self.frame, 0.22)
        self.assertEqual(runtime._panel_gate.calls, 3)
        self.assertEqual(runtime._panel_gate.local_checks, [False, False, True])

    def test_brief_miss_and_recovery_do_not_interrupt_confirmed_tracking(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True, True, False, True], tracker=tracker,
            panel_check_interval_seconds=0.2,
        )
        runtime.update(self.frame, 0.0)
        confirmed = runtime.update(self.frame, 0.01)
        missed = runtime.update(self.frame, 0.25)
        skipped = runtime.update(self.frame, 0.30)
        recovered = runtime.update(self.frame, 0.50)
        for result in (confirmed, missed, skipped, recovered):
            self.assertTrue(result.panel_confirmed)
            self.assertIsNotNone(result.target_frame)
            self.assertIsNotNone(result.tracking)
        self.assertFalse(recovered.panel_just_confirmed)
        self.assertEqual(tracker.calls, 5)
        self.assertEqual(tracker.resets, 0)
        self.assertEqual(runtime._miss_streak, 0)
        self.assertEqual(runtime._seen_streak, 1)

    def test_exit_counts_failed_checks_not_skipped_frames(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True, False], tracker=tracker, panel_confirm_frames=1,
            panel_check_interval_seconds=0.2, panel_miss_frames=3,
        )
        runtime.update(self.frame, 0.0)
        first_miss = runtime.update(self.frame, 0.25)
        for timestamp in (0.26, 0.30, 0.40):
            result = runtime.update(self.frame, timestamp)
            self.assertEqual(runtime._miss_streak, 1)
            self.assertIsNotNone(result.target_frame)
        second_miss = runtime.update(self.frame, 0.50)
        runtime.update(self.frame, 0.55)
        self.assertEqual(runtime._miss_streak, 2)
        self.assertIsNotNone(first_miss.target_frame)
        self.assertIsNotNone(second_miss.target_frame)
        closed = runtime.update(self.frame, 0.75)
        self.assertEqual(closed.phase, PHASE_CONFIRM)
        self.assertIsNone(closed.target_frame)
        self.assertEqual(runtime._panel_gate.calls, 4)
        self.assertEqual(tracker.calls, 7)
        self.assertEqual(tracker.resets, 1)

    def test_new_panel_after_confirm_phase_requires_fresh_confirmation(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True, True, False, True, True], tracker=tracker,
            panel_miss_frames=1, panel_check_interval_seconds=0.2,
        )
        runtime.update(self.frame, 0.0)
        self.assertTrue(runtime.update(self.frame, 0.01).panel_just_confirmed)
        closed = runtime.update(self.frame, 0.25)
        self.assertEqual(closed.phase, PHASE_CONFIRM)
        self.assertFalse(runtime._panel_confirmed)
        new_panel = runtime.update(self.frame, 0.30)
        self.assertFalse(new_panel.panel_confirmed)
        self.assertIsNone(new_panel.target_frame)
        confirmed = runtime.update(self.frame, 0.31)
        self.assertTrue(confirmed.panel_just_confirmed)
        self.assertIsNotNone(confirmed.target_frame)
        self.assertEqual(tracker.resets, 1)

    def test_sdk_keeps_receiving_frames_during_transient_title_misses(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True, True, False, True], tracker=tracker, backend="trtyolo",
            panel_check_interval_seconds=0.2,
        )
        runtime.update(self.frame, 0.0)
        runtime.update(self.frame, 0.01)
        for timestamp in (0.25, 0.30, 0.50):
            result = runtime.update(self.frame, timestamp)
            self.assertEqual(result.panel_roi, (290, 81, 702, 468))
            self.assertEqual(result.target_frame, (320.0, 121.0))
        self.assertEqual(tracker.calls, 5)
        self.assertEqual(tracker.resets, 0)

    def test_unconfirmed_panel_miss_still_holds_pointer(self):
        tracker = _FakeTracker()
        runtime = self._runtime([True, False, True, True], tracker=tracker)
        first = runtime.update(self.frame, 0.0)
        missed = runtime.update(self.frame, 0.01)
        recovered = runtime.update(self.frame, 0.02)
        confirmed = runtime.update(self.frame, 0.03)
        for result in (first, missed, recovered):
            self.assertFalse(result.panel_confirmed)
            self.assertIsNone(result.target_frame)
        self.assertTrue(confirmed.panel_confirmed)
        self.assertEqual(tracker.calls, 3)
        self.assertEqual(runtime._panel_gate.calls, 4)

    def test_capture_size_change_forces_an_early_panel_check(self):
        runtime = self._runtime(
            [True], backend="trtyolo", panel_confirm_frames=1,
            panel_check_interval_seconds=0.2,
        )
        runtime.update(self.frame, 0.0)
        resized = np.zeros((1080, 1920, 3), np.uint8)
        result = runtime.update(resized, 0.05)
        self.assertEqual(runtime._panel_gate.calls, 2)
        self.assertEqual(result.panel_roi, (290, 81, 702, 468))
        self.assertIsNotNone(result.target_frame)

    @patch("src.engine.LieDetectorRuntime.logger")
    def test_panel_gate_log_reports_check_cost_and_misses(self, log):
        runtime = self._runtime(
            [True, False], panel_confirm_frames=1,
            panel_check_interval_seconds=0.2,
        )
        runtime.update(self.frame, 0.0)
        runtime.update(self.frame, 0.25)
        messages = [call.args[0] for call in log.info.call_args_list]
        diagnostics = [text for text in messages if "Panel gate;" in text]
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("checks=1", diagnostics[0])
        self.assertIn("miss_streak=1/3", diagnostics[0])
        self.assertIn("avg_check_ms=", diagnostics[0])

    def test_sdk_mouse_diagnostics_are_forwarded_without_native_calls(self):
        tracker = _FakeTracker()
        tracker.record_mouse_action = Mock()
        runtime = self._runtime([True], backend="trtyolo", tracker=tracker)
        runtime.record_mouse_action(1.0, "move_requested", (700, 450), 1.5, 1.2)
        tracker.record_mouse_action.assert_called_once_with(1.0, "move_requested", (700, 450), 1.5, 1.2)
        native = self._runtime([True], backend="native", tracker=_FakeTracker())
        native.record_mouse_action(1.0, "no_target", None, None, 1.2)

    def test_short_panel_miss_holds_before_confirm_phase(self):
        tracker = _FakeTracker()
        runtime = self._runtime(
            [True, False, False],
            tracker=tracker,
            panel_confirm_frames=1,
            panel_miss_frames=2,
        )

        self.assertEqual(runtime.update(self.frame, 1.0).phase, PHASE_PANEL)
        self.assertEqual(runtime.update(self.frame, 2.0).phase, PHASE_PANEL)
        third = runtime.update(self.frame, 3.0)

        self.assertEqual(third.phase, PHASE_CONFIRM)
        self.assertTrue(third.engaged)
        self.assertEqual(tracker.resets, 1)
        self.assertFalse(runtime._roi_locked)
        self.assertEqual(runtime._roi_samples, [])

    def test_confirm_button_is_clicked_once_then_runtime_resumes(self):
        dialog = [(750.0, 415.0), (750.0, 415.0), None]
        runtime = self._runtime(
            [True, False, False, False],
            confirm=lambda _frame: dialog.pop(0) if dialog else None,
            panel_confirm_frames=1,
            panel_miss_frames=2,
        )

        runtime.update(self.frame, 0.0)
        runtime.update(self.frame, 0.1)
        clicked = runtime.update(self.frame, 0.2)
        held = runtime.update(self.frame, 0.3)
        closed = runtime.update(self.frame, 0.4)

        self.assertEqual(clicked.confirm_click, (750.0, 415.0))
        # A second click must respect the interval instead of spamming.
        self.assertIsNone(held.confirm_click)
        self.assertFalse(closed.engaged)
        self.assertEqual(closed.phase, PHASE_IDLE)

    def test_confirm_phase_times_out_when_no_dialog_appears(self):
        runtime = self._runtime(
            [True, False, False],
            panel_confirm_frames=1,
            panel_miss_frames=2,
            confirm_wait_seconds=1.0,
        )

        runtime.update(self.frame, 0.0)
        runtime.update(self.frame, 0.1)
        self.assertEqual(runtime.update(self.frame, 0.2).phase, PHASE_CONFIRM)
        self.assertFalse(runtime.update(self.frame, 1.5).engaged)


class LiePanelGateTest(unittest.TestCase):
    def test_matches_title_text_and_rejects_plain_frame(self):
        gate = LiePanelGate(TITLE_TEMPLATE, 0.70)
        template = cv2.imread(str(TITLE_TEMPLATE), cv2.IMREAD_COLOR)
        frame = _noise_frame()
        height, width = template.shape[:2]
        frame[84:84 + height, 284:284 + width] = template

        self.assertTrue(gate.present(frame))
        self.assertFalse(gate.present(_noise_frame(seed=11)))

    def test_cached_title_uses_a_smaller_search_region(self):
        gate = LiePanelGate(TITLE_TEMPLATE, 0.70)
        template = cv2.imread(str(TITLE_TEMPLATE), cv2.IMREAD_COLOR)
        frame = _noise_frame()
        height, width = template.shape[:2]
        frame[84:84 + height, 284:284 + width] = template
        self.assertTrue(gate.present(frame))
        self.assertEqual(gate._cached_title_box[:2], (284, 84))
        with patch("cv2.matchTemplate", wraps=cv2.matchTemplate) as match:
            self.assertTrue(gate.present(frame, local=True))
        region = match.call_args.args[0]
        self.assertLess(region.shape[0], frame.shape[0] // 4)
        self.assertLess(region.shape[1], frame.shape[1] // 2)
        self.assertEqual(gate.last_search, "local")

    def test_local_miss_falls_back_to_global_for_panel_movement(self):
        gate = LiePanelGate(TITLE_TEMPLATE, 0.70)
        template = cv2.imread(str(TITLE_TEMPLATE), cv2.IMREAD_COLOR)
        height, width = template.shape[:2]
        initial = _noise_frame()
        initial[84:84 + height, 284:284 + width] = template
        self.assertTrue(gate.present(initial))
        moved = _noise_frame(seed=11)
        moved[54:54 + height, 600:600 + width] = template
        self.assertTrue(gate.present(moved, local=True))
        self.assertEqual(gate.last_search, "local+global")
        self.assertEqual(gate._cached_title_box[:2], (600, 54))
        self.assertTrue(gate.present(moved, local=True))
        self.assertEqual(gate.last_search, "local")

    def test_missing_title_fails_both_local_and_global_checks(self):
        gate = LiePanelGate(TITLE_TEMPLATE, 0.70)
        template = cv2.imread(str(TITLE_TEMPLATE), cv2.IMREAD_COLOR)
        height, width = template.shape[:2]
        frame = _noise_frame()
        frame[84:84 + height, 284:284 + width] = template
        self.assertTrue(gate.present(frame))
        self.assertFalse(gate.present(_noise_frame(seed=11), local=True))
        self.assertEqual(gate.last_search, "local+global")
        self.assertLess(gate.last_score, gate.threshold)

    def test_scale_cache_is_dropped_when_frame_size_changes(self):
        gate = LiePanelGate(TITLE_TEMPLATE, 0.70)
        template = cv2.imread(str(TITLE_TEMPLATE), cv2.IMREAD_COLOR)
        frame = _noise_frame()
        height, width = template.shape[:2]
        frame[84:84 + height, 284:284 + width] = template
        self.assertTrue(gate.present(frame))

        scaled = cv2.resize(frame, (1920, 1080))
        self.assertTrue(gate.present(scaled))


class SuccessDialogTest(unittest.TestCase):
    def test_finds_confirm_button_of_success_dialog(self):
        frame, expected = _success_dialog_frame()

        found = find_success_confirm(frame)

        self.assertIsNotNone(found)
        self.assertLess(abs(found[0] - expected[0]), 6.0)
        self.assertLess(abs(found[1] - expected[1]), 6.0)

    def test_scales_with_dialog_size(self):
        frame, expected = _success_dialog_frame(
            width=1920, height=1080, dialog_width=387
        )

        found = find_success_confirm(frame)

        self.assertIsNotNone(found)
        self.assertLess(abs(found[0] - expected[0]), 9.0)
        self.assertLess(abs(found[1] - expected[1]), 9.0)

    def test_no_dialog_returns_none(self):
        self.assertIsNone(find_success_confirm(_noise_frame()))
        # Sandy terrain alone carries the button's orange tones but no dialog.
        self.assertIsNone(find_success_confirm(_terrain_frame()))


if __name__ == "__main__":
    unittest.main()
