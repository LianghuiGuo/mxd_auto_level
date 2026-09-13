import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def present(self, _frame):
        self.calls += 1
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
            "panel_confirm_frames": 2,
            "panel_miss_frames": 3,
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
            {"panel_confirm_frames": 2, "panel_miss_frames": 3},
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
