import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.lie_detector_replay import (
    EvaluationPhase,
    cursor_ground_truth,
    evaluation_phase,
    load_recording_from_config,
)


class LieDetectorReplayTest(unittest.TestCase):
    def test_evaluation_phase_uses_labeled_active_window(self):
        self.assertEqual(evaluation_phase(1.9, 2.0, 8.0), EvaluationPhase.SEED)
        self.assertEqual(evaluation_phase(2.0, 2.0, 8.0), EvaluationPhase.ACTIVE)
        self.assertEqual(evaluation_phase(8.0, 2.0, 8.0), EvaluationPhase.ACTIVE)
        self.assertEqual(evaluation_phase(8.1, 2.0, 8.0), EvaluationPhase.DONE)

    def test_cursor_ground_truth_scales_with_panel_resolution(self):
        frame = np.zeros((463, 690, 3), dtype=np.uint8)
        cv2.circle(frame, (230, 170), 17, (0, 255, 0), 6)
        standard = cursor_ground_truth(frame)
        self.assertIsNotNone(standard)
        self.assertLess(np.linalg.norm(np.subtract(standard, (230, 170))), 2.0)

        enlarged = cv2.resize(frame, (1056, 726), interpolation=cv2.INTER_NEAREST)
        scaled = cursor_ground_truth(enlarged)
        self.assertIsNotNone(scaled)
        expected = (230 * 1056 / 690, 170 * 726 / 463)
        self.assertLess(np.linalg.norm(np.subtract(scaled, expected)), 3.0)

    def test_recording_config_exposes_active_window(self):
        recording = load_recording_from_config(
            Path("ml/videos/测谎录屏7.mp4"),
            Path("ml/lie_videos_config.json"),
        )
        self.assertIsNotNone(recording)
        self.assertEqual(recording["active_start"], 4.35)
        self.assertEqual(recording["active_end"], 8.96)


if __name__ == "__main__":
    unittest.main()
