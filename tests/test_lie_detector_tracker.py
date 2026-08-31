import unittest

import cv2
import numpy as np

from src.engine.LieDetectorTracker import LieDetectorTracker, ShapeDetection


class LieDetectorTrackerTest(unittest.TestCase):
    @staticmethod
    def frame(target, decoys=(), alpha=1.0):
        image = np.full((360, 520, 3), (120, 165, 190), dtype=np.uint8)
        for center in decoys:
            cv2.circle(image, center, 48, (70, 105, 125), 4)
        overlay = image.copy()
        cv2.circle(overlay, target, 55, (255, 255, 255), -1)
        return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    def test_acquires_bright_circle(self):
        tracker = LieDetectorTracker(min_radius=30, max_radius=70, hough_param2=22)
        result = tracker.update(self.frame((260, 180)), 0.0)
        self.assertTrue(result.acquired)
        self.assertIsNotNone(result.center)
        self.assertLess(np.linalg.norm(np.subtract(result.center, (260, 180))), 12)

    def test_preserves_target_id_while_circle_moves(self):
        tracker = LieDetectorTracker(min_radius=30, max_radius=70, hough_param2=22)
        target_id = None
        errors = []
        for index in range(20):
            target = (180 + index * 4, 170 + index)
            alpha = max(0.22, 1.0 - index * 0.045)
            result = tracker.update(
                self.frame(target, decoys=((360 - index * 2, 100), (350, 270)), alpha=alpha),
                index / 30.0,
            )
            if target_id is None:
                target_id = result.target_id
            self.assertEqual(result.target_id, target_id)
            self.assertIsNotNone(result.center)
            errors.append(np.linalg.norm(np.subtract(result.center, target)))
        self.assertLess(float(np.percentile(errors, 95)), 25.0)

    def test_classifies_star_as_contour_target(self):
        image = np.full((360, 520, 3), (120, 165, 190), dtype=np.uint8)
        points = []
        center = np.array([260.0, 180.0])
        for index in range(10):
            radius = 58.0 if index % 2 == 0 else 25.0
            angle = -np.pi / 2.0 + index * np.pi / 5.0
            points.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))
        cv2.fillPoly(image, [np.asarray(points, dtype=np.int32)], (255, 255, 255))
        tracker = LieDetectorTracker(min_radius=20, max_radius=80)
        result = tracker.update(image, 0.0)
        self.assertTrue(result.acquired)
        self.assertEqual(tracker.target_kind, "contour")
        self.assertLess(np.linalg.norm(np.subtract(result.center, center)), 15.0)
        contour_detections = [
            detection for detection in result.detections if detection.source == "contour"
        ]
        self.assertTrue(contour_detections)
        self.assertTrue(
            all(
                abs(detection.radius - tracker.target_radius) < 1e-6
                for detection in contour_detections
            )
        )

    def test_contour_hypothesis_beam_keeps_coast_and_detection_branches(self):
        tracker = LieDetectorTracker(hypothesis_count=4)
        seed = ShapeDetection((100.0, 100.0), 45.0, (55, 55, 90, 90))
        tracker._seed_target_hypothesis(seed, 0.0)
        detections = [
            ShapeDetection(
                (108.0, 100.0),
                45.0,
                (63, 55, 90, 90),
                source="contour",
                shape_distance=0.05,
            ),
            ShapeDetection(
                (132.0, 102.0),
                46.0,
                (86, 56, 92, 92),
                source="contour",
                shape_distance=0.10,
            ),
        ]
        tracker._update_target_hypotheses(detections, 1.0 / 30.0, None)
        self.assertGreaterEqual(len(tracker.target_hypotheses), 2)
        self.assertLessEqual(len(tracker.target_hypotheses), 4)
        self.assertTrue(any(item.predicted_only for item in tracker.target_hypotheses))
        self.assertTrue(any(not item.predicted_only for item in tracker.target_hypotheses))

    def test_collective_motion_marks_non_group_outlier(self):
        tracker = LieDetectorTracker()
        previous = [
            ShapeDetection((80.0 + 70.0 * index, 100.0), 40.0, (0, 0, 80, 80), orientation=0.0)
            for index in range(5)
        ]
        tracker._annotate_collective_motion(previous)
        current = [
            ShapeDetection(
                (84.0 + 70.0 * index, 102.0),
                40.0,
                (0, 0, 80, 80),
                orientation=0.05,
            )
            for index in range(5)
        ]
        outlier = ShapeDetection(
            (430.0, 180.0),
            40.0,
            (0, 0, 80, 80),
            orientation=0.30,
        )
        current.append(outlier)
        tracker._annotate_collective_motion(current)
        self.assertLess(np.linalg.norm(tracker.collective_delta - (4.0, 2.0)), 2.0)
        self.assertLess(float(np.median([item.collective_residual for item in current[:-1]])), 4.0)
        self.assertGreater(outlier.collective_residual, 30.0)


if __name__ == "__main__":
    unittest.main()
