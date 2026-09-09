import math
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
        self.assertTrue(any(view.role == "real" for view in result.constellation))

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

    def test_constellation_tracks_all_shapes_and_labels_real(self):
        tracker = LieDetectorTracker()
        timestamp = 0.0
        # Frame 0: white seed on the true target.
        seed = ShapeDetection(
            (100.0, 100.0),
            40.0,
            (60, 60, 80, 80),
            score=1.0,
            source="bright",
            observed_radius=40.0,
            orientation=0.0,
        )
        for index, center in enumerate([(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]):
            det = ShapeDetection(
                center,
                40.0,
                (int(center[0] - 40), int(center[1] - 40), 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.0,
            )
            tracker._spawn(det if index else seed, timestamp)
        tracker.target_id = 1
        tracker.white_seen = True
        for track in tracker.tracks.values():
            track.role = "real" if track.id == 1 else "bg"

        # Frame 1: decoys translate together; REAL breaks layout and rotates.
        detections = [
            ShapeDetection(
                (104.0, 140.0),
                40.0,
                (64, 100, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.6,
            ),
            ShapeDetection(
                (204.0, 102.0),
                40.0,
                (164, 62, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.0,
            ),
            ShapeDetection(
                (304.0, 102.0),
                40.0,
                (264, 62, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.0,
            ),
        ]
        tracker.previous_centers = np.asarray(
            [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)], dtype=np.float64
        )
        tracker.previous_orientations = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        tracker._annotate_collective_motion(detections)
        self.assertGreater(detections[0].collective_residual, 20.0)
        self.assertGreater(detections[0].collective_rotation_residual, 15.0)
        tracker._associate(detections, 1.0 / 30.0)
        switched = tracker._assign_roles()
        self.assertFalse(switched)
        self.assertEqual(tracker.target_id, 1)
        roles = {track.id: track.role for track in tracker.tracks.values()}
        self.assertEqual(roles[1], "real")
        self.assertEqual(roles[2], "bg")
        self.assertEqual(roles[3], "bg")

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
                orientation=0.0,
            )
            for index in range(5)
        ]
        outlier = ShapeDetection(
            (430.0, 180.0),
            40.0,
            (0, 0, 80, 80),
            orientation=0.55,
        )
        current.append(outlier)
        tracker._annotate_collective_motion(current)
        self.assertLess(np.linalg.norm(tracker.collective_delta - (4.0, 2.0)), 2.5)
        self.assertLess(
            float(np.median([item.collective_residual for item in current[:-1]])),
            5.0,
        )
        self.assertGreater(outlier.collective_residual, 30.0)
        self.assertGreater(outlier.collective_rotation_residual, 15.0)
        self.assertLess(
            float(np.median([item.collective_rotation_residual for item in current[:-1]])),
            5.0,
        )

    def test_limit_patterns_caps_at_twenty(self):
        tracker = LieDetectorTracker()
        detections = [
            ShapeDetection(
                (float(index), 0.0),
                40.0,
                (0, 0, 80, 80),
                shape_distance=float(index) * 0.01,
            )
            for index in range(35)
        ]
        detections[30].source = "bright"
        limited = tracker._limit_patterns(detections)
        self.assertEqual(len(limited), 20)
        self.assertEqual(limited[0].source, "bright")

    def test_fuses_prefer_yolo_and_gap_fill_classical(self):
        tracker = LieDetectorTracker()
        tracker.target_radius = 40.0
        classical = [
            ShapeDetection(
                (100.0, 100.0),
                40.0,
                (60, 60, 80, 80),
                source="contour",
                shape_distance=0.2,
                orientation=0.5,
                observed_radius=40.0,
            ),
            ShapeDetection(
                (250.0, 180.0),
                40.0,
                (210, 140, 80, 80),
                source="contour",
                shape_distance=0.3,
                observed_radius=40.0,
            ),
        ]
        learned = [
            ShapeDetection(
                (103.0, 98.0),
                42.0,
                (63, 58, 84, 84),
                score=0.72,
                source="yolo",
                observed_radius=42.0,
                yolo_confidence=0.72,
            )
        ]
        fused = tracker._fuse_learned_candidates(classical, learned)
        self.assertEqual(len(fused), 2)
        yolo_hit = next(item for item in fused if item.source == "yolo")
        self.assertAlmostEqual(yolo_hit.yolo_confidence, 0.72)
        self.assertAlmostEqual(yolo_hit.shape_distance, 0.2)
        self.assertAlmostEqual(yolo_hit.orientation, 0.5)
        classical_gap = next(item for item in fused if item.source == "contour")
        self.assertAlmostEqual(classical_gap.center[0], 250.0)

    def test_skips_classical_when_yolo_coverage_is_high(self):
        tracker = LieDetectorTracker()
        tracker.target_radius = 40.0
        classical = [
            ShapeDetection(
                (400.0, 200.0),
                20.0,
                (380, 180, 40, 40),
                source="contour",
                observed_radius=20.0,
            )
        ]
        learned = [
            ShapeDetection(
                (50.0 + 40 * index, 80.0),
                42.0,
                (30 + 40 * index, 60, 40, 40),
                score=0.8,
                source="yolo",
                observed_radius=42.0,
                yolo_confidence=0.8,
            )
            for index in range(8)
        ]
        fused = tracker._fuse_learned_candidates(classical, learned)
        self.assertEqual(len(fused), 8)
        self.assertTrue(all(item.source == "yolo" for item in fused))

    def test_four_cue_score_prefers_rotating_direction_outlier(self):
        tracker = LieDetectorTracker()
        for center in [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0), (400.0, 100.0)]:
            tracker._spawn(
                ShapeDetection(
                    center,
                    40.0,
                    (int(center[0] - 40), int(center[1] - 40), 80, 80),
                    source="yolo",
                    observed_radius=40.0,
                    yolo_confidence=0.9,
                    orientation=0.0,
                ),
                0.0,
            )
        tracker.target_id = 1
        tracker.white_seen = True
        updated = [
            ShapeDetection(
                (118.0, 90.0),
                40.0,
                (78, 50, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.7,
                collective_residual=20.0,
                collective_rotation_residual=25.0,
            ),
            ShapeDetection(
                (204.0, 102.0),
                40.0,
                (164, 62, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.0,
                collective_residual=1.0,
            ),
            ShapeDetection(
                (304.0, 102.0),
                40.0,
                (264, 62, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.0,
                collective_residual=1.0,
            ),
            ShapeDetection(
                (404.0, 102.0),
                40.0,
                (364, 62, 80, 80),
                source="yolo",
                observed_radius=40.0,
                yolo_confidence=0.9,
                orientation=0.0,
                collective_residual=1.0,
            ),
        ]
        for track, detection in zip(
            sorted(tracker.tracks.values(), key=lambda item: item.id), updated
        ):
            track.update(detection, 1.0 / 30.0)
            track.visible_streak = 6
            track.association_quality_history = [1.0, 1.0, 1.0]
            track.motion_history = [track.velocity] * 5
        scores = tracker._score_real_candidates()
        self.assertEqual(max(scores, key=scores.get), 1)
        real = tracker.tracks[1]
        decoy = tracker.tracks[2]
        self.assertGreater(real.rotation_score, decoy.rotation_score)
        self.assertGreater(real.direction_score, decoy.direction_score)
        self.assertGreater(real.speed_score, decoy.speed_score)
        self.assertGreaterEqual(real.rigidity_score, decoy.rigidity_score)
        self.assertLessEqual(real.direction_score, 180.0)

    def test_angle_delta_wraps_full_turn(self):
        delta = LieDetectorTracker._angle_delta_degrees(
            math.radians(2.0), math.radians(358.0)
        )
        self.assertLess(delta, 8.0)

    def test_matches_rotation_for_arbitrary_polygons(self):
        def polygon(sides, angle_degrees, inner_ratio=None):
            image = np.zeros((180, 180), dtype=np.uint8)
            center = np.array([90.0, 90.0])
            points = []
            count = sides if inner_ratio is None else sides * 2
            for index in range(count):
                radius = (
                    55.0
                    if inner_ratio is None or index % 2 == 0
                    else 55.0 * inner_ratio
                )
                angle = math.radians(
                    angle_degrees - 90.0 + index * 360.0 / count
                )
                points.append(
                    center + radius * np.array([math.cos(angle), math.sin(angle)])
                )
            cv2.fillPoly(image, [np.asarray(points, dtype=np.int32)], 220)
            return image

        detection = ShapeDetection(
            (90.0, 90.0),
            60.0,
            (28, 28, 124, 124),
            source="yolo",
        )
        for sides, inner_ratio in ((3, None), (4, None), (5, 0.44), (6, None)):
            with self.subTest(sides=sides):
                first = LieDetectorTracker._rotation_descriptor(
                    LieDetectorTracker._rotation_edge_map(
                        polygon(sides, 0.0, inner_ratio)
                    ),
                    detection,
                )
                rotated = LieDetectorTracker._rotation_descriptor(
                    LieDetectorTracker._rotation_edge_map(
                        polygon(sides, 12.0, inner_ratio)
                    ),
                    detection,
                )
                self.assertIsNotNone(first)
                self.assertIsNotNone(rotated)
                delta, confidence = LieDetectorTracker._match_rotation_descriptors(
                    first, rotated
                )
                self.assertIsNotNone(delta)
                self.assertAlmostEqual(math.degrees(delta), 12.0, delta=2.1)
                self.assertGreater(confidence, 0.4)

    def test_plain_circle_has_no_observable_rotation(self):
        image = np.zeros((180, 180), dtype=np.uint8)
        cv2.circle(image, (90, 90), 55, 220, -1)
        detection = ShapeDetection(
            (90.0, 90.0), 60.0, (28, 28, 124, 124), source="yolo"
        )
        descriptor = LieDetectorTracker._rotation_descriptor(
            LieDetectorTracker._rotation_edge_map(image), detection
        )
        self.assertIsNone(descriptor)

    def test_rotation_score_rewards_sustained_direction(self):
        dt = [1.0 / 30.0] * 4
        quality = [0.9] * 4
        coherent = LieDetectorTracker._rotation_motion_score(
            [math.radians(value) for value in (5.0, 6.0, 5.0, 6.0)],
            dt,
            quality,
        )
        oscillating = LieDetectorTracker._rotation_motion_score(
            [math.radians(value) for value in (5.0, -5.0, 5.0, -5.0)],
            dt,
            quality,
        )
        unreliable = LieDetectorTracker._rotation_motion_score(
            [math.radians(5.0)] * 4,
            dt,
            [0.9, 0.0, 0.0, 0.0],
        )
        self.assertGreater(coherent, 10.0)
        self.assertLess(oscillating, coherent * 0.2)
        self.assertEqual(unreliable, 0.0)

    def test_challenger_needs_repeated_evidence_to_flip_real(self):
        tracker = LieDetectorTracker()
        for center in ((100.0, 100.0), (140.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        current = tracker.tracks[1]
        current.rotation_score = 20.0
        current.direction_score = 40.0
        current.translation_residual = 12.0
        tracker._score_real_candidates = lambda: {1: 10.0, 2: 100.0}

        # A strong lead alone must not flip the label on its first frame.
        self.assertFalse(tracker._assign_roles())
        self.assertEqual(tracker.target_id, 1)

        switched = False
        for _ in range(5):
            switched = tracker._assign_roles()
            if switched:
                break
        self.assertTrue(switched)
        self.assertEqual(tracker.target_id, 2)

    def test_multi_hypothesis_delays_disconnected_far_recovery(self):
        tracker = LieDetectorTracker(multi_hypothesis_identity=True)
        for center in ((100.0, 100.0), (380.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        tracker._score_real_candidates = lambda: {1: 0.0, 2: 100.0}

        for _ in range(3):
            self.assertFalse(tracker._assign_roles())

        switched = False
        for _ in range(6):
            switched = tracker._assign_roles()
            if switched:
                break
        self.assertTrue(switched)
        self.assertEqual(tracker.target_id, 2)
        self.assertNotIn(2, tracker.target_hypotheses)

    def test_multi_hypothesis_recovers_extreme_loss_more_quickly(self):
        tracker = LieDetectorTracker(multi_hypothesis_identity=True)
        for center in ((100.0, 100.0), (520.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        tracker._score_real_candidates = lambda: {1: 0.0, 2: 100.0}

        self.assertFalse(tracker._assign_roles())
        switched = False
        for _ in range(5):
            switched = tracker._assign_roles()
            if switched:
                break
        self.assertTrue(switched)
        self.assertEqual(tracker.target_id, 2)

    def test_multi_hypothesis_switches_over_continuous_handoff(self):
        tracker = LieDetectorTracker(multi_hypothesis_identity=True)
        for center in ((100.0, 100.0), (150.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        tracker._score_real_candidates = lambda: {1: 0.0, 2: 100.0}

        switched = False
        for _ in range(12):
            switched = tracker._assign_roles()
            if switched:
                break

        self.assertTrue(switched)
        self.assertEqual(tracker.target_id, 2)
        self.assertGreaterEqual(len(tracker.target_hypotheses), 2)

    def test_multi_hypothesis_ignores_one_frame_cue_spike(self):
        tracker = LieDetectorTracker(multi_hypothesis_identity=True)
        for center in ((100.0, 100.0), (150.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        score_frames = iter((
            {1: 0.0, 2: 100.0},
            {1: 100.0, 2: 0.0},
            {1: 100.0, 2: 0.0},
            {1: 100.0, 2: 0.0},
        ))
        tracker._score_real_candidates = lambda: next(score_frames)

        for _ in range(4):
            self.assertFalse(tracker._assign_roles())

        self.assertEqual(tracker.target_id, 1)

    def test_shadow_switch_requires_identity_continuity_and_score_support(self):
        tracker = LieDetectorTracker(shadow_switch=True)
        for center in ((100.0, 100.0), (140.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        challenger = tracker.tracks[2]
        challenger.visible_streak = 40
        challenger.association_quality_history = [0.01] * 3
        challenger.direction_score = 40.0
        challenger.speed_score = 5.0
        tracker._score_real_candidates = lambda: {1: 10.0, 2: 50.0}

        for _ in range(5):
            self.assertFalse(tracker._assign_roles())
        self.assertEqual(tracker.target_id, 1)

        challenger.association_quality_history = [1.0] * 3
        self.assertFalse(tracker._assign_roles())
        self.assertFalse(tracker._assign_roles())
        self.assertTrue(tracker._assign_roles())
        self.assertEqual(tracker.target_id, 2)

    def test_long_lost_real_releases_after_coast_budget(self):
        tracker = LieDetectorTracker()
        for center in ((100.0, 100.0), (110.0, 100.0)):
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80)),
                0.0,
            )
        tracker.target_id = 1
        for _ in range(26):
            tracker.tracks[1].mark_missed()
        candidate = tracker.tracks[2]
        candidate.visible_streak = 8
        candidate.association_quality_history = [1.0] * 3
        tracker._score_real_candidates = lambda: {2: 40.0}

        tracker._drop_stale()
        self.assertIsNone(tracker.target_id)
        self.assertTrue(tracker._assign_roles())
        self.assertEqual(tracker.target_id, 2)

    def test_coast_estimate_can_remain_acquired_but_not_actionable(self):
        tracker = LieDetectorTracker(classical_candidates=False)
        tracker._spawn(
            ShapeDetection((100.0, 100.0), 40.0, (60, 60, 80, 80)),
            0.0,
        )
        tracker.target_id = 1
        tracker.white_seen = True
        blank = np.zeros((240, 320, 3), dtype=np.uint8)

        result = None
        for frame in range(1, 21):
            result = tracker.update(blank, frame / 30.0)
        self.assertIsNotNone(result)
        self.assertTrue(result.acquired)
        self.assertIsNotNone(result.center)
        self.assertFalse(result.actionable)
        self.assertTrue(result.recovery_active)
        self.assertGreater(result.position_uncertainty_px, 10.0)

    def test_long_gap_reassociation_commits_only_after_provisional_observations(self):
        tracker = LieDetectorTracker(provisional_reassociation=True)
        centers = ((100.0, 100.0), (220.0, 100.0), (100.0, 220.0), (220.0, 220.0))

        def detections(offset=0.0):
            return [
                ShapeDetection(
                    (center[0] + offset, center[1]),
                    40.0,
                    (0, 0, 80, 80),
                    appearance=np.asarray(
                        [1.0, 0.0] if index == 0 else [0.0, 1.0],
                        dtype=np.float32,
                    ),
                )
                for index, center in enumerate(centers)
            ]

        for detection in detections():
            tracker._spawn(detection, 0.0)
        tracker.target_id = 1
        for _ in range(8):
            tracker.tracks[1].mark_missed()

        self.assertFalse(tracker._associate(detections(2.0), 1.0 / 30.0))
        pending = tracker.pending_reassociation
        self.assertIsNotNone(pending)
        child_id = pending.child_track_id
        self.assertEqual(tracker.target_id, 1)

        self.assertFalse(tracker._associate(detections(3.0), 2.0 / 30.0))
        self.assertFalse(tracker._associate(detections(4.0), 3.0 / 30.0))
        self.assertTrue(tracker._associate(detections(5.0), 4.0 / 30.0))
        self.assertEqual(tracker.target_id, child_id)
        self.assertNotIn(1, tracker.tracks)

    def test_motion_reliability_rejects_unstable_association(self):
        tracker = LieDetectorTracker()
        track = tracker._spawn(
            ShapeDetection((100.0, 100.0), 40.0, (0, 0, 80, 80)),
            0.0,
        )
        track.visible_streak = 8
        track.association_quality_history = [1.0, 0.05, 0.05]
        self.assertLess(track.motion_reliability, 0.1)

    def test_low_speed_tracks_do_not_get_direction_score(self):
        tracker = LieDetectorTracker()
        centers = [(100.0, 100.0), (200.0, 100.0), (300.0, 100.0), (400.0, 100.0)]
        for center in centers:
            tracker._spawn(
                ShapeDetection(center, 40.0, (0, 0, 80, 80), orientation=0.0),
                0.0,
            )
        for index, track in enumerate(tracker.tracks.values()):
            dx = 0.15 if index % 2 else -0.15
            dy = 0.12 if index < 2 else -0.12
            track.update(
                ShapeDetection(
                    (centers[index][0] + dx, centers[index][1] + dy),
                    40.0,
                    (0, 0, 80, 80),
                    orientation=0.0,
                ),
                1.0 / 30.0,
            )
        tracker._score_real_candidates()
        self.assertTrue(
            all(track.direction_score == 0.0 for track in tracker.tracks.values())
        )

    def test_appearance_descriptor_is_rotation_tolerant(self):
        image = np.zeros((160, 160), dtype=np.uint8)
        cv2.ellipse(image, (80, 80), (50, 32), 20, 0, 360, 180, -1)
        cv2.line(image, (45, 65), (115, 95), 245, 7)
        detection = ShapeDetection(
            (80.0, 80.0),
            55.0,
            (24, 24, 112, 112),
            source="yolo",
        )
        first = LieDetectorTracker._appearance_descriptor(image, detection)
        matrix = cv2.getRotationMatrix2D((80, 80), 25.0, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (160, 160))
        rotated = LieDetectorTracker._appearance_descriptor(
            rotated_image, detection
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(rotated)
        self.assertLess(float(1.0 - np.dot(first, rotated)), 0.08)


if __name__ == "__main__":
    unittest.main()
