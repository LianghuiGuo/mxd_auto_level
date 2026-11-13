import argparse
import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tools.fragment_voting_experiment import (
    BackgroundMotion,
    FragmentVotingTracker,
    GuardedVotingTracker,
    MotionFeedbackTracker,
    OptimizedFragmentTracker,
    OriginalVotingTracker,
    RevisedMotionFeedbackTracker,
    SyntheticSequence,
    Vote,
    aggregate_recording_metrics,
    aggregate_votes,
    evaluate_recordings,
    evaluation_diagnostics,
    load_annotations,
    make_template,
    opening_template,
    placed_mask,
    real_metrics,
    recording_signature,
    resumable_recording,
    rotate_points,
    summarize,
)


def render(template, center, angle=0.0, size=(260, 200)):
    frame = np.full((size[1], size[0], 3), 65, dtype=np.uint8)
    mask = placed_mask(template, size, center, angle)
    frame[mask > 0] = 232
    return frame


class FragmentVotingGeometryTest(unittest.TestCase):
    def test_rotation_updates_fragment_offset_and_recovers_center(self):
        offset = np.array(((18.0, -7.0),))
        rotated = rotate_points(offset, 90.0)[0]
        np.testing.assert_allclose(rotated, (7.0, 18.0), atol=1e-5)
        center = np.array((115.0, 96.0))
        np.testing.assert_allclose(center + rotated - rotated, center)

        tracker = FragmentVotingTracker()
        tracker.initialize(make_template("irregular"), (110.0, 96.0))
        fragment = tracker.fragments[0]
        kernel, low = tracker._kernel(fragment, 90.0)
        rotated_points = rotate_points(fragment.points + fragment.offset, 90.0)
        np.testing.assert_array_equal(
            low, np.rint(rotated_points).astype(np.int32).min(axis=0)
        )
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_repeated_matches_from_one_fragment_cannot_inflate_votes(self):
        votes = [Vote(0, (100.0 + index * 0.1, 80.0), 0.9) for index in range(10)]
        votes += [Vote(1, (100.0, 81.0), 0.8), Vote(2, (101.0, 80.0), 0.85)]
        peak = aggregate_votes(votes, 6)[0]
        self.assertEqual(peak.votes, 3)
        self.assertLess(peak.evidence, 0.5)
        self.assertLess(math.dist(peak.center, (100, 80)), 2)

    def test_dispersed_centers_do_not_form_consensus(self):
        votes = [Vote(index, (index * 30.0, 80.0), 1.0) for index in range(6)]
        self.assertTrue(all(peak.votes == 1 for peak in aggregate_votes(votes, 6)))

    def test_interior_without_boundary_does_not_become_a_fragment(self):
        tracker = FragmentVotingTracker()
        tracker.initialize(make_template("circle"), (110, 96))
        self.assertNotIn(4, [fragment.fragment_id for fragment in tracker.fragments])
        self.assertGreaterEqual(len(tracker.fragments), 3)


class FragmentVotingTrackingTest(unittest.TestCase):
    def setUp(self):
        self.template = make_template("irregular")
        self.center = (110.0, 96.0)
        self.tracker = FragmentVotingTracker()
        self.tracker.initialize(self.template, self.center)

    def test_rotating_translating_outline_is_observed(self):
        for index in range(1, 13):
            timestamp = index / 24.0
            center = (110 + 28 * timestamp, 96 + 12 * timestamp)
            result = self.tracker.update(
                render(self.template, center, 48 * timestamp), timestamp
            )
            self.assertEqual(result.status, "observed")
            self.assertLess(math.dist(result.center, center), 3.0)
        self.assertGreater(result.angular_velocity_dps, 20)
        self.assertLess(abs(result.rotation_degrees - 48 * timestamp), 9)

    def test_similar_distractor_outside_local_search_is_ignored(self):
        timestamp = 1 / 24
        expected = (112.0, 97.0)
        frame = render(self.template, expected, 2.0)
        distractor = placed_mask(self.template, (260, 200), (205, 97), 2.0)
        frame[distractor > 0] = 232
        result = self.tracker.update(frame, timestamp)
        self.assertEqual(result.status, "observed")
        self.assertLess(math.dist(result.center, expected), 3)

    def test_ambiguous_identical_candidates_are_not_accepted(self):
        template = np.zeros((71, 71), dtype=np.uint8)
        cv2.ellipse(template, (35, 35), (8, 30), 0, 0, 360, 255, -1)
        tracker = FragmentVotingTracker()
        tracker.initialize(template, self.center)
        frame = render(template, (94, 96))
        other = placed_mask(template, (260, 200), (126, 96), 0.0)
        frame[other > 0] = 232
        result = tracker.update(frame, 1 / 24)
        self.assertNotEqual(result.status, "observed")
        self.assertEqual(result.confidence, 0.0)

    def test_stronger_out_of_gate_match_cannot_hide_local_candidate(self):
        frame = render(self.template, (120, 96), 3.0)
        outside = placed_mask(self.template, (260, 200), (132, 120), 0.0)
        frame[outside > 0] = 232
        features = self.tracker._features(frame)
        votes = self.tracker._fragment_votes(
            features, np.asarray(self.center), 16.0, 3.0
        )
        self.assertTrue(votes)
        self.assertTrue(
            all(math.dist(vote.center, self.center) <= 16.0 for vote in votes)
        )

    def test_partial_occlusion_preserves_independent_visible_fragments(self):
        frame = render(self.template, (112, 96), 3.0)
        frame[58:134, 127:155] = 65
        result = self.tracker.update(frame, 1 / 24)
        self.assertEqual(result.status, "observed")
        self.assertGreaterEqual(result.votes, 3)
        self.assertLess(math.dist(result.center, (112, 96)), 3)

    def test_border_cutoff_uses_visible_fragments_without_wrapping(self):
        tracker = FragmentVotingTracker()
        tracker.initialize(self.template, (238, 96))
        frame = render(self.template, (240, 96), 2.0)
        result = tracker.update(frame, 1 / 24)
        self.assertEqual(result.status, "observed")
        self.assertLess(math.dist(result.center, (240, 96)), 3)
        self.assertLess(result.center[0], frame.shape[1])

    def test_invisible_frames_are_prediction_then_lost(self):
        blank = np.full((200, 260, 3), 65, dtype=np.uint8)
        predicted = self.tracker.update(blank, 0.1)
        lost = self.tracker.update(blank, 0.6)
        self.assertEqual(predicted.status, "predicted")
        self.assertEqual(predicted.votes, 0)
        self.assertEqual(predicted.confidence, 0.0)
        self.assertEqual(lost.status, "lost")
        self.assertIsNone(lost.center)
        self.assertAlmostEqual(lost.seconds_without_observation, 0.6)

    def test_reacquisition_requires_new_image_evidence(self):
        blank = np.full((200, 260, 3), 65, dtype=np.uint8)
        self.assertEqual(self.tracker.update(blank, 0.6).status, "lost")
        result = self.tracker.update(render(self.template, self.center), 0.7)
        self.assertEqual(result.status, "observed")
        self.assertAlmostEqual(result.seconds_without_observation, 0.0)

    def test_prediction_uses_elapsed_seconds_not_frame_count(self):
        self.tracker.velocity = np.array((20.0, -10.0))
        self.tracker.angular_velocity = 40.0
        blank = np.full((200, 260, 3), 65, dtype=np.uint8)
        first = self.tracker.update(blank, 0.1)
        second = self.tracker.update(blank, 0.35)
        np.testing.assert_allclose(first.center, (112, 95))
        np.testing.assert_allclose(second.center, (117, 92.5))
        self.assertAlmostEqual(second.rotation_degrees, 14.0)

    def test_irregular_sampling_tracks_without_fixed_rate_hint(self):
        for timestamp in (0.04, 0.09, 0.17, 0.20, 0.29, 0.37):
            center = (110 + 24 * timestamp, 96 + 8 * timestamp)
            result = self.tracker.update(
                render(self.template, center, 40 * timestamp), timestamp
            )
            self.assertEqual(result.status, "observed")
            self.assertLess(math.dist(result.center, center), 3)

    def test_circle_does_not_report_observed_angular_velocity(self):
        tracker = FragmentVotingTracker()
        template = make_template("circle")
        tracker.initialize(template, self.center)
        self.assertTrue(tracker.rotationally_symmetric)
        result = tracker.update(render(template, (112, 96), 25.0), 0.1)
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.angular_velocity_dps, 0.0)
        self.assertLess(math.dist(result.center, (112, 96)), 3)

    def test_reset_discards_identity_geometry_and_motion(self):
        self.tracker.velocity[:] = 12
        self.tracker.reset()
        self.assertEqual(self.tracker.fragments, [])
        self.assertIsNone(self.tracker.center)
        np.testing.assert_array_equal(self.tracker.velocity, (0, 0))
        with self.assertRaises(RuntimeError):
            self.tracker.update(render(self.template, self.center), 1)
        self.tracker.initialize(make_template("circle"), (75, 75), 5.0)
        result = self.tracker.update(render(make_template("circle"), (76, 75)), 5.1)
        self.assertEqual(result.status, "observed")
        self.assertLess(math.dist(result.center, (76, 75)), 3)

    def test_input_boundaries_reject_bad_template_frames_and_timestamps(self):
        with self.assertRaises(ValueError):
            FragmentVotingTracker().initialize(
                np.zeros((21, 21), dtype=np.uint8), self.center
            )
        for timestamp in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.tracker.update(render(self.template, self.center), timestamp)
        with self.assertRaises(ValueError):
            self.tracker.update(np.zeros((0, 0, 3), dtype=np.uint8), 0.1)


class FragmentVotingEvaluationTest(unittest.TestCase):
    def test_synthetic_frames_are_reproducible(self):
        first = SyntheticSequence("distractors", 17)
        second = SyntheticSequence("distractors", 17)
        np.testing.assert_array_equal(first.sample(0.4).frame, second.sample(0.4).frame)
        different = SyntheticSequence("distractors", 29)
        self.assertFalse(
            np.array_equal(first.sample(0.4).frame, different.sample(0.4).frame)
        )

    def test_scoring_truth_is_not_in_tracker_update_api(self):
        sequence = SyntheticSequence("clean", 3)
        sample = sequence.sample(0.1)
        tracker = FragmentVotingTracker()
        tracker.initialize(sequence.template, sequence.initial_center)
        result = tracker.update(sample.frame, sample.timestamp)
        self.assertEqual(result.status, "observed")
        self.assertLess(math.dist(result.center, sample.truth_center), 3)
        with self.assertRaises(TypeError):
            tracker.update(
                sample.frame, sample.timestamp + 0.1, truth_center=sample.truth_center
            )

    def test_metrics_count_lost_frames_in_coverage_and_accuracy(self):
        rows = [
            {
                "timestamp": 0.1,
                "status": "observed",
                "error_px": 2.0,
                "visibility": 1.0,
                "processing_ms": 10.0,
            },
            {
                "timestamp": 0.2,
                "status": "predicted",
                "error_px": 8.0,
                "visibility": 0.0,
                "processing_ms": 10.0,
            },
            {
                "timestamp": 0.4,
                "status": "lost",
                "error_px": None,
                "visibility": 0.0,
                "processing_ms": 10.0,
            },
            {
                "timestamp": 0.7,
                "status": "lost",
                "error_px": None,
                "visibility": 0.0,
                "processing_ms": 10.0,
            },
        ]
        metrics = summarize(rows)
        self.assertEqual(metrics["prediction_coverage"], 0.5)
        self.assertEqual(metrics["observation_coverage"], 0.25)
        self.assertEqual(metrics["within_5px_all_frames"], 0.25)
        self.assertAlmostEqual(metrics["longest_lost_seconds"], 0.5)
        self.assertAlmostEqual(metrics["longest_without_observation_seconds"], 0.6)
        self.assertEqual(metrics["false_observations_when_invisible"], 0)


class OfflineRecordingEvaluationTest(unittest.TestCase):
    def test_seed_is_visible_shape_not_countdown_text(self):
        template = make_template("irregular")
        frame = render(template, (345, 260), size=(690, 463))
        cv2.putText(
            frame, "5", (300, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 10
        )
        seed = opening_template(frame)
        self.assertIsNotNone(seed)
        mask, center = seed
        self.assertLess(math.dist(center, (345, 260)), 10)
        tracker = FragmentVotingTracker()
        tracker.initialize(mask, center)
        self.assertGreaterEqual(len(tracker.fragments), 3)

    def test_no_visible_opening_shape_does_not_use_proxy_seed(self):
        frame = np.full((463, 690, 3), 65, dtype=np.uint8)
        cv2.circle(frame, (345, 260), 17, (0, 255, 0), 6)
        self.assertIsNone(opening_template(frame))

    def test_uninitialized_proxy_frames_remain_in_accuracy_denominator(self):
        rows = [
            {
                "evaluation_active": True,
                "proxy_x": 10,
                "proxy_error_px": None,
                "status": "uninitialized",
                "center_x": None,
                "processing_ms": 1.0,
            },
            {
                "evaluation_active": True,
                "proxy_x": 10,
                "proxy_error_px": 15.0,
                "status": "observed",
                "center_x": 25,
                "processing_ms": 4.0,
            },
            {
                "evaluation_active": True,
                "proxy_x": 10,
                "proxy_error_px": None,
                "status": "lost",
                "center_x": None,
                "processing_ms": 4.0,
            },
        ]
        metrics = real_metrics(rows, 30.0)
        self.assertEqual(metrics["cursor_proxy_frames"], 3)
        self.assertAlmostEqual(metrics["within_40px_proxy_all_frames"], 1 / 3)
        self.assertAlmostEqual(metrics["proxy_prediction_coverage"], 1 / 3)
        self.assertEqual(metrics["mean_cursor_proxy_error_px"], 15)

    def test_aggregate_is_frame_weighted_and_retains_failed_video(self):
        rows = [
            {
                "evaluation_active": True,
                "proxy_x": 10,
                "proxy_error_px": None,
                "status": "uninitialized",
                "processing_ms": 1.0,
            },
            {
                "evaluation_active": True,
                "proxy_x": 10,
                "proxy_error_px": 15.0,
                "status": "observed",
                "processing_ms": 4.0,
            },
        ]
        videos = [
            {"video": "a.mp4", "source_frames": 1, "initialization": "failed"},
            {
                "video": "b.mp4",
                "source_frames": 1,
                "initialization": "opening_white_silhouette",
            },
        ]
        metrics = aggregate_recording_metrics(rows, videos)
        self.assertEqual(metrics["initialization_failures"], ["a.mp4"])
        self.assertEqual(metrics["proxy_prediction_coverage"], 0.5)
        self.assertEqual(metrics["within_40px_proxy_all_frames"], 0.5)
        self.assertEqual(metrics["mean_tracking_ms"], 4.0)


class OptimizedTrackerTest(unittest.TestCase):
    def setUp(self):
        self.template = make_template("irregular")
        self.center = (120.0, 100.0)

    def test_background_translation_produces_small_residual(self):
        rng = np.random.default_rng(7)
        gray = cv2.GaussianBlur(
            rng.integers(0, 255, (240, 320), dtype=np.uint8), (5, 5), 1
        )
        valid = np.ones(gray.shape, bool)
        observer = BackgroundMotion()
        observer.commit(gray, valid, np.array((160, 120)))
        moved = cv2.warpAffine(
            gray, np.array(((1.0, 0.0, 3.0), (0.0, 1.0, -2.0))), (320, 240)
        )
        residual, diagnostics = observer.observe(moved, valid, np.array((163, 118)), 25)
        self.assertEqual(diagnostics["background_status"], "accepted")
        self.assertLess(residual[65:210, 15:305].mean(), 0.06)

    def test_untextured_background_does_not_manufacture_motion(self):
        gray = np.full((200, 260), 65, dtype=np.uint8)
        observer = BackgroundMotion()
        observer.commit(gray, np.ones(gray.shape, bool), np.array(self.center))
        residual, diagnostics = observer.observe(
            gray, np.ones(gray.shape, bool), np.array(self.center), 30
        )
        self.assertNotEqual(diagnostics["background_status"], "accepted")
        self.assertEqual(float(residual.sum()), 0)

    def test_weak_contrast_and_reversed_brightness_are_trackable(self):
        for foreground, background in ((100, 85), (45, 85)):
            tracker = OptimizedFragmentTracker()
            tracker.initialize(self.template, self.center)
            frame = np.full((220, 300, 3), background, dtype=np.uint8)
            mask = placed_mask(self.template, (300, 220), (122, 100), 2)
            frame[mask > 0] = foreground
            result = tracker.update(frame, 0.05)
            self.assertEqual(result.status, "observed")
            self.assertLess(math.dist(result.center, (122, 100)), 4)

    def test_invalid_cursor_boundary_cannot_create_shape_evidence(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center)
        frame = np.full((220, 300, 3), 65, dtype=np.uint8)
        mask = placed_mask(self.template, (300, 220), self.center, 0) > 0
        frame[mask] = (0, 255, 0)
        invalid = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        result = tracker.update(frame, 0.1, invalid)
        self.assertNotEqual(result.status, "observed")
        self.assertEqual(result.confidence, 0)

    def test_masked_template_boundary_is_excluded(self):
        valid = np.ones(self.template.shape, bool)
        valid[:, 40:] = False
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center, template_valid=valid)
        origin = (self.template.shape[0] - 1) / 2
        self.assertTrue(np.all(tracker.points[:, 0] + origin < 40))

    def test_white_frames_fuse_template_without_using_truth(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center)
        for i in range(1, 7):
            t = i * 0.04
            result = tracker.update(
                render(self.template, (120 + 15 * t, 100), 40 * t, (300, 220)), t
            )
            self.assertEqual(result.status, "observed")
        self.assertGreater(tracker.white_samples, 1)
        self.assertLess(abs(result.rotation_degrees - 40 * t), 8)

    def test_symmetry_period_and_circle_angular_unobservability(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(make_template("star"), self.center)
        self.assertAlmostEqual(tracker.period, 72.0)
        tracker.initialize(make_template("circle"), self.center)
        self.assertEqual(tracker.period, 0)
        result = tracker.update(
            render(make_template("circle"), (122, 100), 20, (300, 220)), 0.1
        )
        self.assertEqual(result.angular_velocity_dps, 0)

    def test_invisible_coast_is_bounded_and_recovery_requires_continuity(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center)
        blank = np.full((220, 300, 3), 65, dtype=np.uint8)
        self.assertEqual(tracker.update(blank, 0.1).status, "predicted")
        self.assertEqual(tracker.update(blank, 0.6).status, "lost")
        visible = render(self.template, self.center, size=(300, 220))
        first = tracker.update(visible, 0.65)
        self.assertNotEqual(first.status, "observed")
        final = tracker.update(visible, 0.75)
        self.assertEqual(final.status, "observed")
        self.assertEqual(final.diagnostics["recovered"], 1)

    def test_boundary_and_irregular_time_intervals(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, (279, 100))
        for t in (0.03, 0.09, 0.18):
            result = tracker.update(
                render(self.template, (279 + 8 * t, 100), 20 * t, (300, 220)), t
            )
            self.assertEqual(result.status, "observed")
            self.assertLess(math.dist(result.center, (279 + 8 * t, 100)), 4)

    def test_lost_prediction_does_not_drift_indefinitely(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center)
        tracker.velocity[:] = (100, 0)
        blank = np.full((220, 300, 3), 65, dtype=np.uint8)
        for i in range(1, 31):
            tracker.update(blank, i / 10)
        self.assertLess(tracker.center[0], 190)
        self.assertLess(np.linalg.norm(tracker.velocity), 0.1)

    def test_cache_and_candidate_budget_are_bounded(self):
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center)
        first = tracker._rotated(20)
        self.assertIs(first[0], tracker._rotated(20)[0])
        residual = np.zeros((240, 320), np.float32)
        self.assertLessEqual(
            len(tracker._centers(np.array(self.center), 150, residual)), 320
        )
        tracker.reset()
        self.assertEqual(tracker.cache, {})

    def test_annotations_are_validated_and_scored_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.csv"
            path.write_text(
                "video,frame,x,y,phase,visibility\na.mp4,3,40,50,occluded,partial\n"
            )
            annotations = load_annotations(path)
            self.assertEqual(annotations[("a.mp4", 3)]["center"], (40, 50))
            path.write_text("video,frame,x,y\na.mp4,3,nan,50\n")
            with self.assertRaises(ValueError):
                load_annotations(path)
        rows = [
            {
                "evaluation_active": True,
                "status": "observed",
                "processing_ms": 10,
                "frame_budget_ms": 30,
                "truth_x": 40,
                "truth_error_px": 3,
                "annotation_phase": "occluded",
            }
        ]
        diagnostics = evaluation_diagnostics(rows)
        self.assertEqual(diagnostics["mean_independent_truth_error_px"], 3)
        self.assertEqual(
            diagnostics["annotation_phase_metrics"]["occluded"]["frames"], 1
        )
        tracker = OptimizedFragmentTracker()
        tracker.initialize(self.template, self.center)
        with self.assertRaises(TypeError):
            tracker.update(render(self.template, self.center), 0.1, truth=(40, 50))


class RevisedTrackerTest(unittest.TestCase):
    def test_occluded_contour_uses_local_votes_not_whole_ring_average(self):
        template = make_template("irregular")
        center = (120.0, 100.0)
        tracker = OptimizedFragmentTracker("local_vote")
        tracker.initialize(template, center)
        frame = render(template, center, size=(300, 220))
        frame[:, 130:] = 65
        valid = np.ones(frame.shape[:2], bool)
        maps, _ = tracker._edge_maps(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), valid)
        quality, _, groups, _ = tracker._score(
            np.array([center], np.float32),
            0.0,
            maps,
            valid,
            np.zeros(valid.shape, np.float32),
        )
        self.assertLess(tracker.last_score_full_quality[0], 0.70)
        self.assertGreaterEqual(np.count_nonzero(groups[0] >= 0.72), 4)
        self.assertGreater(quality[0], 0.74)

    def test_masked_star_symmetry_ignores_unknown_pixels(self):
        template = make_template("star")
        valid = np.ones(template.shape, bool)
        cv2.circle(valid.view(np.uint8), (40, 35), 9, 0, -1)
        template[~valid] = 0
        tracker = OptimizedFragmentTracker("template_fixed")
        tracker.initialize(template, (120, 100), template_valid=valid)
        self.assertEqual(tracker.period, 72.0)

    def test_white_center_is_aligned_to_fixed_template_not_shifted_centroid(self):
        template = make_template("irregular")
        tracker = OptimizedFragmentTracker("template_fixed")
        tracker.initialize(template, (120, 100))
        shifted = cv2.warpAffine(
            template,
            np.array(((1.0, 0.0, 3.0), (0.0, 1.0, -2.0))),
            (template.shape[1], template.shape[0]),
        )
        mask, valid, point = tracker._align_white_center(
            (shifted, np.ones(template.shape, bool), (120.0, 100.0)), 0.0
        )
        self.assertEqual(point, (123.0, 98.0))
        self.assertGreater(
            np.count_nonzero((mask > 0) & (template > 0)) / np.count_nonzero(template),
            0.98,
        )
        self.assertTrue(valid[20, 20])

    def test_fusion_recomputes_symmetry_period(self):
        template = np.zeros((81, 81), np.uint8)
        cv2.rectangle(template, (15, 15), (65, 65), 255, -1)
        tracker = OptimizedFragmentTracker("template_fixed")
        tracker.initialize(template, (120, 100))
        tracker.period = 360.0
        tracker._fuse_white((template, np.ones(template.shape, bool), (120, 100)), 0.0)
        self.assertEqual(tracker.period, 90.0)
        self.assertGreater(tracker.white_samples, 1)

    def test_search_budget_preserves_outer_coverage_without_residual(self):
        tracker = OptimizedFragmentTracker("search_fixed")
        tracker.initialize(make_template("irregular"), (200, 150))
        centers = tracker._centers(
            np.array((200.0, 150.0)), 140.0, np.zeros((320, 420), np.float32)
        )
        self.assertLessEqual(len(centers), 320)
        self.assertGreater(np.linalg.norm(centers - (200, 150), axis=1).max(), 125)
        self.assertGreater(
            np.count_nonzero(np.linalg.norm(centers - (200, 150), axis=1) > 100), 25
        )

    def test_static_background_hypothesis_cannot_pass_recovery(self):
        tracker = OptimizedFragmentTracker("revised")
        tracker.initialize(make_template("irregular"), (120, 100))
        tracker.current_background = {"background_status": "accepted"}
        for i in range(6):
            accepted = tracker._recovery_evidence(
                np.array((120.0, 100.0)), 0.0, np.ones(8), 0.2, 0.9, False, 1 + i * 0.04
            )
            self.assertFalse(accepted)

    def test_background_rotation_does_not_count_as_foreground_rotation(self):
        tracker = OptimizedFragmentTracker("revised")
        tracker.initialize(make_template("irregular"), (120, 100))
        tracker.current_background = {"background_status": "accepted"}
        groups = np.ones(8)
        self.assertFalse(
            tracker._recovery_evidence(
                np.array((120.0, 100.0)), 0.0, groups, 0.2, 0.9, False, 1.0
            )
        )
        for index in range(1, 5):
            tracker.pending_bg_angle = index * 2.0
            accepted = tracker._recovery_evidence(
                np.array((120.0, 100.0)),
                index * 2.0,
                groups,
                0.2,
                0.9,
                False,
                1 + index * 0.04,
            )
            self.assertFalse(accepted)

    def test_recovery_checks_group_and_pose_consistency(self):
        tracker = OptimizedFragmentTracker("revised")
        tracker.initialize(make_template("irregular"), (120, 100))
        tracker.current_background = {"background_status": "unavailable"}
        groups = np.ones(8)
        for i, angle in enumerate((0.0, 60.0, 0.0, 60.0)):
            self.assertFalse(
                tracker._recovery_evidence(
                    np.array((120.0, 100.0)),
                    angle,
                    groups,
                    0.0,
                    0.9,
                    False,
                    1 + i * 0.04,
                )
            )
        self.assertEqual(tracker.pending_frames, 1)
        tracker._reset_recovery()
        for i in range(4):
            accepted = tracker._recovery_evidence(
                np.array((120.0, 100.0)), 0.0, groups, 0.0, 0.9, False, 2 + i * 0.04
            )
        self.assertTrue(accepted)

    def test_candidate_recovery_does_not_change_official_state(self):
        tracker = OptimizedFragmentTracker("revised")
        tracker.initialize(make_template("irregular"), (120, 100))
        original = tracker.model.copy()
        tracker._recovery_evidence(
            np.array((160.0, 110.0)), 8.0, np.ones(8), 0.2, 0.9, False, 1.0
        )
        np.testing.assert_array_equal(tracker.model, original)
        np.testing.assert_allclose(tracker.center, (120, 100))
        self.assertEqual(tracker.angular_velocity, 0.0)

    def test_no_residual_and_blank_image_are_never_observed(self):
        tracker = OptimizedFragmentTracker("revised")
        tracker.initialize(make_template("irregular"), (120, 100))
        blank = np.full((220, 300, 3), 65, np.uint8)
        for timestamp in (0.1, 0.6, 1.0):
            result = tracker.update(blank, timestamp)
            self.assertNotEqual(result.status, "observed")
            self.assertEqual(result.confidence, 0)
        self.assertEqual(result.status, "lost")


class OriginalVotingTrackerTest(unittest.TestCase):
    def setUp(self):
        self.template = make_template("irregular")
        self.center = (120.0, 100.0)
        self.tracker = OriginalVotingTracker()
        self.tracker.initialize(self.template, self.center)

    def test_fragment_matches_explicitly_infer_center_and_never_enumerate_centers(self):
        with (
            patch.object(
                OptimizedFragmentTracker,
                "_centers",
                side_effect=AssertionError("center enumeration forbidden"),
            ),
            patch.object(
                OptimizedFragmentTracker,
                "_score",
                side_effect=AssertionError("whole contour scoring forbidden"),
            ),
        ):
            result = self.tracker.update(
                render(self.template, (122, 100), 2, (300, 220)), 0.05
            )
        self.assertEqual(result.status, "observed")
        self.assertGreaterEqual(result.votes, 3)
        self.assertEqual(self.tracker.output_trace["source"], "fragment_votes")
        self.assertTrue(self.tracker.winning_records)
        for record in self.tracker.match_records:
            np.testing.assert_allclose(
                np.array(record["matched_point"]) - record["rotated_offset"],
                record["center"],
                atol=1e-5,
            )
        inferred = np.average(
            [r["center"] for r in self.tracker.winning_records],
            axis=0,
            weights=[r["quality"] for r in self.tracker.winning_records],
        )
        np.testing.assert_allclose(
            inferred, self.tracker.output_trace["measured_center"]
        )
        np.testing.assert_allclose(
            0.9 * inferred + 0.1 * np.asarray(self.tracker.output_trace["prediction"]),
            result.center,
        )

    def test_rotation_transforms_fragment_and_center_offset_together(self):
        fragment = self.tracker.fragments[0]
        kernel, offset = self.tracker._rotated_fragment(fragment, 90.0)
        np.testing.assert_allclose(
            offset, rotate_points(fragment.offset[None, :], 90)[0], atol=1e-5
        )
        self.assertEqual(kernel.shape[1], kernel.shape[2])
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)
        self.assertIs(kernel, self.tracker._rotated_fragment(fragment, 90.0)[0])

    def test_pure_interior_grid_does_not_vote(self):
        tracker = OriginalVotingTracker(grid_size=3)
        tracker.initialize(make_template("circle"), self.center)
        self.assertNotIn(4, [f.fragment_id for f in tracker.fragments])
        with self.assertRaises(ValueError):
            OriginalVotingTracker(grid_size=2)

    def test_visible_partial_fragments_can_locate_center(self):
        frame = render(self.template, (122, 100), 2, (300, 220))
        frame[:, 137:] = 65
        result = self.tracker.update(frame, 0.05)
        self.assertEqual(result.status, "observed")
        self.assertLess(math.dist(result.center, (122, 100)), 4)

    def test_duplicate_fragment_peaks_are_counted_once(self):
        result = self.tracker.update(
            render(self.template, self.center, size=(300, 220)), 0.05
        )
        ids = [record["fragment_id"] for record in self.tracker.winning_records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(result.votes, len(ids))
        self.assertTrue(
            all(
                record["angle"] == self.tracker.selected_angle
                for record in self.tracker.winning_records
            )
        )

    def test_invalid_cursor_cannot_create_matching_votes(self):
        mask = placed_mask(self.template, (300, 220), self.center, 0) > 0
        frame = np.full((220, 300, 3), 65, np.uint8)
        frame[mask] = (0, 255, 0)
        invalid = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        result = self.tracker.update(frame, 0.05, invalid)
        self.assertNotEqual(result.status, "observed")
        self.assertEqual(result.confidence, 0)

    def test_blank_prediction_and_loss_are_distinct_from_observation(self):
        blank = np.full((220, 300, 3), 65, np.uint8)
        self.assertEqual(self.tracker.update(blank, 0.1).status, "predicted")
        result = self.tracker.update(blank, 0.6)
        self.assertEqual(result.status, "lost")
        self.assertIsNone(result.center)
        self.assertEqual(result.votes, 0)
        self.assertEqual(self.tracker.output_trace["source"], "prediction")

    def test_pending_recovery_is_used_in_next_fragment_search(self):
        self.tracker.last_observed_at = 0.0
        blank = np.full((220, 300, 3), 65, np.uint8)
        self.tracker.update(blank, 0.6)
        visible = render(self.template, (136, 100), size=(300, 220))
        first = self.tracker.update(visible, 0.65)
        self.assertNotEqual(first.status, "observed")
        self.assertIsNotNone(self.tracker.pending)
        pending_center = self.tracker.pending["center"].copy()
        original = self.tracker._fragment_matches
        centers = []

        def capture(fragment, angle, search_center, *args):
            centers.append(search_center.copy())
            return original(fragment, angle, search_center, *args)

        with patch.object(self.tracker, "_fragment_matches", side_effect=capture):
            self.tracker.update(visible, 0.7)
        np.testing.assert_allclose(centers[0], pending_center)
        result = self.tracker.update(visible, 0.75)
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.diagnostics["recovered"], 1)

    def test_votes_from_different_angles_never_form_one_peak(self):
        def mismatched_votes(fragment, angle, *args):
            assigned = (
                self.tracker.fragments[:2] if angle < 0 else self.tracker.fragments[2:4]
            )
            if fragment.fragment_id not in {item.fragment_id for item in assigned}:
                return []
            offset = rotate_points(fragment.offset[None, :], angle)[0]
            return [
                Vote(
                    fragment.fragment_id,
                    self.center,
                    0.95,
                    tuple(np.asarray(self.center) + offset),
                    tuple(offset),
                )
            ]

        with patch.object(
            self.tracker, "_fragment_matches", side_effect=mismatched_votes
        ):
            result = self.tracker.update(np.full((220, 300, 3), 65, np.uint8), 0.05)
        self.assertNotEqual(result.status, "observed")
        self.assertLessEqual(result.votes, 2)

    def test_rotating_shape_remains_in_local_fragment_search(self):
        for timestamp in (0.04, 0.09, 0.15, 0.23, 0.30):
            center = (120 + 20 * timestamp, 100 + 8 * timestamp)
            result = self.tracker.update(
                render(self.template, center, 45 * timestamp, (300, 220)), timestamp
            )
            self.assertEqual(result.status, "observed")
            self.assertLess(math.dist(result.center, center), 4)

    def test_smooth_textured_negative_has_no_target_votes(self):
        rng = np.random.default_rng(14)
        background = cv2.resize(
            rng.integers(40, 100, (12, 18), np.uint8),
            (300, 220),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.repeat(background[:, :, None], 3, axis=2)
        result = self.tracker.update(frame, 0.05)
        self.assertNotEqual(result.status, "observed")

    def test_matching_grid_four_extracts_boundary_fragments(self):
        tracker = OriginalVotingTracker(grid_size=4)
        tracker.initialize(make_template("circle"), self.center)
        self.assertFalse(
            {5, 6, 9, 10} & {fragment.fragment_id for fragment in tracker.fragments}
        )
        self.assertGreaterEqual(tracker.fragment_count, 3)

    def test_circle_rotation_is_unobservable_and_reset_discards_votes(self):
        template = make_template("circle")
        self.tracker.initialize(template, self.center)
        result = self.tracker.update(render(template, (122, 100), 30, (300, 220)), 0.1)
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.angular_velocity_dps, 0.0)
        self.tracker.reset()
        self.assertFalse(self.tracker.fragments)
        self.assertFalse(self.tracker.cache)
        self.assertFalse(self.tracker.match_records)


class GuardedVotingTrackerTest(unittest.TestCase):
    def setUp(self):
        self.template = make_template("irregular")
        self.center = (120.0, 100.0)
        self.tracker = GuardedVotingTracker()
        self.tracker.initialize(self.template, self.center)

    def test_trimmed_response_matches_full_kernel_at_all_locations(self):
        frame = render(self.template, (123, 102), 8, (300, 220))
        valid = np.ones(frame.shape[:2], bool)
        distances, _ = self.tracker.geometry._edge_maps(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), valid
        )
        features = np.exp(-0.5 * (distances / 1.8) ** 2).astype(np.float32)
        zeros = np.zeros(valid.shape, np.float32)
        for fragment in self.tracker.fragments:
            original = self.tracker._response_maps(
                fragment,
                8.0,
                np.array(self.center),
                22.0,
                features,
                valid,
                zeros,
                zeros,
            )
            trimmed = self.tracker._response_maps(
                fragment,
                8.0,
                np.array(self.center),
                22.0,
                features,
                valid,
                zeros,
                zeros,
                trim=True,
            )
            for index in range(4):
                np.testing.assert_allclose(original[index], trimmed[index], atol=2e-5)
            np.testing.assert_array_equal(original[4], trimmed[4])
            np.testing.assert_array_equal(original[5], trimmed[5])
            np.testing.assert_array_equal(original[6], trimmed[6])

    def test_guarded_centers_still_come_from_fragment_matches(self):
        with (
            patch.object(
                OptimizedFragmentTracker,
                "_centers",
                side_effect=AssertionError("forbidden"),
            ),
            patch.object(
                OptimizedFragmentTracker,
                "_score",
                side_effect=AssertionError("forbidden"),
            ),
        ):
            result = self.tracker.update(
                render(self.template, (122, 100), 2, (300, 220)), 0.05
            )
        self.assertEqual(result.status, "observed")
        sources = self.tracker.winning_records
        weights = np.array(
            [record["quality"] * record["fragment_weight"] for record in sources]
        )
        center = np.average(
            [record["center"] for record in sources], axis=0, weights=weights
        )
        np.testing.assert_allclose(
            center, self.tracker.output_trace["measured_center"], atol=1e-5
        )
        for record in self.tracker.match_records:
            np.testing.assert_allclose(
                np.array(record["matched_point"]) - record["rotated_offset"],
                record["center"],
                atol=1e-5,
            )
        self.assertGreaterEqual(self.tracker.calibration_samples, 1)

    def test_single_false_peak_does_not_change_official_state(self):
        sources = [
            Vote(
                i,
                (150.0, 100.0),
                0.95,
                (150.0 + i * 10, 100.0),
                (i * 10.0, 20.0),
                fragment_weight=0.8,
            )
            for i in range(4)
        ]
        peak = type("Peak", (), {"center": (150.0, 100.0), "quality": 0.95})()
        best = (1.0, peak, 0.0, sources, 0.3, 0.1)
        accepted, reason = self.tracker._submission_gate(
            best, np.array(self.center), 0.0, 0.04, 0.04, 0.04, {}
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "trajectory_submission_pending")
        np.testing.assert_allclose(self.tracker.center, self.center)
        self.assertEqual(self.tracker.angular_velocity, 0.0)
        original = self.tracker._fragment_matches
        targets = []

        def capture(fragment, angle, search_center, *args):
            targets.append(search_center.copy())
            return original(fragment, angle, search_center, *args)

        with patch.object(self.tracker, "_fragment_matches", side_effect=capture):
            self.tracker.update(np.full((220, 300, 3), 65, np.uint8), 0.08)
        np.testing.assert_allclose(targets[0], (150, 100))

    def test_consistent_real_turn_can_be_confirmed(self):
        sources = [
            Vote(
                i,
                (140.0, 100.0),
                0.95,
                (140.0 + i * 10, 100.0),
                (i * 10.0, 20.0),
                fragment_weight=0.8,
            )
            for i in range(4)
        ]
        peak = type("Peak", (), {"center": (140.0, 100.0), "quality": 0.95})()
        best = (1.0, peak, 0.0, sources, 0.3, 0.1)
        for timestamp in (0.04, 0.08, 0.12):
            accepted, reason = self.tracker._submission_gate(
                best, np.array(self.center), 0.0, timestamp, 0.04, timestamp, {}
            )
        self.assertTrue(accepted)
        self.assertEqual(reason, "confirmed_trajectory_submission")
        self.assertIsNone(self.tracker.transition)

    def test_moving_transition_learns_velocity_before_confirmation(self):
        velocity = np.array((340.0, 30.0))
        first = np.array((140.0, 100.0))
        for index, timestamp in enumerate((0.04, 0.075, 0.12)):
            center = first + velocity * (timestamp - 0.04)
            angle = 20 + 160 * (timestamp - 0.04)
            sources = [Vote(i, tuple(center), 0.95) for i in range(4)]
            peak = type("Peak", (), {"center": tuple(center), "quality": 0.95})()
            accepted, reason = self.tracker._submission_gate(
                (1.0, peak, angle, sources, 0.3, 0.1),
                np.array(self.center),
                0.0,
                timestamp,
                0.04,
                timestamp,
                {},
            )
            np.testing.assert_allclose(self.tracker.center, self.center)
            self.assertEqual(self.tracker.angular_velocity, 0.0)
            if index < 2:
                self.assertFalse(accepted)
                self.assertEqual(reason, "trajectory_submission_pending")
            if index == 1:
                np.testing.assert_allclose(
                    self.tracker.transition["velocity"], velocity
                )
                self.assertAlmostEqual(
                    self.tracker.transition["angular_velocity"], 160.0
                )
                search_center, search_angle = self.tracker._search_prediction(
                    np.array(self.center), 0.0, 0.045, 0.12
                )
                np.testing.assert_allclose(search_center, first + velocity * 0.08)
                self.assertAlmostEqual(search_angle, 32.8)
        self.assertTrue(accepted)
        self.assertEqual(reason, "confirmed_trajectory_submission")

    def test_transition_bootstrap_rejects_excessive_speed_or_inconsistent_third_sample(
        self,
    ):
        for positions in ((140.0, 200.0, 214.0), (140.0, 154.0, 140.0)):
            with self.subTest(positions=positions):
                self.tracker.reset()
                self.tracker.initialize(self.template, self.center)
                for x, timestamp in zip(positions, (0.04, 0.08, 0.12)):
                    sources = [Vote(i, (x, 100.0), 0.95) for i in range(4)]
                    peak = type("Peak", (), {"center": (x, 100.0), "quality": 0.95})()
                    accepted, _ = self.tracker._submission_gate(
                        (1.0, peak, 0.0, sources, 0.3, 0.1),
                        np.array(self.center),
                        0.0,
                        timestamp,
                        0.04,
                        timestamp,
                        {},
                    )
                    self.assertFalse(accepted)
                np.testing.assert_allclose(self.tracker.center, self.center)
                self.assertLess(self.tracker.transition["frames"], 3)

    def test_boundary_motion_bootstraps_and_survives_direction_change(self):
        sequence = SyntheticSequence("boundary", 3)
        self.tracker.initialize(sequence.template, sequence.initial_center)
        observed = []
        for sample in sequence:
            if sample.timestamp == 0:
                continue
            result = self.tracker.update(sample.frame, sample.timestamp)
            if result.status == "observed":
                observed.append(
                    (sample.timestamp, math.dist(result.center, sample.truth_center))
                )
                self.assertTrue(self.tracker.winning_records)
                self.assertEqual(self.tracker.output_trace["source"], "fragment_votes")
        self.assertGreater(len(observed), 35)
        self.assertLess(observed[0][0], 0.2)
        self.assertTrue(any(timestamp > 2 for timestamp, _ in observed))
        self.assertLess(float(np.mean([error for _, error in observed])), 3.0)

    def test_fragment_weights_retain_raw_quality_and_distinctiveness(self):
        frame = render(self.template, self.center, size=(300, 220))
        result = self.tracker.update(frame, 0.05)
        self.assertEqual(result.status, "observed")
        self.assertTrue(
            all(
                record["raw_quality"] == record["quality"]
                for record in self.tracker.match_records
            )
        )
        self.assertTrue(
            all(
                0 < record["fragment_weight"] <= 1
                for record in self.tracker.match_records
            )
        )
        self.assertTrue(
            all(record["peak_margin"] >= 0 for record in self.tracker.match_records)
        )

    def test_geometry_association_does_not_expand_after_selecting_sources(self):
        votes = [
            Vote(
                i,
                (120.0 + dx, 100.0),
                0.95,
                (120.0 + dx, 100.0),
                (i * 15.0, 20.0),
                fragment_weight=0.9,
            )
            for i, dx in enumerate((-3.0, 0.0, 3.0, 7.0))
        ]
        peaks = self.tracker._associated_peaks(votes, 3.0)
        for peak, sources in peaks:
            self.assertTrue(
                all(math.dist(vote.center, peak.center) <= 3.001 for vote in sources)
            )
            self.assertEqual(len(sources), len({vote.fragment_id for vote in sources}))

    def test_rejected_recovery_is_not_counted_before_official_submission(self):
        blank = np.full((220, 300, 3), 65, np.uint8)
        self.tracker.update(blank, 0.6)
        visible = render(self.template, self.center, size=(300, 220))
        with patch.object(
            self.tracker,
            "_submission_gate",
            return_value=(False, "trajectory_submission_pending"),
        ):
            for timestamp in (0.65, 0.70, 0.75, 0.80):
                result = self.tracker.update(visible, timestamp)
                self.assertNotEqual(result.status, "observed")
                self.assertEqual(result.diagnostics["recovered"], 0)
        self.assertEqual(self.tracker.recovery_count, 0)

    def test_same_matched_edge_cannot_count_as_multiple_independent_fragments(self):
        votes = [
            Vote(
                i,
                (120.0 + i * 0.2, 100.0),
                0.95,
                (140.0, 110.0),
                (20.0, 10.0),
                fragment_weight=0.9,
            )
            for i in range(4)
        ]
        peaks = self.tracker._associated_peaks(votes, 3.0)
        self.assertTrue(all(len(sources) == 1 for _, sources in peaks))

    def test_expansion_fit_is_not_reported_as_rotation(self):
        self.tracker.flow_previous = np.full((220, 300), 65, np.uint8)
        self.tracker.flow_previous_valid = np.ones((220, 300), bool)
        self.tracker.motion.affine = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        xs, ys = np.meshgrid(np.arange(95, 146, 10), np.arange(75, 126, 10))
        source = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)
        matrix = np.array(((1.10, 0.0, -12.0), (0.0, 1.10, -10.0)))
        target = (source @ matrix[:, :2].T + matrix[:, 2]).astype(np.float32)
        status = np.ones((len(source), 1), np.uint8)
        with (
            patch("cv2.goodFeaturesToTrack", return_value=source.reshape(-1, 1, 2)),
            patch(
                "cv2.calcOpticalFlowPyrLK",
                side_effect=[
                    (target.reshape(-1, 1, 2), status, None),
                    (source.reshape(-1, 1, 2), status, None),
                ],
            ),
            patch("cv2.estimateAffine2D", return_value=(matrix, status)),
        ):
            rotation, reliability = self.tracker._rotation_map(
                self.tracker.flow_previous, np.ones((220, 300), bool)
            )
        self.assertGreater(reliability, 0.25)
        self.assertAlmostEqual(self.tracker.local_expansion, 0.1, places=5)
        self.assertAlmostEqual(self.tracker.local_spin, 0.0, places=5)
        self.assertEqual(float(rotation.sum()), 0.0)

    def test_reliable_shear_cannot_confirm_suspicious_transition(self):
        sources = [
            Vote(
                i,
                (140.0, 100.0),
                0.95,
                (140.0 + i * 10.0, 100.0),
                (i * 10.0, 20.0),
                fragment_weight=0.8,
            )
            for i in range(4)
        ]
        peak = type("Peak", (), {"center": (140.0, 100.0), "quality": 0.95})()
        best = (1.0, peak, 0.0, sources, 0.3, 0.1)
        self.tracker.local_motion_quality = 0.9
        self.tracker.local_shear = 0.15
        self.tracker.local_spin = 0.01
        for timestamp in (1.0, 1.04, 1.08):
            accepted, reason = self.tracker._submission_gate(
                best,
                np.array(self.center),
                0.0,
                timestamp,
                0.04,
                timestamp,
                {"background_status": "accepted"},
            )
        self.assertFalse(accepted)
        self.assertEqual(reason, "unreliable_transition_motion")
        np.testing.assert_allclose(self.tracker.center, self.center)

    def test_repeated_equal_peaks_reduce_weight_without_changing_raw_score(self):
        size = 41
        yy, xx = np.indices((size, size))
        shape = np.zeros((size, size), np.float32)
        shape[10, 10] = shape[25, 25] = 0.95
        zeros = np.zeros(shape.shape, np.float32)
        response = (
            shape,
            zeros,
            zeros,
            np.ones(shape.shape, np.float32),
            np.ones(shape.shape, bool),
            xx + 100,
            yy + 80,
            np.zeros(2),
        )
        fragment = self.tracker.fragments[0]
        with patch.object(self.tracker, "_response_maps", return_value=response):
            ambiguous = self.tracker._fragment_matches(
                fragment,
                0.0,
                np.array(self.center),
                40.0,
                np.empty(0),
                np.ones(shape.shape, bool),
                zeros,
                zeros,
                False,
            )
        self.assertAlmostEqual(ambiguous[0].peak_margin, 0.0, places=5)
        unique_shape = shape.copy()
        unique_shape[25, 25] = 0.72
        unique_response = (unique_shape, *response[1:])
        with patch.object(self.tracker, "_response_maps", return_value=unique_response):
            unique = self.tracker._fragment_matches(
                fragment,
                0.0,
                np.array(self.center),
                40.0,
                np.empty(0),
                np.ones(shape.shape, bool),
                zeros,
                zeros,
                False,
            )
        self.assertAlmostEqual(ambiguous[0].raw_score, unique[0].raw_score, places=5)
        self.assertLess(ambiguous[0].fragment_weight, unique[0].fragment_weight)

    def test_trimmed_response_remains_equivalent_at_frame_boundary(self):
        valid = np.ones((220, 300), bool)
        valid[:, :9] = False
        frame = render(self.template, (15, 100), 14, (300, 220))
        distances, _ = self.tracker.geometry._edge_maps(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), valid
        )
        features = np.exp(-0.5 * (distances / 1.8) ** 2).astype(np.float32)
        features[:, ~valid] = 0
        zeros = np.zeros(valid.shape, np.float32)
        fragment = self.tracker.fragments[0]
        full = self.tracker._response_maps(
            fragment, 14.0, np.array((15.0, 100.0)), 18.0, features, valid, zeros, zeros
        )
        trimmed = self.tracker._response_maps(
            fragment,
            14.0,
            np.array((15.0, 100.0)),
            18.0,
            features,
            valid,
            zeros,
            zeros,
            trim=True,
        )
        usable = full[3] >= 0.45
        np.testing.assert_allclose(full[0][usable], trimmed[0][usable], atol=2e-5)
        np.testing.assert_array_equal(full[4], trimmed[4])
        np.testing.assert_array_equal(full[5], trimmed[5])
        np.testing.assert_array_equal(full[6], trimmed[6])

    def test_empty_motion_does_not_claim_rotation_confidence(self):
        frame = np.full((220, 300, 3), 65, np.uint8)
        result = self.tracker.update(frame, 0.05)
        self.assertEqual(result.diagnostics["local_motion_reliability"], 0.0)
        self.assertEqual(result.diagnostics["rotation_flow_confidence"], 0.0)


class MotionFeedbackTrackerTest(unittest.TestCase):
    def setUp(self):
        self.sequence = SyntheticSequence("clean", 3)
        self.tracker = MotionFeedbackTracker()
        self.tracker.initialize(
            self.sequence.template,
            self.sequence.initial_center,
            initial_frame=self.sequence.sample(0).frame,
        )

    def motion_points(self, matrix=None, shift=(3.0, 1.0)):
        offsets = np.array(
            [(x, y) for x in (-20, -10, 0, 10, 20) for y in (-20, -10, 0, 10, 20)],
            np.float32,
        )
        source = offsets + self.tracker.image_center
        matrix = np.eye(2) if matrix is None else np.asarray(matrix)
        target = offsets @ matrix.T + self.tracker.image_center + shift
        self.tracker.motion.affine = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        return source.astype(np.float32), target.astype(np.float32)

    def test_independent_motion_tracker_never_calls_fragment_chain(self):
        with (
            patch.object(
                OriginalVotingTracker, "update", side_effect=AssertionError("forbidden")
            ),
            patch.object(
                OptimizedFragmentTracker,
                "_centers",
                side_effect=AssertionError("forbidden"),
            ),
            patch(
                "tools.fragment_voting_experiment.aggregate_votes",
                side_effect=AssertionError("forbidden"),
            ),
        ):
            result = self.tracker.update(self.sequence.sample(0.05).frame, 0.05)
        self.assertEqual(result.status, "observed")
        self.assertEqual(result.votes, 0)
        self.assertEqual(result.fragment_count, 0)
        self.assertEqual(self.tracker.output_trace["source"], "white_observation")

    def test_background_translation_and_rotation_do_not_create_target_motion(self):
        source, _ = self.motion_points()
        for angle in (0, 3):
            matrix = rotate_points(np.eye(2), angle).T
            affine = np.column_stack((matrix, (3.0, 1.0)))
            self.tracker.motion.affine = affine
            target = source @ affine[:, :2].T + affine[:, 2]
            hypotheses, reason = self.tracker._motion_hypotheses(
                source, target, self.tracker.image_center, 0.04, 0.02
            )
            self.assertFalse(hypotheses)
            self.assertEqual(reason, "background_motion_only")

    def test_local_translation_proposes_center_from_motion_not_centroid(self):
        source, target = self.motion_points()
        hypotheses, _ = self.tracker._motion_hypotheses(
            source, target, self.tracker.image_center, 0.04, 0.02
        )
        self.assertTrue(hypotheses)
        np.testing.assert_allclose(
            hypotheses[0]["center"], self.tracker.image_center + (3, 1), atol=0.01
        )
        self.assertGreaterEqual(hypotheses[0]["support_points"], 6)

    def test_rotation_can_move_reference_without_precise_template_pose(self):
        matrix = rotate_points(np.eye(2), 3).T
        source, target = self.motion_points(matrix, (2.0, 1.0))
        hypotheses, _ = self.tracker._motion_hypotheses(
            source, target, self.tracker.image_center, 0.04, 0.02
        )
        self.assertTrue(hypotheses)
        self.assertAlmostEqual(hypotheses[0]["rotation_degrees"], 3.0, places=2)
        np.testing.assert_allclose(
            hypotheses[0]["center"], self.tracker.image_center + (2, 1), atol=0.01
        )

    def test_expansion_and_shear_are_rejected_as_nonrigid(self):
        for matrix in (np.eye(2) * 1.10, np.array(((1.0, 0.2), (0.0, 1.0)))):
            source, target = self.motion_points(matrix)
            hypotheses, _ = self.tracker._motion_hypotheses(
                source, target, self.tracker.image_center, 0.04, 0.02
            )
            self.assertFalse(hypotheses)
            self.assertTrue(
                any(
                    h["reason"] == "nonrigid_local_motion"
                    for h in self.tracker.hypothesis_records
                )
            )

    def test_far_points_cannot_enter_local_feedback(self):
        frame = self.sequence.sample(0.05).frame
        source, _, _, _ = self.tracker._local_flow(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            np.ones(frame.shape[:2], bool),
            self.tracker.image_center,
        )
        self.assertTrue(
            np.all(
                np.linalg.norm(source - self.tracker.image_center, axis=1)
                <= self.tracker.range_factor * self.tracker.radius + 0.01
            )
        )

    def test_feedback_velocity_acceleration_and_correction_are_bounded(self):
        center = self.tracker.center.copy()
        for dt in (0.02, 0.04, 0.075):
            _, velocity, correction = self.tracker._feedback(
                center + (100, 50), center, dt
            )
            self.assertLessEqual(
                np.linalg.norm(velocity), self.tracker.max_speed + 1e-6
            )
            self.assertLessEqual(
                np.linalg.norm(velocity - self.tracker.velocity),
                self.tracker.max_acceleration * dt + 1e-6,
            )
            self.assertLessEqual(
                np.linalg.norm(correction), self.tracker.max_acceleration * dt**2 + 1e-6
            )

    def test_dead_zone_suppresses_tiny_feedback_correction(self):
        _, _, correction = self.tracker._feedback(
            self.tracker.center + (0.1, 0.1), self.tracker.center, 0.04
        )
        np.testing.assert_array_equal(correction, (0, 0))

    def test_invisible_sequence_never_accepts_background_as_observation(self):
        sequence = SyntheticSequence("invisible", 3)
        self.tracker.initialize(
            sequence.template,
            sequence.initial_center,
            initial_frame=sequence.sample(0).frame,
        )
        statuses = []
        for sample in sequence:
            if sample.timestamp == 0:
                continue
            result = self.tracker.update(sample.frame, sample.timestamp)
            if sample.visibility == 0:
                self.assertNotEqual(result.status, "observed")
                statuses.append(result.status)
        self.assertIn("predicted", statuses)
        self.assertIn("lost", statuses)

    def test_boundary_motion_and_direction_change_preserve_local_flow_anchor(self):
        sequence = SyntheticSequence("boundary", 3)
        self.tracker.initialize(
            sequence.template,
            sequence.initial_center,
            initial_frame=sequence.sample(0).frame,
        )
        errors = []
        for sample in sequence:
            if sample.timestamp == 0:
                continue
            result = self.tracker.update(sample.frame, sample.timestamp)
            if result.status == "observed":
                errors.append(math.dist(result.center, sample.truth_center))
            np.testing.assert_allclose(
                self.tracker.motion.previous_center, self.tracker.image_center
            )
        self.assertGreater(len(errors), 50)
        self.assertLess(np.mean(errors), 7)
        self.assertLess(
            math.dist(self.tracker.image_center, sequence.sample(3).truth_center), 3
        )

    def test_invalid_pointer_regions_never_supply_flow_support(self):
        frame = self.sequence.sample(0.05).frame
        invalid = np.zeros(frame.shape[:2], bool)
        invalid[130:175, 130:165] = True
        self.tracker.initialize(
            self.sequence.template,
            self.sequence.initial_center,
            initial_frame=self.sequence.sample(0).frame,
            invalid_mask=invalid,
        )
        self.tracker.update(frame, 0.05, invalid)
        for record in self.tracker.flow_records:
            for key in ("source", "target"):
                x, y = np.rint(record[key]).astype(int)
                self.assertFalse(invalid[y, x])

    def test_circle_does_not_report_observable_rotation(self):
        sequence = SyntheticSequence("circle", 3)
        self.tracker.initialize(
            sequence.template,
            sequence.initial_center,
            initial_frame=sequence.sample(0).frame,
        )
        result = self.tracker.update(sequence.sample(0.05).frame, 0.05)
        self.assertEqual(result.angular_velocity_dps, 0)
        self.assertEqual(result.diagnostics["symmetry_period_degrees"], 0)

    def test_irregular_intervals_and_reset(self):
        for timestamp in (0.02, 0.065, 0.13, 0.18, 0.29):
            result = self.tracker.update(
                self.sequence.sample(timestamp).frame, timestamp
            )
            self.assertEqual(result.status, "observed")
        self.tracker.reset()
        self.assertIsNone(self.tracker.center)
        self.assertIsNone(self.tracker.image_center)
        self.assertIsNone(self.tracker.previous_gray)
        self.assertFalse(self.tracker.flow_records)
        self.assertEqual(self.tracker.recovery_count, 0)

    def test_weak_motion_sequence_uses_actual_flow_after_white_phase(self):
        sequence = SyntheticSequence("weak_motion", 3)
        self.tracker.initialize(
            sequence.template,
            sequence.initial_center,
            initial_frame=sequence.sample(0).frame,
        )
        motion_observations = 0
        for sample in sequence:
            if sample.timestamp == 0:
                continue
            result = self.tracker.update(sample.frame, sample.timestamp)
            if sample.timestamp > 0.6 and result.status == "observed":
                self.assertEqual(self.tracker.output_trace["source"], "local_motion")
                affine = np.array(self.tracker.output_trace["affine_matrix"])
                reference = np.array(self.tracker.output_trace["reference_center"])
                np.testing.assert_allclose(
                    affine[:, :2] @ reference + affine[:, 2],
                    self.tracker.output_trace["measured_center"],
                    atol=1e-5,
                )
                motion_observations += 1
        self.assertGreater(motion_observations, 5)

    def test_fresh_background_points_cannot_replace_lost_target_identity(self):
        source, target = self.motion_points()
        self.tracker.last_timestamp = 0.5
        self.tracker.last_white_at = 0
        self.tracker.flow_persistent = np.zeros(len(source), bool)
        hypotheses, _ = self.tracker._motion_hypotheses(
            source, target, self.tracker.image_center, 0.04, 0.02
        )
        self.assertFalse(hypotheses)
        self.assertTrue(
            any(
                h["reason"] == "lost_motion_point_identity"
                for h in self.tracker.hypothesis_records
            )
        )

    def test_recovery_requires_three_new_image_motion_samples(self):
        blank = np.full((320, 480, 3), 65, np.uint8)
        self.tracker.update(blank, 0.6)
        source, target = self.motion_points()
        hypotheses, _ = self.tracker._motion_hypotheses(
            source, target, self.tracker.image_center, 0.04, 0.02
        )
        self.assertTrue(hypotheses)
        candidate = hypotheses[0]
        frame = self.sequence.sample(0.65).frame
        with (
            patch.object(
                self.tracker.motion,
                "observe",
                return_value=(
                    np.zeros(frame.shape[:2]),
                    {"background_status": "accepted", "background_fit_error": 0.02},
                ),
            ),
            patch.object(
                self.tracker,
                "_motion_hypotheses",
                return_value=([candidate], "accepted_local_motion"),
            ),
            patch.object(
                self.tracker,
                "_local_flow",
                return_value=(
                    source,
                    target,
                    np.zeros(len(source)),
                    "local_flow_ready",
                ),
            ),
            patch(
                "tools.fragment_voting_experiment.visible_template", return_value=None
            ),
        ):
            for index, timestamp in enumerate((0.65, 0.69, 0.73)):
                result = self.tracker.update(frame, timestamp)
                self.assertEqual(result.status, "observed" if index == 2 else "lost")
                self.assertEqual(self.tracker.recovery_count, 1 if index == 2 else 0)

    def test_untextured_scene_is_prediction_then_loss(self):
        blank = np.full((320, 480, 3), 65, np.uint8)
        result = self.tracker.update(blank, 0.05)
        self.assertEqual(result.status, "predicted")
        result = self.tracker.update(blank, 0.6)
        self.assertEqual(result.status, "lost")
        self.assertIsNone(result.center)


class RevisedMotionFeedbackTrackerTest(unittest.TestCase):
    def setUp(self):
        self.sequence = SyntheticSequence("clean", 3)
        self.tracker = RevisedMotionFeedbackTracker()
        self.tracker.initialize(
            self.sequence.template,
            self.sequence.initial_center,
            initial_frame=self.sequence.sample(0).frame,
        )

    def test_white_reappears_after_coast_without_permanent_detection_disable(self):
        blank = np.full((320, 480, 3), 65, np.uint8)
        self.tracker.update(blank, 0.6)
        for index, time in enumerate((0.65, 0.70, 0.75)):
            frame = self.sequence.sample(0).frame
            result = self.tracker.update(frame, time)
            if index < 2:
                self.assertEqual(result.status, "lost")
                self.assertEqual(self.tracker.recovery_count, 0)
            else:
                self.assertEqual(result.status, "observed")
                self.assertEqual(self.tracker.recovery_count, 1)
                self.assertEqual(
                    self.tracker.output_trace["source"], "white_observation"
                )
                self.assertGreater(len(self.tracker.tracked_points), 5)

    def test_white_reseeds_lost_points_using_only_valid_current_pixels(self):
        self.tracker.tracked_points = np.empty((0, 2), np.float32)
        self.tracker.point_ids = np.empty(0, int)
        self.tracker.point_since = np.empty(0)
        result = self.tracker.update(self.sequence.sample(0.05).frame, 0.05)
        self.assertEqual(result.status, "observed")
        self.assertGreater(len(self.tracker.tracked_points), 5)
        for x, y in self.tracker.tracked_points.astype(int):
            self.assertTrue(self.tracker.previous_valid[y, x])

    def test_pointer_split_white_keeps_bbox_reference_not_visible_centroid(self):
        template = make_template("irregular", 40)
        frame = render(template, (200, 150), size=(400, 300))
        self.tracker.initialize(template, (200, 150), initial_frame=frame)
        valid = np.ones((300, 400), bool)
        valid[:, 205:218] = False
        best, reason = self.tracker._white_observation(
            frame, valid, np.array((200.0, 150.0)), 0.1
        )
        self.assertIsNotNone(best)
        self.assertEqual(reason, "white_accepted")
        self.assertTrue(best["partial"])
        self.assertLess(math.dist(best["center"], (200, 150)), 6)
        self.assertFalse(np.any(best["pixels"] & ~valid))

    def test_candidate_points_need_multiple_consistent_samples_to_promote(self):
        target = np.array(((150, 150),), np.float32)
        best = {"support_indices": [0], "needs_confirmation": False}
        for frames, time in ((0, 0.03), (1, 0.06), (2, 0.10)):
            self.tracker.flow_metadata = [
                {"point_id": 999, "since": 0.0, "frames": frames, "trusted": False}
            ]
            self.tracker._update_points(target, best, time, True)
            self.assertEqual(len(self.tracker.tracked_points), 1 if frames == 2 else 0)
            self.assertEqual(self.tracker.point_promotions, 1 if frames == 2 else 0)

    def test_rejected_candidate_does_not_promote_or_move_official_center(self):
        original = self.tracker.center.copy()
        self.tracker.flow_metadata = [
            {"point_id": 999, "since": 0.0, "frames": 5, "trusted": False}
        ]
        self.tracker._update_points(
            np.array(((170, 150),), np.float32), {"support_indices": [0]}, 0.2, False
        )
        self.assertEqual(len(self.tracker.tracked_points), 0)
        np.testing.assert_allclose(self.tracker.center, original)

    def test_blank_frames_remain_loss_and_cannot_create_motion_recovery(self):
        blank = np.full((320, 480, 3), 65, np.uint8)
        for time in (0.1, 0.3, 0.6, 0.8):
            result = self.tracker.update(blank, time)
        self.assertEqual(result.status, "lost")
        self.assertEqual(self.tracker.recovery_count, 0)
        self.assertIsNone(result.center)

    def test_sampling_rates_use_accumulated_motion_without_lost_run(self):
        for fps in (24, 60, 120):
            with self.subTest(fps=fps):
                sequence = SyntheticSequence("weak_motion", 3, fps=fps)
                tracker = RevisedMotionFeedbackTracker()
                tracker.initialize(
                    sequence.template,
                    sequence.initial_center,
                    initial_frame=sequence.sample(0).frame,
                )
                states = []
                errors = []
                for sample in sequence:
                    if not sample.timestamp:
                        continue
                    result = tracker.update(sample.frame, sample.timestamp)
                    states.append(result.status)
                    if result.center is not None:
                        errors.append(math.dist(result.center, sample.truth_center))
                self.assertNotIn("lost", states)
                self.assertLess(np.mean(errors), 10)
                self.assertGreater(sum(state == "observed" for state in states), 20)

    def test_pause_is_not_reported_as_target_observation_from_background(self):
        sequence = SyntheticSequence("weak_motion", 3)
        frame = sequence.sample(0.8).frame
        for time in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            self.tracker.update(sequence.sample(time).frame, time)
        for time in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1):
            result = self.tracker.update(frame, time)
            self.assertNotEqual(result.status, "observed")
        self.assertEqual(self.tracker.output_trace["source"], "prediction")

    def test_boundary_roundtrip_and_center_evidence_provenance(self):
        sequence = SyntheticSequence("boundary", 3)
        self.tracker.initialize(
            sequence.template,
            sequence.initial_center,
            initial_frame=sequence.sample(0).frame,
        )
        observed = 0
        with (
            patch.object(
                OriginalVotingTracker, "update", side_effect=AssertionError("forbidden")
            ),
            patch.object(
                OptimizedFragmentTracker,
                "_centers",
                side_effect=AssertionError("forbidden"),
            ),
        ):
            for sample in sequence:
                if not sample.timestamp:
                    continue
                result = self.tracker.update(sample.frame, sample.timestamp)
                self.assertEqual(result.votes, 0)
                if result.status == "observed":
                    observed += 1
                    trace = self.tracker.output_trace
                    if trace["source"] == "local_motion":
                        matrix = np.array(trace["affine_matrix"])
                        np.testing.assert_allclose(
                            matrix[:, :2] @ trace["reference_center"] + matrix[:, 2],
                            trace["measured_center"],
                            atol=1e-5,
                        )
        self.assertGreater(observed, 50)
        self.assertLess(
            math.dist(self.tracker.image_center, sequence.sample(3).truth_center), 5
        )

    def test_overlapping_white_distractor_does_not_drag_center_to_union(self):
        sequence = SyntheticSequence("distractors", 3)
        self.tracker.initialize(
            sequence.template,
            sequence.initial_center,
            initial_frame=sequence.sample(0).frame,
        )
        errors = []
        for sample in sequence:
            if not sample.timestamp:
                continue
            result = self.tracker.update(sample.frame, sample.timestamp)
            if result.center is not None:
                errors.append(math.dist(result.center, sample.truth_center))
        self.assertLess(np.mean(errors), 5)
        self.assertLess(
            math.dist(self.tracker.image_center, sequence.sample(3).truth_center), 5
        )

    def test_consistent_pointer_occlusion_does_not_restart_confirmation_each_frame(
        self,
    ):
        template = make_template("irregular", 40)
        frame = render(template, (200, 150), size=(400, 300))
        self.tracker.initialize(template, (200, 150), initial_frame=frame)
        invalid = np.zeros((300, 400), bool)
        invalid[:, 205:218] = True
        observed = 0
        for time in (0.03, 0.06, 0.09, 0.12, 0.15):
            result = self.tracker.update(frame, time, invalid)
            observed += result.status == "observed"
        self.assertGreaterEqual(observed, 4)
        self.assertIsNone(self.tracker.pending)

    def test_unconfirmed_motion_recovery_uses_candidate_ids_not_old_trusted_ids(self):
        best = {
            "center": np.array((170.0, 150.0)),
            "kind": "local_motion",
            "support_indices": [0, 1, 2],
        }
        original = self.tracker.image_center.copy()
        self.tracker.flow_metadata = [
            {"point_id": i + 100, "trusted": False} for i in range(3)
        ]
        for time in (0.60, 0.65):
            self.assertFalse(self.tracker._confirm(best, time))
            np.testing.assert_allclose(self.tracker.image_center, original)
        self.assertTrue(self.tracker._confirm(best, 0.70))
        np.testing.assert_allclose(self.tracker.image_center, original)


class EvaluationIsolationIntegrationTest(unittest.TestCase):
    def test_changed_proxy_and_annotation_do_not_change_tracking(self):
        self._check_isolation("optimized")

    def test_revised_predictions_are_independent_of_proxy_and_annotation(self):
        self._check_isolation("revised")

    def test_original_voting_is_independent_of_proxy_and_annotation(self):
        self._check_isolation("original_voting")

    def test_guarded_voting_is_independent_of_proxy_and_annotation(self):
        self._check_isolation("original_guarded")

    def test_motion_feedback_is_independent_of_proxy_and_annotation(self):
        self._check_isolation("motion_feedback")

    def test_motion_revised_is_independent_of_proxy_and_annotation(self):
        self._check_isolation("motion_revised")

    def _check_isolation(self, variant):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            video = videos / "测谎录屏测试.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (300, 220)
            )
            self.assertTrue(writer.isOpened())
            template = make_template("irregular")
            for i in range(6):
                writer.write(render(template, (150 + i, 120), 2 * i, (300, 220)))
            writer.release()
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "recordings": [
                            {
                                "filename": video.name,
                                "roi": [0, 0, 300, 220],
                                "active_start": 0,
                                "active_end": 1,
                            }
                        ]
                    }
                )
            )
            prediction_rows = []
            for index, point in enumerate(((10, 10), (270, 190))):
                annotations = root / f"annotations_{index}.csv"
                annotations.write_text(
                    f"video,frame,x,y,phase\n{video.name},2,{point[0]},{point[1]},visible\n"
                )
                output = root / f"output_{index}"
                args = argparse.Namespace(
                    recording_config=config,
                    recordings_dir=videos,
                    recording=[],
                    output_dir=output,
                    no_video=True,
                    variant=variant,
                    annotations=annotations,
                )
                with (
                    patch(
                        "tools.lie_detector_replay.cursor_ground_truth",
                        return_value=point,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    evaluate_recordings(args)
                with (output / "frames.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                prediction_rows.append(
                    [
                        (row["status"], row["center_x"], row["center_y"], row["votes"])
                        for row in rows
                    ]
                )
                report = json.loads((output / "summary.json").read_text())
                self.assertEqual(report["aggregate"]["independent_truth_frames"], 1)
                self.assertEqual(len(rows), 6)
                signature = recording_signature(
                    video,
                    {"roi": [0, 0, 300, 220], "active_start": 0, "active_end": 1},
                    variant,
                    annotations,
                    False,
                )
                resumed = resumable_recording(output, video, signature)
                self.assertIsNotNone(resumed)
                args.resume = True
                with (
                    patch(
                        "tools.fragment_voting_experiment.make_tracker",
                        side_effect=AssertionError(
                            "completed run should not be replayed"
                        ),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    evaluate_recordings(args)
                self.assertTrue(
                    json.loads((output / "summary.json").read_text())[
                        "evaluation_complete"
                    ]
                )
                changed = signature.copy()
                changed["input_size"] += 1
                with self.assertRaises(ValueError):
                    resumable_recording(output, video, changed)
            self.assertEqual(prediction_rows[0], prediction_rows[1])


if __name__ == "__main__":
    unittest.main()
