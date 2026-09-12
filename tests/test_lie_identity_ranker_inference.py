"""Pure-Python LightGBM inference + ranker gating unit tests."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieIdentityRanker import (  # noqa: E402
    AVAILABLE_FEATURE_NAMES,
    FLAT_FEATURE_NAMES,
    LightGbmTextModel,
    SwitchMotionFeatureHistory,
    TrajectoryIdentityRanker,
    aggregate_history,
    extract_similarity_features,
    extract_frame_features,
)
from src.engine.LieSwitchEventModel import (  # noqa: E402
    SWITCH_EVENT_FEATURE_NAMES,
    SWITCH_TRACK_PREFLOW_FEATURE_NAMES,
    SwitchEventModel,
    build_switch_event_features,
)

MODEL_PATH = PROJECT_ROOT / "models" / "lie_identity_ranker.txt"
VAL_CSV = PROJECT_ROOT / "ml" / "lie_identity_dataset" / "val.csv"


@dataclass
class _View:
    track_id: int
    center: tuple[float, float]
    radius: float = 30.0
    role: str = "bg"
    orientation: float | None = 0.0
    lost_frames: int = 0
    translation_residual: float = 0.0
    rotation_residual: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    real_score: float = 0.0
    rotation_score: float = 0.0
    direction_score: float = 0.0
    speed_score: float = 0.0
    rigidity_score: float = 0.0
    motion_reliability: float = 0.5
    visible_streak: int = 10
    angular_velocity: float = 0.0
    appearance_distance: float = 0.1
    association_quality: float = 0.5
    yolo_confidence: float = 0.5
    predicted_only: bool = False
    bg_certified: bool = False
    ranker_score: float = 0.0


class LightGbmTextModelTest(unittest.TestCase):
    def test_loads_expected_features(self) -> None:
        model = LightGbmTextModel.load(MODEL_PATH)
        self.assertTrue(model.trees)
        self.assertLessEqual(set(model.feature_names), set(FLAT_FEATURE_NAMES))

    def test_matches_lightgbm_predictions(self) -> None:
        try:
            import csv

            import lightgbm as lgb
        except OSError:  # pragma: no cover - libomp missing
            self.skipTest("lightgbm runtime unavailable")
        except ImportError:  # pragma: no cover
            self.skipTest("lightgbm not installed")
        model = LightGbmTextModel.load(MODEL_PATH)
        booster = lgb.Booster(model_file=str(MODEL_PATH))
        with VAL_CSV.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[:150]
        for row in rows:
            vector = [float(row[name]) for name in model.feature_names]
            expected = float(booster.predict([vector])[0])
            self.assertAlmostEqual(model.predict(vector), expected, places=6)


class SwitchEventModelTest(unittest.TestCase):
    def test_feature_builder_has_complete_finite_schema(self) -> None:
        event = {
            "ranker_switch_delta": 3.0,
            "ranker_current_rank": 2,
            "candidate_count": 15,
            "current_radius_norm": 0.10,
            "challenger_radius_norm": 0.13,
        }
        features = build_switch_event_features(event)

        self.assertEqual(set(features), set(SWITCH_EVENT_FEATURE_NAMES))
        self.assertAlmostEqual(features["delta_radius_norm"], 0.03)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertIn("delta_optical_spin_evidence", features)
        self.assertIn("delta_rotation_confidence", features)
        self.assertIn("delta_peer_rotation_residual_norm", features)
        self.assertIn("delta_preflow_group_residual_norm", features)
        self.assertTrue(SWITCH_TRACK_PREFLOW_FEATURE_NAMES)

    def test_portable_switch_model_matches_lightgbm_probability(self) -> None:
        model_path = PROJECT_ROOT / "models" / "lie_switch_event_model.txt"
        if not model_path.is_file():
            self.skipTest("switch-event model has not been trained")
        try:
            import lightgbm as lgb
        except (ImportError, OSError):  # pragma: no cover
            self.skipTest("lightgbm runtime unavailable")
        portable = SwitchEventModel(model_path)
        self.assertFalse(portable.needs_preflow_features)
        native = lgb.Booster(model_file=str(model_path))
        event = {
            "ranker_switch_delta": 2.4,
            "ranker_top_margin": 2.1,
            "ranker_current_rank": 3,
            "candidate_count": 14,
            "ranker_disagreement_streak": 4,
            "ranker_evidence_after": 3.0,
            "path_support": 1.0,
            "current_visible_streak_norm": 0.8,
            "challenger_visible_streak_norm": 0.4,
        }
        features = build_switch_event_features(event)
        vector = [features[name] for name in portable.model.feature_names]

        self.assertAlmostEqual(
            portable.predict_probability(event),
            float(native.predict([vector])[0]),
            places=6,
        )


class FeatureAndHistoryTest(unittest.TestCase):
    def test_switch_motion_history_keeps_track_positions_frame_aligned(self) -> None:
        history = SwitchMotionFeatureHistory()
        frame = np.zeros((160, 240), dtype=np.uint8)
        cv2.rectangle(frame, (65, 65), (95, 95), 220, 3)
        views = [_View(1, (80.0, 80.0), radius=35.0)]
        history.update(views, gray_frame=frame)
        history.update(views, gray_frame=frame.copy())

        features = history.features([1])

        self.assertEqual(set(features[1]), {
            "optical_spin_abs_norm",
            "optical_spin_confidence",
            "optical_spin_evidence",
            "multilag_spin_abs_norm",
            "multilag_spin_confidence",
            "multilag_spin_consistency",
            "spin_estimator_agreement",
            "spin_joint_confidence",
        })
        self.assertTrue(all(np.isfinite(value) for value in features[1].values()))

    def test_flat_row_covers_all_features(self) -> None:
        views = [_View(1, (100.0, 100.0)), _View(2, (300.0, 200.0))]
        frame = extract_frame_features(
            views, target_id=1, frame_width=640, frame_height=480
        )
        flat = aggregate_history([frame[1]], history_size=16)
        for name in FLAT_FEATURE_NAMES:
            self.assertIn(name, flat)
        self.assertAlmostEqual(flat["history_length_norm"], 1.0 / 16.0)

    def test_similarity_motion_marks_relative_outlier(self) -> None:
        histories = defaultdict(lambda: deque(maxlen=16))
        previous = [(100.0, 100.0), (300.0, 100.0), (100.0, 300.0), (300.0, 300.0)]
        current = [(105.0, 98.0), (305.0, 98.0), (105.0, 298.0), (345.0, 320.0)]
        for track_id, (before, after) in enumerate(zip(previous, current), 1):
            histories[track_id].append(
                {"x_norm": before[0] / 640.0, "y_norm": before[1] / 480.0}
            )
            histories[track_id].append(
                {"x_norm": after[0] / 640.0, "y_norm": after[1] / 480.0}
            )
        features = extract_similarity_features(
            [1, 2, 3, 4], histories, frame_width=640, frame_height=480
        )
        self.assertEqual(set(features), {1, 2, 3, 4})
        self.assertGreater(
            features[4]["global_similarity_residual_norm"],
            max(
                features[index]["global_similarity_residual_norm"]
                for index in (1, 2, 3)
            ),
        )
        self.assertLessEqual(
            set(features[4]),
            set(AVAILABLE_FEATURE_NAMES),
        )


class RankerVotingTest(unittest.TestCase):
    def test_requires_consecutive_votes_before_switch(self) -> None:
        ranker = TrajectoryIdentityRanker(MODEL_PATH)

        # Force a deterministic winner regardless of learned weights.
        def _fake_update(
            views, *, target_id, frame_width, frame_height, gray_frame=None
        ):
            return {view.track_id: (5.0 if view.track_id == 2 else 0.0) for view in views}

        ranker.update = _fake_update  # type: ignore[assignment]

        import src.engine.LieDetectorTracker as tracker_mod

        tracker = object.__new__(tracker_mod.LieDetectorTracker)
        tracker.identity_ranker = ranker
        tracker.identity_ranker_min_margin = 0.5
        tracker.frame_size = (640, 480)
        tracker.target_id = 1
        tracker.multi_hypothesis_identity_enabled = False
        tracker.ranker_candidate_id = None
        tracker.ranker_candidate_streak = 0
        tracker.ranker_margin = 0.0
        tracker.ranker_top_id = None
        tracker.ranker_current_rank = 0
        tracker.ranker_current_score = 0.0
        tracker.ranker_switch_delta = 0.0
        tracker.ranker_evidence = 0.0
        tracker.motion_corroborated = False
        tracker.state_aware_ranker_enabled = False
        tracker.motion_corroboration_ranker = None

        @dataclass
        class _Track:
            id: int
            lost_frames: int = 0
            predicted_only: bool = False
            visible_streak: int = 20
            motion_reliability: float = 0.5
            state: tuple = (0.0, 0.0, 30.0)
            ranker_score: float = 0.0

        tracker.tracks = {1: _Track(1), 2: _Track(2)}
        tracker._constellation_views = lambda: [_View(1, (0, 0)), _View(2, (10, 10))]
        tracker._leaving_frame = lambda track: False

        current = tracker.tracks[1]
        # A healthy current REAL needs five consecutive high-margin votes.
        results = [
            tracker_mod.LieDetectorTracker._ranker_switch_candidate(tracker, current)
            for _ in range(5)
        ]
        self.assertTrue(all(result is None for result in results[:4]))
        self.assertIsNotNone(results[4])
        self.assertEqual(results[4].id, 2)


class IdentitySafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        import src.engine.LieDetectorTracker as tracker_mod

        self.tracker_mod = tracker_mod
        self.tracker = object.__new__(tracker_mod.LieDetectorTracker)
        self.tracker.white_seen = True
        self.tracker.white_active = False
        self.tracker.identity_switched = False
        self.tracker.identity_confidence = 0.0
        self.tracker.identity_state = "uninitialized"
        self.tracker.identity_safe = False
        self.tracker.identity_match_streak = 0
        self.tracker.identity_mismatch_streak = 0
        self.tracker.identity_ranker_unavailable_frames = 0
        self.tracker.identity_probation_frames = 0
        self.tracker.ranker_top_id = 1
        self.tracker.ranker_margin = 2.0
        self.tracker.ranker_switch_delta = 0.0

    def update(self, target, position_actionable=True) -> None:
        self.tracker_mod.LieDetectorTracker._update_identity_confidence(
            self.tracker, target, position_actionable=position_actionable
        )

    def test_holds_after_two_ranker_disagreements(self) -> None:
        target = SimpleNamespace(id=1, predicted_only=False, lost_frames=0)
        self.tracker.identity_safe = True
        self.tracker.ranker_top_id = 2
        self.tracker.ranker_switch_delta = 1.0

        self.tracker.identity_mismatch_streak = 1
        self.update(target)
        self.assertTrue(self.tracker.identity_safe)

        self.tracker.identity_mismatch_streak = 2
        self.update(target)
        self.assertFalse(self.tracker.identity_safe)
        self.assertEqual(self.tracker.hold_reason, "ranker_disagrees")

    def test_bounded_coast_requires_prior_identity_lock(self) -> None:
        target = SimpleNamespace(id=1, predicted_only=True, lost_frames=1)
        self.tracker.identity_safe = True
        self.update(target)
        self.assertTrue(self.tracker.identity_safe)
        self.assertEqual(self.tracker.identity_state, "coasting")

        target.lost_frames = self.tracker_mod._IDENTITY_SAFETY_COAST_GRACE + 1
        self.update(target)
        self.assertFalse(self.tracker.identity_safe)
        self.assertEqual(self.tracker.hold_reason, "target_predicted")

    def test_position_uncertainty_always_holds(self) -> None:
        target = SimpleNamespace(id=1, predicted_only=False, lost_frames=0)
        self.tracker.identity_match_streak = 4
        self.update(target, position_actionable=False)
        self.assertFalse(self.tracker.identity_safe)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
