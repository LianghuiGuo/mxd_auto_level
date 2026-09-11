"""Pure-Python LightGBM inference + ranker gating unit tests."""

from __future__ import annotations

import sys
import unittest
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.LieIdentityRanker import (  # noqa: E402
    AVAILABLE_FEATURE_NAMES,
    FLAT_FEATURE_NAMES,
    LightGbmTextModel,
    TrajectoryIdentityRanker,
    aggregate_history,
    extract_similarity_features,
    extract_frame_features,
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


class FeatureAndHistoryTest(unittest.TestCase):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
