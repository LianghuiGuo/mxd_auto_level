import unittest

from ml.build_lie_identity_dataset import (
    BASE_FEATURE_NAMES,
    FLAT_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    aggregate_history,
    extract_frame_features,
    select_cursor_label,
)
from src.engine.LieDetectorTracker import ConstellationTrackView


def _view(
    track_id,
    center,
    *,
    lost_frames=0,
    predicted_only=False,
    visible_streak=6,
    velocity=(0.0, 0.0),
    rotation_score=0.0,
    appearance_distance=0.1,
):
    return ConstellationTrackView(
        track_id=track_id,
        center=center,
        radius=40.0,
        role="unknown",
        orientation=0.0,
        lost_frames=lost_frames,
        velocity=velocity,
        rotation_score=rotation_score,
        appearance_distance=appearance_distance,
        predicted_only=predicted_only,
        visible_streak=visible_streak,
    )


class CursorLabelTest(unittest.TestCase):
    def test_accepts_unique_nearby_visible_track(self):
        decision = select_cursor_label(
            [_view(1, (105.0, 100.0)), _view(2, (180.0, 100.0))],
            (100.0, 100.0),
        )

        self.assertEqual(decision.track_id, 1)
        self.assertEqual(decision.reason, "accepted")

    def test_rejects_ambiguous_crossing(self):
        decision = select_cursor_label(
            [_view(1, (110.0, 100.0)), _view(2, (125.0, 100.0))],
            (100.0, 100.0),
        )

        self.assertIsNone(decision.track_id)
        self.assertEqual(decision.reason, "ambiguous_tracks")

    def test_rejects_two_tracks_inside_label_radius(self):
        decision = select_cursor_label(
            [_view(1, (102.0, 100.0)), _view(2, (132.0, 100.0))],
            (100.0, 100.0),
        )

        self.assertIsNone(decision.track_id)
        self.assertEqual(decision.reason, "ambiguous_tracks")

    def test_ignores_coasting_and_immature_tracks(self):
        decision = select_cursor_label(
            [
                _view(1, (100.0, 100.0), lost_frames=1, predicted_only=True),
                _view(2, (105.0, 100.0), visible_streak=1),
            ],
            (100.0, 100.0),
        )

        self.assertIsNone(decision.track_id)
        self.assertEqual(decision.reason, "no_visible_track")


class FeatureExtractionTest(unittest.TestCase):
    def test_exports_normalized_track_and_peer_features(self):
        views = [
            _view(
                1,
                (100.0, 50.0),
                velocity=(10.0, 0.0),
                rotation_score=90.0,
                appearance_distance=0.05,
            ),
            _view(
                2,
                (300.0, 150.0),
                velocity=(0.0, 0.0),
                rotation_score=0.0,
                appearance_distance=0.20,
            ),
        ]

        features = extract_frame_features(
            views,
            target_id=1,
            frame_width=400,
            frame_height=200,
        )

        self.assertEqual(set(features[1]), set(BASE_FEATURE_NAMES))
        self.assertAlmostEqual(features[1]["x_norm"], 0.25)
        self.assertAlmostEqual(features[1]["y_norm"], 0.25)
        self.assertEqual(features[1]["is_current_target"], 1.0)
        self.assertGreater(features[1]["rotation_score_peer_z"], 0.0)
        self.assertLess(features[1]["appearance_peer_z"], 0.0)

    def test_aggregates_history_for_tree_model(self):
        base = {name: 0.0 for name in BASE_FEATURE_NAMES}
        first = dict(base, speed_norm=0.1, appearance_distance=0.2)
        second = dict(base, speed_norm=0.3, appearance_distance=0.4)

        flat = aggregate_history([first, second], history_size=4)

        self.assertEqual(set(flat), set(FLAT_FEATURE_NAMES))
        self.assertAlmostEqual(flat["history_mean_speed_norm"], 0.2)
        self.assertAlmostEqual(flat["history_std_speed_norm"], 0.1)
        self.assertAlmostEqual(flat["history_length_norm"], 0.5)
        self.assertTrue(set(HISTORY_FEATURE_NAMES).issubset(flat))


if __name__ == "__main__":
    unittest.main()
