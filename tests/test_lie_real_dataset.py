import unittest

import numpy as np

from ml.build_lie_dataset import RecordingSpec


def make_spec(active_start: float, active_end: float) -> RecordingSpec:
    return RecordingSpec(
        filename="dummy.mp4",
        roi=(0, 0, 700, 464),
        safe_crop=(0, 0, 700, 464),
        seed_time=active_start,
        active_start=active_start,
        active_end=active_end,
        family="unknown",
    )


import json
import tempfile
from pathlib import Path

from ml.build_lie_real_dataset import (
    Candidate,
    SamplingProfile,
    _apply_trajectory_continuity,
    _background_diff_candidates,
    _nms,
    _phase_at,
    _sample_timestamps,
    _snap_to_reference,
    load_specs,
)


class SamplingTest(unittest.TestCase):
    def test_phase_boundaries(self):
        spec = make_spec(active_start=0.3, active_end=13.0)
        profile = SamplingProfile()
        self.assertEqual(_phase_at(spec, 0.0, profile), "highlight")
        self.assertEqual(_phase_at(spec, spec.active_end, profile), "motion")

    def test_fade_denser_than_motion(self):
        spec = make_spec(active_start=8.5, active_end=24.5)  # longer active window
        profile = SamplingProfile(highlight_fps=2, fade_fps=9, motion_fps=4)
        stops = _sample_timestamps(spec, profile)
        phases = [phase for _, phase in stops]
        self.assertIn("fade", phases)
        self.assertIn("motion", phases)
        # Timestamps must be strictly increasing and unique.
        times = [t for t, _ in stops]
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(times), len(set(round(t, 3) for t in times)))


class SnapTest(unittest.TestCase):
    def test_snap_uses_reference_size_and_clips(self):
        # Box near the top-left corner, reference bigger than the box.
        bbox = (10, 10, 20, 20)
        snapped = _snap_to_reference(bbox, (40.0, 40.0), 200, 200)
        # Center preserved (~15,15), size fixed to 40 but clipped at 0.
        self.assertEqual(snapped[0], 0)
        self.assertEqual(snapped[1], 0)
        self.assertLessEqual(snapped[2], 200)

    def test_snap_without_reference_keeps_box(self):
        bbox = (30, 30, 70, 70)
        self.assertEqual(_snap_to_reference(bbox, None, 200, 200), bbox)


class NmsTest(unittest.TestCase):
    def test_nms_drops_overlapping_lower_score(self):
        a = Candidate((0, 0, 50, 50), 0.9, "bgdiff")
        b = Candidate((5, 5, 55, 55), 0.4, "yolo")
        c = Candidate((100, 100, 150, 150), 0.5, "bgdiff")
        kept = _nms([a, b, c], 0.5)
        boxes = {cand.bbox for cand in kept}
        self.assertIn((0, 0, 50, 50), boxes)
        self.assertIn((100, 100, 150, 150), boxes)
        self.assertNotIn((5, 5, 55, 55), boxes)


class TrajectoryTest(unittest.TestCase):
    def test_lone_detection_removed_and_gap_filled(self):
        # A stable object present in frames 0,1, (missing 2), 3 -> gap filled.
        moving = [
            [Candidate((10, 10, 50, 50), 1.0, "bgdiff")],
            [Candidate((14, 10, 54, 50), 1.0, "bgdiff")],
            [],  # single miss
            [Candidate((22, 10, 62, 50), 1.0, "bgdiff")],
        ]
        # A lone spurious detection only in frame 1.
        moving[1].append(Candidate((300, 300, 340, 340), 1.0, "bgdiff"))
        result = _apply_trajectory_continuity(
            moving,
            reference=(40.0, 40.0),
            frame_size=(700, 464),
            match_distance=48.0,
            min_hits=2,
            max_gap=2,
        )
        # Frame 2 gap should be interpolated for the tracked object.
        self.assertTrue(len(result[2]) >= 1)
        # The lone spurious box must be gone from frame 1.
        spurious = any(
            cand.bbox[0] > 250 for frame in result for cand in frame
        )
        self.assertFalse(spurious)


class BackgroundDiffTest(unittest.TestCase):
    def test_detects_bright_blob_on_flat_background(self):
        background = np.full((200, 200, 3), 40, dtype=np.uint8)
        canvas = background.copy()
        canvas[80:120, 80:120] = 220  # a 40x40 bright square
        candidates = _background_diff_candidates(
            canvas, background, (40.0, 40.0), diff_threshold=22, min_area_ratio=0.18
        )
        self.assertGreaterEqual(len(candidates), 1)
        cx = (candidates[0].bbox[0] + candidates[0].bbox[2]) / 2
        cy = (candidates[0].bbox[1] + candidates[0].bbox[3]) / 2
        self.assertTrue(80 <= cx <= 120)
        self.assertTrue(80 <= cy <= 120)


class LoadSpecsTest(unittest.TestCase):
    def _write_config(self, tmp: Path, val: list) -> Path:
        config = {
            "videos_dir": "ml/videos",
            "val": val,
            "recordings": [
                {
                    "filename": "a.mp4",
                    "roi": [10, 20, 300, 200],
                    "safe_crop": [0, 0, 300, 200],
                    "seed_time": 0.0,
                    "active_start": 0.3,
                    "active_end": 12.0,
                    "family": "circle",
                },
                {
                    "filename": "b.mp4",
                    "roi": [5, 5, 400, 260],
                    "active_start": 1.0,
                    "active_end": 15.0,
                },
            ],
        }
        path = tmp / "cfg.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_val_list_assigns_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_config(Path(tmp), val=["b.mp4"])
            _, specs = load_specs(path)
            splits = {spec.filename: split for spec, split in specs}
            self.assertEqual(splits["a.mp4"], "train")
            self.assertEqual(splits["b.mp4"], "val")
            # Missing safe_crop defaults to full ROI.
            spec_b = next(s for s, _ in specs if s.filename == "b.mp4")
            self.assertEqual(spec_b.safe_crop, (0, 0, 400, 260))

    def test_invalid_roi_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "videos_dir": "ml/videos",
                "val": [],
                "recordings": [
                    {"filename": "x.mp4", "roi": [0, 0, 0, 100], "active_end": 5.0}
                ],
            }
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_specs(path)


if __name__ == "__main__":
    unittest.main()
