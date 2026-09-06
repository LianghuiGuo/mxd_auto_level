import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.build_lie_dataset_v2 import (
    Geometry,
    RenderProfile,
    make_objects,
    render_refractive_shape,
    transformed_polygon,
    _validated_yolo_label_count,
)


class LieDatasetV2Test(unittest.TestCase):
    @staticmethod
    def square_geometry() -> Geometry:
        return Geometry(
            family="square",
            points=np.array(
                [[-40, -40], [40, -40], [40, 40], [-40, 40]],
                dtype=np.float32,
            ),
            base_diameter=80.0,
            recorded=True,
            min_count=5,
            max_count=7,
        )

    def test_refractive_renderer_changes_boundary_and_interior(self):
        yy, xx = np.mgrid[:240, :320]
        background = np.dstack(
            (
                (xx * 3 + yy) % 255,
                (xx + yy * 2) % 255,
                (xx * 2 + yy * 3) % 255,
            )
        ).astype(np.uint8)
        image = background.copy()
        polygon = transformed_polygon(
            self.square_geometry().points,
            (160.0, 120.0),
            1.0,
            23.0,
        )
        profile = RenderProfile(8.0, 0.03, 0.8, 2.0, 4.0, 1.12, 0.75)
        render_refractive_shape(image, polygon, profile, highlight_alpha=0.0)
        difference = np.abs(image.astype(np.int16) - background.astype(np.int16))
        self.assertGreater(float(np.mean(difference)), 0.5)
        self.assertTrue(np.array_equal(image[:20, :20], background[:20, :20]))

    def test_objects_share_scene_scale_and_decoy_motion(self):
        objects = make_objects(
            self.square_geometry(),
            700,
            464,
            random.Random(17),
        )
        self.assertGreaterEqual(len(objects), 3)
        scales = np.asarray([item.scale for item in objects])
        self.assertLess(float(np.max(scales) / np.min(scales)), 1.07)
        decoys = [item for item in objects if not item.is_target]
        self.assertTrue(decoys)
        for decoy in decoys[1:]:
            np.testing.assert_allclose(decoy.velocity, decoys[0].velocity)
            self.assertAlmostEqual(
                decoy.angular_velocity,
                decoys[0].angular_velocity,
            )
        target = next(item for item in objects if item.is_target)
        self.assertFalse(np.allclose(target.velocity, decoys[0].velocity))

    def test_reviewed_yolo_labels_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.4 0.2 0.3\n", encoding="utf-8")
            self.assertEqual(_validated_yolo_label_count(label), 1)
            label.write_text("1 0.5 0.4 0.2 0.3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validated_yolo_label_count(label)


if __name__ == "__main__":
    unittest.main()
