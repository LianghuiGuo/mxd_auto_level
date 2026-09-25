import unittest
from pathlib import Path

import cv2
import numpy as np

from ml.synth import GREEN, _background_mask, sprite_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GreenScreenMaskTest(unittest.TestCase):
    def test_loose_green_body_does_not_flood_from_screen(self):
        image = np.full((15, 15, 3), GREEN, dtype=np.uint8)
        # A green monster body whose loose-green pixels touch the exact screen.
        # The old connected-component implementation erased this whole block.
        body_color = (20, 210, 80)
        image[4:11, 4:11] = body_color

        background = _background_mask(image)

        self.assertTrue(background[0, 0])
        self.assertFalse(background[7, 7])
        self.assertTrue(np.all(image[7, 7] == body_color))

    def test_one_pixel_green_fringe_next_to_screen_is_removed(self):
        image = np.full((13, 13, 3), GREEN, dtype=np.uint8)
        image[3:10, 3:10] = (0, 0, 255)
        fringe_color = (0, 220, 35)
        image[3, 3:10] = fringe_color

        background = _background_mask(image)

        self.assertTrue(np.all(background[3, 3:10]))
        self.assertFalse(background[5, 5])

    def test_enclosed_exact_green_is_not_treated_as_screen(self):
        image = np.full((13, 13, 3), GREEN, dtype=np.uint8)
        image[3:10, 3:10] = (0, 0, 0)
        image[4:9, 4:9] = GREEN

        background = _background_mask(image)

        self.assertTrue(background[0, 0])
        self.assertFalse(background[6, 6])

    def test_slime_fixture_keeps_its_green_body(self):
        path = PROJECT_ROOT / "monster" / "slime" / "slime_3.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)

        _, keep = sprite_mask(image)

        # Before this regression fix only 433 pixels survived because almost
        # the entire green body was connected to the loose green-screen mask.
        self.assertGreater(np.count_nonzero(keep), 1200)

    def test_curse_eye_fixture_keeps_its_green_body(self):
        path = (PROJECT_ROOT / "monster" / "curse_eye" /
                "curse_eye_7.png")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)

        _, keep = sprite_mask(image)

        # The old loose-green flood kept only 774 pixels of this frame.
        self.assertGreater(np.count_nonzero(keep), 1600)


if __name__ == "__main__":
    unittest.main()
