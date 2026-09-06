"""Unit tests for chrome-bar based lie-panel ROI detection."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from ml.detect_lie_panels import (
    detect_frame,
    find_instruction_bar,
    find_title_bar,
    pattern_roi_from_chrome,
)


def _fake_panel_frame(
    width: int = 1280,
    height: int = 720,
    *,
    panel_x: int = 280,
    panel_y: int = 80,
    panel_w: int = 720,
    title_h: int = 28,
    instr_h: int = 40,
    pattern_h: int = 450,
) -> np.ndarray:
    """Synthesize a game frame with pale title/instruction chrome around a rock panel."""

    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    # Rocky-ish pattern (brown, higher saturation).
    rock = np.zeros((pattern_h, panel_w, 3), dtype=np.uint8)
    rock[:, :] = (60, 120, 160)  # BGR tan
    noise = np.random.default_rng(0).integers(0, 30, rock.shape, dtype=np.uint8)
    rock = np.clip(rock.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Pale chrome bars.
    pale = (210, 200, 190)
    title = np.full((title_h, panel_w, 3), pale, dtype=np.uint8)
    instr = np.full((instr_h, panel_w, 3), pale, dtype=np.uint8)
    y = panel_y
    frame[y : y + title_h, panel_x : panel_x + panel_w] = title
    y += title_h
    frame[y : y + pattern_h, panel_x : panel_x + panel_w] = rock
    y += pattern_h
    frame[y : y + instr_h, panel_x : panel_x + panel_w] = instr
    return frame


class ChromeBarDetectionTest(unittest.TestCase):
    def test_finds_instruction_and_pattern_roi(self):
        frame = _fake_panel_frame()
        instr = find_instruction_bar(frame)
        self.assertIsNotNone(instr)
        assert instr is not None
        title = find_title_bar(frame, instr)
        self.assertIsNotNone(title)
        roi = pattern_roi_from_chrome(instr, title)
        self.assertIsNotNone(roi)
        assert roi is not None
        x, y, w, h = roi
        # Pattern should sit between title and instruction, roughly full panel width.
        self.assertGreater(w, 600)
        self.assertGreater(h, 350)
        self.assertGreater(y, 80)  # below title
        self.assertLess(y + h, instr[1] + 5)

    def test_detect_frame_end_to_end(self):
        frame = _fake_panel_frame()
        roi = detect_frame(frame)
        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertEqual(len(roi), 4)
        self.assertTrue(all(v > 0 for v in roi))

    def test_scales_to_1080p_bar_height(self):
        frame = _fake_panel_frame(
            1920,
            1080,
            panel_x=420,
            panel_y=120,
            panel_w=1060,
            title_h=36,
            instr_h=86,  # previously rejected by hard-coded max=80
            pattern_h=720,
        )
        instr = find_instruction_bar(frame)
        self.assertIsNotNone(instr)
        assert instr is not None
        self.assertGreaterEqual(instr[3], 70)
        roi = detect_frame(frame)
        self.assertIsNotNone(roi)


if __name__ == "__main__":
    unittest.main()
