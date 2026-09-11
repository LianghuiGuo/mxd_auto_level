import unittest

import numpy as np

from src.engine.LieBackgroundCertificates import LieBackgroundCertificates


class LieBackgroundCertificatesTest(unittest.TestCase):
    BG = np.asarray(
        [
            (40.0, 50.0),
            (150.0, 55.0),
            (275.0, 80.0),
            (70.0, 180.0),
            (205.0, 205.0),
            (335.0, 170.0),
            (115.0, 310.0),
            (290.0, 320.0),
        ]
    )
    REAL = (440.0, 220.0)

    def armed(self):
        registry = LieBackgroundCertificates()
        for frame in range(20):
            offset = np.asarray((3.0 * frame, 0.3 * frame))
            points = np.vstack((self.BG + offset, self.REAL))
            registry.update(
                points,
                white_active=True,
                real_center=self.REAL,
                real_radius=40.0,
            )
        for _ in range(registry.FADE_CONFIRM):
            result = registry.update(
                self.BG + (60.0, 6.0), white_active=False
            )
        self.assertEqual(result.state, "armed")
        return registry

    def test_no_white_phase_fails_closed(self):
        registry = LieBackgroundCertificates()
        for _ in range(60):
            result = registry.update(self.BG, white_active=False)
        self.assertEqual(result.state, "dormant")
        self.assertFalse(result.certified_indices)

    def test_carries_supported_bg_forward(self):
        registry = self.armed()
        result = registry.update(self.BG + (66.0, 6.6), white_active=False)
        self.assertEqual(result.state, "armed")
        self.assertEqual(len(result.certified_indices), registry.MAX_EMITTED)

    def test_ambiguous_crossing_revokes_certificate(self):
        registry = self.armed()
        points = self.BG + (66.0, 6.6)
        # A ghost almost coincident with a certified BG makes ownership
        # ambiguous. The registry must revoke instead of choosing either.
        points = np.vstack((points, points[0] + (1.0, 0.0)))
        before = len(registry.certificates)
        registry.update(points, white_active=False)
        self.assertLess(len(registry.certificates), before)

    def test_three_misses_revoke_instead_of_rebinding(self):
        registry = self.armed()
        before = len(registry.certificates)
        points = self.BG[1:] + (66.0, 6.6)
        for frame in range(3):
            registry.update(points + (3.0 * frame, 0.3 * frame), white_active=False)
        self.assertLess(len(registry.certificates), before)

    def test_fade_buffer_keeps_root_close_to_moving_background(self):
        registry = LieBackgroundCertificates()
        for frame in range(20):
            offset = np.asarray((3.0 * frame, 0.3 * frame))
            registry.update(
                np.vstack((self.BG + offset, self.REAL)),
                white_active=True,
                real_center=self.REAL,
                real_radius=40.0,
            )
        for frame in range(registry.FADE_CONFIRM):
            offset = np.asarray((60.0 + 3.0 * frame, 6.0 + 0.3 * frame))
            result = registry.update(self.BG + offset, white_active=False)
        self.assertEqual(result.state, "armed")
        self.assertEqual(len(result.certified_indices), registry.MAX_EMITTED)

    def test_white_flicker_after_finalization_does_not_restart_root(self):
        registry = self.armed()
        rooted_frames = registry.white_frames
        for frame in range(5):
            offset = np.asarray((66.0 + 3.0 * frame, 6.6 + 0.3 * frame))
            result = registry.update(
                np.vstack((self.BG + offset, self.REAL)),
                white_active=True,
                real_center=self.REAL,
                real_radius=40.0,
            )
        self.assertNotIn(result.state, ("rooting", "disabled"))
        self.assertEqual(registry.white_frames, rooted_frames)

    def test_similarity_flow_handles_rotation_scale_and_translation(self):
        registry = self.armed()
        points = self.BG + (60.0, 6.0)
        angle = np.radians(0.30)
        matrix = 1.002 * np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        for _ in range(20):
            points = points @ matrix.T + (3.0, 0.5)
            result = registry.update(points, white_active=False)
        self.assertEqual(result.state, "armed")
        self.assertEqual(len(result.certified_indices), registry.MAX_EMITTED)

    def test_transient_border_outlier_never_matures(self):
        registry = self.armed()
        for frame in range(registry.BORDER_MATURITY - 1):
            points = self.BG + (66.0 + 3.0 * frame, 6.6)
            ghost = np.asarray((5.0 + frame, 120.0))
            result = registry.update(
                np.vstack((points, ghost)),
                white_active=False,
                frame_size=(520, 360),
            )
            self.assertNotIn(len(points), result.certified_indices)

    def test_frame_resize_disables_until_reset(self):
        registry = self.armed()
        registry.update(
            self.BG + (66.0, 6.6),
            white_active=False,
            frame_size=(520, 360),
        )
        result = registry.update(
            self.BG + (69.0, 6.9),
            white_active=False,
            frame_size=(700, 460),
        )
        self.assertEqual(result.state, "disabled")
        result = registry.update(
            np.vstack((self.BG, self.REAL)),
            white_active=True,
            real_center=self.REAL,
            real_radius=40.0,
            frame_size=(700, 460),
        )
        self.assertEqual(result.state, "disabled")


if __name__ == "__main__":
    unittest.main()
