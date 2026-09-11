import unittest

import numpy as np

from ml.run_lie_motion_ablation import (
    _fit_similarity,
    _graph_strain,
    _predict_similarity,
)


class LieMotionAblationTest(unittest.TestCase):
    def test_similarity_fit_recovers_rotation_scale_translation(self):
        previous = np.asarray(
            [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0], [9.0, 7.0]]
        )
        angle = np.radians(7.0)
        scale = 1.03
        matrix = scale * np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        current = previous @ matrix.T + np.asarray([4.0, -2.0])
        center, beta, fitted_scale = _fit_similarity(previous, current)
        predicted = _predict_similarity(previous, center, beta)
        self.assertLess(float(np.max(np.linalg.norm(predicted - current, axis=1))), 1e-5)
        self.assertAlmostEqual(fitted_scale, scale, places=5)

    def test_graph_strain_marks_relative_outlier(self):
        previous = np.asarray(
            [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0], [10.0, 10.0]]
        )
        current = previous + np.asarray([3.0, 2.0])
        current[-1] += np.asarray([12.0, 0.0])
        strain = _graph_strain(previous, current, 1.0)
        self.assertGreater(strain[-1], float(np.median(strain[:-1])))


if __name__ == "__main__":
    unittest.main()
