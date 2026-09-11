import unittest

import numpy as np

from ml.train_lie_identity_ranker import (
    _hard_query_mask,
    _select_queries,
    _top1_hits,
)


class RankerEvalTest(unittest.TestCase):
    def test_leave_one_video_keeps_query_groups_intact(self):
        split = {
            "x": np.arange(7, dtype=np.float32).reshape(-1, 1),
            "y": np.array([1, 0, 0, 1, 0, 1, 0], dtype=np.int32),
            "groups": np.array([2, 3, 2], dtype=np.int32),
            "queries": ["a:1", "b:1", "a:2"],
            "videos": np.array(["a.mp4", "b.mp4", "a.mp4"]),
        }

        held = _select_queries(split, split["videos"] == "b.mp4")
        rest = _select_queries(split, split["videos"] != "b.mp4")

        self.assertEqual(held["queries"], ["b:1"])
        self.assertEqual(rest["queries"], ["a:1", "a:2"])
        np.testing.assert_array_equal(held["groups"], [3])
        np.testing.assert_array_equal(rest["groups"], [2, 2])
        np.testing.assert_array_equal(held["x"].ravel(), [2, 3, 4])
        np.testing.assert_array_equal(rest["x"].ravel(), [0, 1, 5, 6])

    def test_hard_queries_are_where_current_target_is_not_the_label(self):
        labels = np.array([1, 0, 0, 1], dtype=np.int32)
        current = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        groups = np.array([2, 2], dtype=np.int32)

        mask = _hard_query_mask(labels, current, groups)

        np.testing.assert_array_equal(mask, [False, True])
        np.testing.assert_array_equal(
            _top1_hits(np.array([0.9, 0.1, 0.2, 0.8]), labels, groups),
            [1, 1],
        )
