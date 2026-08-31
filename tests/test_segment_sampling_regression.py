import unittest

import numpy as np

from lilia.segment_sampling import pick_non_overlapping_segments


class SegmentSamplingRegressionTests(unittest.TestCase):
    def test_short_recording_returns_empty(self):
        rng = np.random.default_rng(42)
        starts = pick_non_overlapping_segments(
            n_total=100,
            seg_len=200,
            n_segs=2,
            rng=rng,
        )
        self.assertEqual(starts, [])

    def test_non_overlapping_property(self):
        rng = np.random.default_rng(7)
        seg_len = 100
        starts = pick_non_overlapping_segments(
            n_total=10_000,
            seg_len=seg_len,
            n_segs=8,
            rng=rng,
        )
        self.assertLessEqual(len(starts), 8)
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                self.assertGreaterEqual(abs(starts[i] - starts[j]), seg_len)

    def test_reproducible_with_seed(self):
        a = pick_non_overlapping_segments(
            n_total=4_000,
            seg_len=120,
            n_segs=6,
            rng=np.random.default_rng(123),
        )
        b = pick_non_overlapping_segments(
            n_total=4_000,
            seg_len=120,
            n_segs=6,
            rng=np.random.default_rng(123),
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
