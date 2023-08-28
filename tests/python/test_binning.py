import unittest

import numpy as np

from dolma.core.binning import (
    FixedBucketsValTracker,
    InferBucketsValTracker,
    merge_bins,
)


class TestBinning(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_binning(self):
        bin_a = np.arange(0, 10_000, 55).astype(np.float64)
        bin_b = np.arange(0, 10_000, 100).astype(np.float64)

        count_a = np.random.randint(0, 15, len(bin_a))
        count_b = np.random.randint(0, 15, len(bin_b))

        bin_c, count_c = merge_bins(bin_a, count_a, bin_b, count_b)

        self.assertEqual(set(bin_c), set(np.concatenate([bin_a, bin_b])))
        self.assertEqual(sum(count_c), sum(count_a) + sum(count_b))

    def test_binning_with_repetitions(self):
        bin_a = np.random.randint(0, 100, 10_000).astype(np.float64)
        bin_b = np.random.randint(0, 100, 1_000).astype(np.float64)

        bin_a = np.sort(bin_a)
        bin_b = np.sort(bin_b)

        count_a = np.ones_like(bin_a, dtype=np.int64)
        count_b = np.ones_like(bin_b, dtype=np.int64)

        bin_c, count_c = merge_bins(bin_a, count_a, bin_b, count_b)

        self.assertEqual(set(bin_c), set(np.concatenate([bin_a, bin_b])))
        self.assertEqual(sum(count_c), sum(count_a) + sum(count_b))

    def test_binning_no_overlap(self):
        bin_a = np.arange(0, 1_000, 3).astype(np.float64)
        bin_b = np.arange(17, 1_000, 17).astype(np.float64)

        count_a = np.ones_like(bin_a, dtype=np.int64)
        count_b = np.ones_like(bin_b, dtype=np.int64)

        bin_c, count_c = merge_bins(bin_a, count_a, bin_b, count_b)

        self.assertEqual(set(bin_c), set(np.concatenate([bin_a, bin_b])))
        self.assertEqual(sum(count_c), sum(count_a) + sum(count_b))

    def test_bucket_val_trackers(self):
        tracker = InferBucketsValTracker(n=100_000)

        values = np.random.randn(1_000_000)

        for v in values:
            tracker.add(v)

        tracker_counts, tracker_bins = tracker.summarize(10)

        self.assertEqual(sum(tracker_counts), len(values))
        self.assertEqual(sorted(tracker_bins), tracker_bins)

        tracker_dist, tracker_bins = tracker.summarize(10, density=True)
        hist_dist, hist_bins = np.histogram(values, bins=10, density=True)

        for td, hd in zip(tracker_dist, hist_dist):
            self.assertAlmostEqual(np.abs(td - hd), 0, delta=0.1)

        for tb, hb in zip(tracker_bins, hist_bins):
            self.assertAlmostEqual(np.abs(tb - hb), 0, delta=0.5)


class FixedBinning(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_normal_bins(self):
        tr = FixedBucketsValTracker()
        vals = np.random.randn(2_000_000) * 100
        total_count = len(vals)

        for v in vals:
            tr.add(v)

        for (tr_c, tr_b), (hist_c, hist_b) in zip(zip(*tr.summarize(10)), zip(*np.histogram(vals, bins=10))):
            count_diff = np.abs(tr_c - hist_c) / total_count
            bin_diff = np.abs(tr_b - hist_b)
            self.assertLess(count_diff, 0.01)
            self.assertLess(bin_diff, 10)

    def test_uniform_bins(self):
        tr = FixedBucketsValTracker()
        vals = np.random.rand(2_000_000)
        total_count = len(vals)

        for v in vals:
            tr.add(v)

        for (tr_c, tr_b), (hist_c, hist_b) in zip(zip(*tr.summarize(10)), zip(*np.histogram(vals, bins=10))):
            count_diff = np.abs(tr_c - hist_c) / total_count
            bin_diff = np.abs(tr_b - hist_b)
            self.assertLess(count_diff, 0.01)
            self.assertLess(bin_diff, 0.01)
