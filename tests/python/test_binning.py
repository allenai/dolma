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

        tracker_counts, tracker_bins, tracker_total, tracker_sum = tracker.summarize(10)

        self.assertEqual(sum(tracker_counts), len(values))
        self.assertEqual(sorted(tracker_bins), tracker_bins)
        self.assertEqual(tracker_total, len(values))
        self.assertAlmostEqual(tracker_sum, np.sum(values), delta=0.01)

        tracker_dist, tracker_bins, tracker_total, tracker_sum = tracker.summarize(10, density=True)
        hist_dist, hist_bins = np.histogram(values, bins=10, density=True)

        for td, hd in zip(tracker_dist, hist_dist):
            self.assertAlmostEqual(np.abs(td - hd), 0, delta=0.1)

        for tb, hb in zip(tracker_bins, hist_bins):
            self.assertAlmostEqual(np.abs(tb - hb), 0, delta=0.5)

        self.assertEqual(tracker_total, len(values))
        self.assertAlmostEqual(tracker_sum, np.sum(values), delta=0.01)


class FixedBinning(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)

    def test_normal_bins(self):
        tr = FixedBucketsValTracker()
        vals = np.random.randn(2_000_000) * 100
        total_count = len(vals)
        tr.add([float(e) for e in vals], [1 for _ in vals])

        tracker_counts, tracker_bins, _, _ = tr.summarize(10)
        hist_counts, hist_bins = np.histogram(vals, bins=10)

        count_diff = np.abs(tracker_counts - hist_counts) / total_count
        bin_diff = np.abs(tracker_bins - hist_bins)
        self.assertLess(np.sum(count_diff), 0.03)
        self.assertLess(np.sum(bin_diff), 30)

    def test_uniform_bins(self):
        tr = FixedBucketsValTracker()
        vals = np.random.rand(2_000_000)
        total_count = len(vals)

        for v in vals:
            tr.add(v)

        tracker_counts, tracker_bins, _, _ = tr.summarize(10)
        hist_counts, hist_bins = np.histogram(vals, bins=10)

        count_diff = np.abs(tracker_counts - hist_counts) / total_count
        bin_diff = np.abs(tracker_bins - hist_bins)

        self.assertLess(np.sum(count_diff), 0.01)
        self.assertLess(np.sum(bin_diff), 10)
