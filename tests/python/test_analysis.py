from unittest import TestCase

import numpy as np

from dolma.core.binning import cumsum_with_reset, equal_count_hist


class TestResetCumsumNpNoSplit(TestCase):
    def test_multiple_zeros(self):
        arr = np.array([1, 2, 3, 0, 4, 5, 0, 6])
        expected = np.array([1, 3, 6, 0, 4, 9, 0, 6])
        result = np.array(cumsum_with_reset(arr))
        np.testing.assert_array_equal(result, expected)

    def test_no_zeros(self):
        arr = np.array([1, 2, 3, 4, 5])
        expected = np.array([1, 3, 6, 10, 15])
        result = np.array(cumsum_with_reset(arr))
        np.testing.assert_array_equal(result, expected)

    def test_start_with_zero(self):
        arr = np.array([0, 1, 2, 3, 4])
        expected = np.array([0, 1, 3, 6, 10])
        result = np.array(cumsum_with_reset(arr))
        np.testing.assert_array_equal(result, expected)

    def test_end_with_zero(self):
        arr = np.array([1, 2, 3, 0])
        expected = np.array([1, 3, 6, 0])
        result = np.array(cumsum_with_reset(arr))
        np.testing.assert_array_equal(result, expected)

    def test_all_zeros(self):
        arr = np.array([0, 0, 0, 0])
        expected = np.array([0, 0, 0, 0])
        result = np.array(cumsum_with_reset(arr))
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        arr = np.array([])
        expected = np.array([])
        result = np.array(cumsum_with_reset(arr))
        np.testing.assert_array_equal(result, expected)


class TestEqualCountHist(TestCase):
    def test_basic(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        counts, bins = equal_count_hist(a=arr, bins=9)

        self.assertEqual(len(counts), 9)
        self.assertEqual(len(bins), 10)
        self.assertEqual(counts.sum(), len(arr))
        np.testing.assert_array_equal(counts, np.ones_like(counts))
        np.testing.assert_array_equal(bins, [1] + [i + 0.5 for i in range(1, 9)] + [9])

    def test_bin_in_three(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        counts, bins = equal_count_hist(a=arr, bins=3)
        self.assertEqual(len(arr), counts.sum())
        np.testing.assert_array_equal(counts, [3, 3, 3])
        np.testing.assert_array_equal(bins, [1.0, 3.5, 6.5, 9])
