import numpy as np
from dolma.taggers.repetitions.utils import (
    find_end_first_consecutive_true,
    find_start_last_consecutive_true,
    group_consecutive_values
)
from unittest import TestCase


class TestTrueLocsDetection(TestCase):

    def test_find_end_first_consecutive_true(self):
        arr = np.array([True, True, False, True])
        self.assertEqual(find_end_first_consecutive_true(arr), 2)

        arr = np.array([False, True])
        self.assertEqual(find_end_first_consecutive_true(arr), 0)

        arr = np.array([True, False])
        self.assertEqual(find_end_first_consecutive_true(arr), 1)

        arr = np.array([True])
        self.assertEqual(find_end_first_consecutive_true(arr), 1)

        arr = np.array([False])
        self.assertEqual(find_end_first_consecutive_true(arr), 0)

    def test_find_start_last_consecutive_true(self):
        arr = np.array([False, False, True, True])
        self.assertEqual(find_start_last_consecutive_true(arr), 2)

        arr = np.array([True, False, False, True])
        self.assertEqual(find_start_last_consecutive_true(arr), 3)

        arr = np.array([True, True, True, True])
        self.assertEqual(find_start_last_consecutive_true(arr), 0)

        arr = np.array([False, False, False, False])
        self.assertEqual(find_start_last_consecutive_true(arr), -1)

        arr = np.array([True, True, True, False])
        self.assertEqual(find_start_last_consecutive_true(arr), -1)

    def test_group_consecutive_values(self):
        arr = np.array([1, 2, 3, 4, 5])
        grouped = group_consecutive_values(arr)
        self.assertEqual(len(grouped), 1)
        self.assertTrue((grouped[0] == arr).all())

        arr = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11])
        grouped = group_consecutive_values(arr)
        self.assertEqual(len(grouped), 3)
        self.assertTrue((grouped[0] == np.array([1, 2, 3])).all())
        self.assertTrue((grouped[1] == np.array([5, 6, 7])).all())
        self.assertTrue((grouped[2] == np.array([9, 10, 11])).all())

        arr = np.array([1, 3, 5, 7, 9])
        grouped = group_consecutive_values(arr)
        self.assertEqual(len(grouped), 5)
        for i in range(5):
            self.assertEqual(len(grouped[i]), 1)
            self.assertEqual(grouped[i][0], arr[i])
