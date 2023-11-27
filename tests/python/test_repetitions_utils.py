from unittest import TestCase

import numpy as np

from dolma.taggers.repetitions.utils import (
    find_end_first_consecutive_true,
    find_periodic_sequences,
    find_start_last_consecutive_true,
    group_consecutive_values,
)


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

    def _to_array(self, s: str) -> np.ndarray:
        return np.array(list(map(int, s)))

    def test_find_periodic_sequences(self):
        # should be 0 repeated 5 times
        arr = self._to_array("5000007")
        sequences = list(find_periodic_sequences(arr, max_period=1))
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].start, 1)
        self.assertEqual(sequences[0].end, 6)
        self.assertEqual(sequences[0].period, 1)
        self.assertEqual(sequences[0].times, 5)

        # should be 01 repeated 3 times
        arr = self._to_array("501010127")
        sequences = list(find_periodic_sequences(arr, min_period=2, max_period=2))
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].start, 1)
        self.assertEqual(sequences[0].end, 7)
        self.assertEqual(sequences[0].period, 2)
        self.assertEqual(sequences[0].times, 3)

        # should be 0104 repeated 3 times
        arr = self._to_array("5301040104010401")
        sequences = list(find_periodic_sequences(arr, min_period=4, max_period=4))
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].start, 2)
        self.assertEqual(sequences[0].end, 16)
        self.assertEqual(sequences[0].period, 4)
        self.assertEqual(sequences[0].times, 3)

        # should be 040 repeated 4 times
        arr = np.array(list(map(int, "04004004004030")))
        sequences = list(find_periodic_sequences(arr, min_period=3, max_period=3))
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].start, 0)
        self.assertEqual(sequences[0].end, 12)
        self.assertEqual(sequences[0].period, 3)
        self.assertEqual(sequences[0].times, 4)

        # should have two repetitions: 46 repeated 4 times, and 550 repeated 3 times
        arr = np.array(list(map(int, "004646464639955055055046550")))
        sequences = list(find_periodic_sequences(arr, min_period=2, max_period=3))
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0].start, 2)
        self.assertEqual(sequences[0].end, 10)
        self.assertEqual(sequences[0].period, 2)
        self.assertEqual(sequences[0].times, 4)
        self.assertEqual(sequences[1].start, 13)
        self.assertEqual(sequences[1].end, 22)
        self.assertEqual(sequences[1].period, 3)
        self.assertEqual(sequences[1].times, 3)

    def test_find_no_periodic_sequences(self):
        arr = np.array(list(map(int, "123456789")))
        sequences = list(find_periodic_sequences(arr, max_period=10))
        self.assertEqual(len(sequences), 0)

        arr = np.array(list(map(int, "112233445566778899")))
        sequences = list(find_periodic_sequences(arr, max_period=10))
        self.assertEqual(len(sequences), 0)
