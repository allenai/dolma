from typing import List
import numpy as np


def find_end_first_consecutive_true(arr: np.ndarray) -> int:
    """Function to find the end position of the first consecutive sequence of True in an array."""
    if not arr[0]:
        return 0

    prog = np.cumsum(arr)
    if prog[-1] == len(arr):
        return len(arr)

    true_locs = np.where(prog[:-1:] == prog[1::])[0]

    return true_locs[0] + 1


def find_start_last_consecutive_true(arr: np.ndarray) -> int:
    """Function to find the start position of the last consecutive sequence of True in an array."""
    reverse = find_end_first_consecutive_true(arr[::-1])
    return len(arr) - reverse if reverse > 0 else -1


def group_consecutive_values(arr: np.ndarray, stepsize: int = 1) -> List[np.ndarray]:
    """Function to group consecutive values in an array."""
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0] + 1)
