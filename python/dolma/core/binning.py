import math
from abc import abstractmethod, abstractproperty
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

# # # OLD IMPORT # # #
# from sortedcontainers import SortedDict


class SummaryTuple(NamedTuple):
    counts: List[int]
    bins: List[float]


def sort_and_merge_bins(
    bins: npt.NDArray[np.float64], counts: npt.NDArray[np.int64], mask: Optional[npt.NDArray[np.bool_]] = None
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Sort bins and counts; merge counts for duplicate bins"""

    masked_bins = bins[mask] if mask is not None else bins
    masked_counts = counts[mask] if mask is not None else counts

    uniq_bins, uniq_indices, uniq_counts = np.unique(masked_bins, return_counts=True, return_index=True)
    uniq_counts *= masked_counts[uniq_indices]

    return uniq_bins, uniq_counts


def merge_bins(
    bin_a: npt.NDArray[np.float64],
    count_a: npt.NDArray[np.int64],
    bin_b: npt.NDArray[np.float64],
    count_b: npt.NDArray[np.int64],
):
    """Merge two sets of bins and counts into one set of bins and counts;
    assumes that bin_a and bin_b are sorted

    Args:
        bin_a (npt.NDArray[np.float64]): A sorted array of bins
        count_a (npt.NDArray[np.int64]): A corresponding array of counts
        bin_b (npt.NDArray[np.float64]): A sorted array of bins
        count_b (npt.NDArray[np.int64]): A corresponding array of counts
    """
    if bin_a.size < bin_b.size:
        # bin_a is always the larger one
        bin_a, count_a, bin_b, count_b = bin_b, count_b, bin_a, count_a

    # we first find where the bins in bin_b would be inserted into bin_a
    # b_locs = np.minimum(np.searchsorted(bin_a, bin_b, side="left"), bin_a.size - 1)
    b_locs = np.searchsorted(bin_a, bin_b, side="left")

    # make a masked version of b_locs that only contains indices that are in bounds
    b_bounded_mask = b_locs < bin_a.size
    b_bounded_locs = b_locs * b_bounded_mask.astype(b_locs.dtype)

    # we make a few useful masks and arrays for later operations
    # we need to keep track of which bins in bin_b are new and which are duplicates of bins in bin_a
    # for the former, we will insert them into bin_a at the appropriate locations; for the latter,
    # we will add their counts to the counts of the corresponding bins in bin_a

    # new mask consists of either the bins in bin_b that are in bounds (i.e., would be between values of
    # bin_a and not after) and are not equal to the corresponding bins in bin_a, or the bins in bin_b that
    # will be inserted after the last bin in bin_a.
    b_new_mask = (bin_a[b_bounded_locs] != bin_b) | ~b_bounded_mask
    b_new_vals = bin_b[b_new_mask]

    # now we need to find the locations where the new bins will be inserted into new array bin_c which
    # is the size of bin_a + the number of new bins in bin_b
    b_new_locs = np.arange(b_new_vals.size) + b_locs[b_new_mask]

    # this is were we will store the new bins and counts
    bin_c = np.empty(bin_a.size + b_new_vals.size, dtype=bin_a.dtype)

    # we first fill bins from bin_a into bin_c; a_indices is a mask of the indices in bin_c that
    # should be filled with values from bin_a, so we remove values from bin_c.
    a_indices = np.ones(bin_c.size, dtype=bool)
    a_indices[b_new_locs] = False
    bin_c[a_indices] = bin_a

    # finally, we add values from bin_b into bin_c
    bin_c[b_new_locs] = b_new_vals

    # now onto the counts; we start by creating new container, and populate counts from count_a and count_b
    # where bin_a values are different from bin_b values.
    count_c = np.empty_like(bin_c, dtype=count_a.dtype)
    count_c[b_new_locs] = count_b[b_new_mask]
    count_c[a_indices] = count_a

    # finally, for the remaining counts, we group them by bin value and add them to counts from count_a
    # we must group because `array[locs] += values` does not work if there are duplicate indices in locs.
    b_uniq_locs, b_repeats, b_rep_cnt = np.unique(b_locs[~b_new_mask], return_counts=True, return_index=True)
    count_c[b_uniq_locs] += count_b[~b_new_mask][b_repeats] * b_rep_cnt

    return bin_c, count_c


class BaseBucketApi:
    @abstractproperty
    def full(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def add(self, value: Union[int, float], count: int = 1):
        raise NotImplementedError()

    def add_many(self, values: List[Union[int, float]], counts: List[int]):
        for value, count in zip(values, counts):
            self.add(value, count)

    @abstractmethod
    def summarize(self, n: int, density: bool = False) -> SummaryTuple:
        raise NotImplementedError()


class InferBucketsValTracker(BaseBucketApi):
    """Keep track of running values by using two bucketed buffers"""

    _bins: npt.NDArray[np.float64]
    _counts: npt.NDArray[np.int64]
    _buffer_bins: npt.NDArray[np.float64]
    _buffer_counts: npt.NDArray[np.int64]
    _buffer_idx: int
    n: int
    _n: int

    def __init__(self, n: int, b: Optional[int] = None):
        self.n = self._n = n
        self.b = b or int(np.sqrt(n))

        self._bins = np.empty(0, dtype=np.float64)
        self._counts = np.empty_like(self._bins, dtype=np.int64)

        # hold temporary values in a buffer
        self._new_buffer()

    def _new_buffer(self):
        """Create a new buffer and reset the buffer index"""
        self._buffer_bins = np.zeros(self.b, dtype=np.float64)
        self._buffer_counts = np.zeros_like(self._buffer_bins, dtype=np.int64)
        self._buffer_idx = 0

    def _add_buffer_to_bins(self):
        """Bin the values in the buffer and merge them with the existing bins and counts."""
        locs = np.minimum(np.searchsorted(self._bins, self._buffer_bins, side="left"), self._bins.size - 1)
        trim_og_locs, trim_og_counts = locs[: self._buffer_idx], self._buffer_counts[: self._buffer_idx]
        trim_locs, repeats_locs, repeats_cnt = np.unique(trim_og_locs, return_counts=True, return_index=True)
        trim_counts = trim_og_counts[repeats_locs] * repeats_cnt
        self._counts[trim_locs] += trim_counts

        self._new_buffer()

    def _sort_buffer(self):
        mask = np.arange(0, self._buffer_bins.size) < self._buffer_idx
        bins, counts = sort_and_merge_bins(bins=self._buffer_bins, counts=self._buffer_counts, mask=mask)

        self._buffer_bins = bins
        self._buffer_counts = counts
        self._buffer_idx = bins.size

    def _concat_buffer(self):
        """Concatenate the buffer with the existing bins and counts."""

        # make sure the buffer is sorted before merging
        self._sort_buffer()

        if self._bins.size == 0:
            # shortcut: if there are no bins, just copy the buffer
            self._bins = self._buffer_bins
            self._counts = self._buffer_counts
            self._new_buffer()
            return

        # actually do the merge here!
        self._bins, self._counts = merge_bins(
            bin_a=self._bins, count_a=self._counts, bin_b=self._buffer_bins, count_b=self._buffer_counts
        )
        self._new_buffer()

    def _add_not_full(self, value: float, count: int = 1):
        """Add a value to the tracker when the tracker is not full; in this case, the value is
        added to the buffer and eventually merged with existing bins and counts."""

        self._n -= 1
        if self._n < 0:
            return self._add_full(value, count)

        self._buffer_bins[self._buffer_idx] = value
        self._buffer_counts[self._buffer_idx] = count
        self._buffer_idx += 1

        if self._buffer_idx == self._buffer_bins.size:
            self._concat_buffer()

    def _add_full(self, value: float, count: int = 1):
        """Add a value to the tracker when the tracker is full; in this case, the value is added by
        bisecting the tracker and adding the value to the appropriate bucket."""
        self._buffer_bins[self._buffer_idx] = value
        self._buffer_counts[self._buffer_idx] = count
        self._buffer_idx += 1

        if self._buffer_idx == self._buffer_bins.size:
            self._add_buffer_to_bins()

    def __len__(self) -> int:
        return self._counts.size

    @property
    def full(self) -> bool:
        return self._n <= 0

    def add(self, value: Union[int, float], count: int = 1):
        if self._n >= 0:
            self._add_not_full(value=value, count=count)
        else:
            self._add_full(value=value, count=count)

    def summarize(self, n: int, density: bool = False) -> SummaryTuple:
        """Return up to n buckets with counts of merged values"""

        # finalize operations
        self._concat_buffer() if self._n >= 0 else self._add_buffer_to_bins()

        if len(self) <= n:
            # if there are fewer than n buckets, return the buckets as is
            return SummaryTuple(counts=self._counts.tolist(), bins=self._bins.tolist())

        # make weighted histogram using counts
        new_counts, new_values = np.histogram(a=self._bins, bins=n, weights=self._counts, density=density)

        # return lists instead of numpy arrays
        return SummaryTuple(counts=new_counts.tolist(), bins=new_values.tolist())


class FixedBucketsValTracker(BaseBucketApi):
    def __init__(self, n: int = 2):
        # we use n to determine the precision of the bins; for convenience we store it as a power of 10.
        # 10**n will be the maximum number of bins for each power of 2.
        # Too large numbers will cause numeric problems and can cause a lot of memory use.
        assert n >= 0
        assert n <= 100
        self.n = 10**n
        self._bins: Dict[Tuple[int, int], int] = {}

    def add(self, value: Union[int, float], count: int = 1):
        m, e = math.frexp(value)
        k = math.floor(m * self.n), e

        if k not in self._bins:
            self._bins[k] = 0
        self._bins[k] += count

    def __len__(self) -> int:
        return len(self._bins)

    @property
    def full(self) -> bool:
        return False

    def get_bin_upper_bound(self, val: float) -> float:
        """Return the upper bound of the bin containing val"""
        m, e = math.frexp(val)
        k = math.floor(m * self.n) + 1  # Add one to obtain the next bin
        return k / self.n * 2**e

    def summarize(self, n: int, density: bool = False) -> SummaryTuple:
        bins, counts = zip(*sorted((m / self.n * 2**e, c) for (m, e), c in self._bins.items()))

        if len(self) <= n:
            # if there are fewer than n buckets, return the buckets as is
            # To be consistent we also add the limit of the last bin, so the bins denote bin edges
            upper_bin = self.get_bin_upper_bound(max(float(b) for b in bins))
            return SummaryTuple(counts=[int(c) for c in counts], bins=[float(b) for b in bins] + [upper_bin])

        # computing the weighted histograms
        new_counts, new_values = np.histogram(a=bins, bins=n, weights=counts, density=density)

        # return lists instead of numpy arrays
        return SummaryTuple(counts=new_counts.tolist(), bins=new_values.tolist())
