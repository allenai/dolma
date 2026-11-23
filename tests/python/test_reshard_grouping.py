"""Exact grouping compatibility and linear size lookup regression."""

import random
from collections import Counter
from dataclasses import dataclass

from dolma.tokenizer.reshard import group_paths_by_max_size


@dataclass(frozen=True)
class SizedPath:
    name: int
    bytes: int

    @property
    def size(self):
        return self.bytes


def reference(paths, limit):
    counts = Counter(paths)
    groups = []
    while counts:
        groups.append([])
        for path, _ in sorted(counts.items(), key=lambda x: -x[1]):
            if sum(p.size for p in groups[-1]) + path.size > limit:
                groups.append([path])
            else:
                groups[-1].append(path)
        counts = {p: n - 1 for p, n in counts.items() if n > 1}
    return groups


def test_exact_grouping():
    rng = random.Random(719)
    cases = [[0], [4, 4], [8], [9], [0, 8, 0], [9, 1], [3, 7, 2, 8]]
    cases += [[rng.randrange(25) for _ in range(rng.randrange(1, 50))] for _ in range(200)]
    for sizes in cases:
        unique = [SizedPath(i, size) for i, size in enumerate(sizes)]
        paths = [p for p in unique for _ in range(rng.randrange(1, 5))]
        for limit in (0, 1, 8, 16, 100000):
            assert group_paths_by_max_size(paths, limit) == reference(paths, limit)


def test_size_lookup_once_per_occurrence():
    calls = []

    class PathWithCount:
        @property
        def size(self):
            calls.append(1)
            return 4

    paths = [PathWithCount() for _ in range(100)]
    assert group_paths_by_max_size(paths, 100000) == [paths]
    assert len(calls) == len(paths)
