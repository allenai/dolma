from argparse import ArgumentParser
import random
from tempfile import TemporaryDirectory
from typing import Any, Dict
from multiprocessing import cpu_count
from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.paths import glob_path, join_path, make_relative, split_path, mkdir_p
import smart_open


class SamplerProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls, queue: "QueueType", /, files: int = 0, docs: int = 0, sampled: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, docs=docs, sampled=sampled)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        sample_rate = kwargs.get("sample_rate", (0.0, 1.0))
        assert len(sample_rate) == 2
        assert all(isinstance(x, float) or isinstance(x, int) for x in sample_rate)
        assert all(0 <= x <= 1 for x in sample_rate)
        lo, hi = sorted(sample_rate)

        docs = sampled = 0
        update_every = 1

        with smart_open.open(source_path, "r") as input_file, smart_open.open(destination_path, "w") as dest_file:
            for row in input_file:
                docs += 1
                if lo <= random.random() <= hi:
                    sampled += 1
                    dest_file.write(row)

                if docs >= update_every:
                    cls.increment_progressbar(queue, docs=docs, sampled=sampled)
                    docs = sampled = 0
                    if queue.qsize() >= cpu_count():
                        update_every = update_every * 2

        cls.increment_progressbar(queue, files=1, docs=docs, sampled=sampled)


def main():
    ap = ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--destination", required=True)
    ap.add_argument("--sample-rate", type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument("--num-processes", type=int, default=cpu_count())
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    mkdir_p(args.destination)

    with TemporaryDirectory() as tmp:
        root, prefixes = make_relative(list(glob_path(args.source)))
        source_prefixes, destination_prefixes, metadata_prefixes = [], [], []
        for prefix in prefixes:
            dest_prot, dest_parts = split_path(args.destination)
            _, source_parts = split_path(prefix)
            source_prefixes.append(join_path("", root, *source_parts))
            destination_prefixes.append(join_path(dest_prot, dest_parts, *source_parts[:-1]))
            metadata_prefixes.append(join_path("", tmp, *source_parts[:-1]))

        sampler = SamplerProcessor(
            source_prefix=source_prefixes,
            destination_prefix=destination_prefixes,
            metadata_prefix=metadata_prefixes,
            num_processes=min(args.num_processes, len(source_prefixes)),
            debug=args.debug,
        )
        sampler(sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
