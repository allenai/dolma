import argparse
import hashlib
import json
import multiprocessing
import re
from contextlib import ExitStack
from itertools import product
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union

import msgspec
import smart_open
import uniseg.wordbreak
from dolma.core.data_types import InputSpec
from dolma.core.parallel import BaseParallelProcessor
from dolma.core.paths import join_path
from dolma.core.runtime import _make_paths_from_prefix

# we like higher digits first for consistency with previous sampling strategies
HEX_DIGITS = list("0123456789abcdef")[::-1]


def calculate_md5_suffix(prob: float) -> List[str]:
    assert 0 < prob < 1, "Sampling probability must be between 0 and 1 exclusive"

    # calculate the closest fraction of 16 to the probability
    n = 16 * prob

    if n > 1:
        suffix = HEX_DIGITS[: int(round(n))]
    else:
        # not enough hex positions to represent the probability
        # so we use more digits
        more_suffix = calculate_md5_suffix(prob * 16)
        suffix = ["".join(e) for e in product(HEX_DIGITS[:1], more_suffix)]

    assert all(len(s) < 32 for s in suffix), "Suffixes must be less than 32 characters"

    return suffix


class HashSampler(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: Queue[Union[Tuple[int, ...], None]],
        /,
        files: int = 0,
        documents: int = 0,
        words: int = 0,
        extracted: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(
            queue, files=files, documents=documents, extracted=extracted, uniseg_words=words
        )

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        decoder = msgspec.json.Decoder(InputSpec)

        probability = kwargs.get("probability", None)
        if probability is None:
            raise ValueError("Probability must be specified")
        probability = float(probability)

        complement = kwargs.get("complement", False)

        suffixes = calculate_md5_suffix(probability)
        re_suffixes = re.compile(rf'({"|".join(suffixes)})$')
        extracted_count = documents_count = words_count = 0
        update_interval = 1

        with ExitStack() as stack:
            source = stack.enter_context(smart_open.open(source_path, "rt"))
            destination = stack.enter_context(smart_open.open(destination_path, "wt"))

            for line in source:
                data = decoder.decode(line)
                md5 = hashlib.md5(data.text.encode()).hexdigest()

                if complement and not re_suffixes.search(md5):
                    extracted_count += 1
                    words_count += sum(1 for w in uniseg.wordbreak.words(data.text.strip()) if w.strip())
                    destination.write(line)
                elif not complement and re_suffixes.search(md5):
                    extracted_count += 1
                    words_count += sum(1 for w in uniseg.wordbreak.words(data.text.strip()) if w.strip())
                    destination.write(line)

                documents_count += 1

                if documents_count % update_interval == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(
                        queue, documents=documents_count, extracted=extracted_count, words=words_count
                    )
                    extracted_count = documents_count = words_count = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

        cls.increment_progressbar(
            queue, files=1, documents=documents_count, extracted=extracted_count, words=words_count
        )


def main(
    source: Union[List[str], str],
    destination: str,
    probability: float,
    complement: bool = False,
    num_workers: int = 1,
    debug: bool = False,
    dryrun: bool = False,
) -> None:
    # make source always a list
    source = [source] if isinstance(source, str) else source

    pattern = calculate_md5_suffix(probability)
    if complement:
        print(f"Sampling complement of probability {probability} split by excluding MD5 suffixes {pattern}")
    else:
        print(f"Sampling with probability {probability} using MD5 suffixes {pattern}")

    if dryrun:
        return

    with TemporaryDirectory() as tempdir:
        if len(source) > 1:
            dest_prefixes = _make_paths_from_prefix(source, join_path(None, destination))
            meta_prefixes = _make_paths_from_prefix(source, join_path(None, tempdir))
        else:
            dest_prefixes = [destination]
            meta_prefixes = [tempdir]

        processor = HashSampler(
            source_prefix=source,
            destination_prefix=dest_prefixes,
            metadata_prefix=meta_prefixes,
            num_processes=num_workers,
            debug=debug,
        )
        processor(probability=probability, complement=complement)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", type=str, required=True, help="glob pattern for source files", nargs="+")
    ap.add_argument("-d", "--destination", type=str, required=True, help="destination prefix")
    ap.add_argument("-p", "--probability", type=float, required=True, help="sampling probability")
    ap.add_argument("-n", "--num-workers", type=int, default=1, help="number of workers")
    ap.add_argument("--debug", action="store_true", help="debug mode")
    ap.add_argument("--dryrun", action="store_true", help="dry run")
    ap.add_argument("--complement", action="store_true", help="get the complement of the sample")
    opts = ap.parse_args()

    print(json.dumps(vars(opts), indent=2, sort_keys=True))
    return opts


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
