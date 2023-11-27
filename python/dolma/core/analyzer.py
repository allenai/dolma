import math
import multiprocessing
import re
import shutil
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import msgspec
import smart_open
import tqdm
from msgspec.json import Decoder
from rich.console import Console
from rich.table import Table

from .binning import BaseBucketApi, FixedBucketsValTracker, InferBucketsValTracker
from .data_types import OutputSpec
from .errors import DolmaError
from .parallel import BaseParallelProcessor, QueueType
from .paths import glob_path, mkdir_p

NUM_BINS = 100_000
BUFF_SIZE = 1_000


def _make_tracker(type_: str = "fixed", **kwargs: int) -> BaseBucketApi:
    """Make a tracker of given type. Choose between `infer` or `fixed`"""
    if type_ == "infer":
        return InferBucketsValTracker(**{"n": NUM_BINS, "b": BUFF_SIZE, **kwargs})
    elif type_ == "fixed":
        return FixedBucketsValTracker(**{"n": int(math.log10(NUM_BINS)), **kwargs})
    else:
        raise ValueError(f"Unknown tracker type {type_}")


class SummarySpec(msgspec.Struct):
    name: str
    counts: List[int]
    bins: List[float]

    @classmethod
    def from_tracker(cls, name: str, tracker: "BaseBucketApi", n: int) -> "SummarySpec":
        counts, bins = tracker.summarize(n=n)
        return SummarySpec(name=name, counts=counts, bins=bins)

    def to_tracker(self) -> "BaseBucketApi":
        tracker = _make_tracker()
        tracker.add_many(values=self.bins, counts=self.counts)
        return tracker


class AnalyzerProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: QueueType,  # queue must be the first argument, and it should be a positional-only argument
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        """We override this method to specify which units we want to keep track of in a progress bar.
        Specifically, we keep track of files and documents in this example. Their default value must be zero."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs,
    ):
        # instantiate a decoder for faster decoding
        decoder = Decoder(OutputSpec)

        # number of bins to use
        num_bins = kwargs.get("num_bins", 1000)

        # regex to filter attribute names
        name_regex = re.compile(r) if (r := kwargs.get("name_regex", None)) else None

        # keep track of the length and score of each attribute
        trackers: Dict[str, BaseBucketApi] = {}

        # interval at which to update the progress bar; will double if queue is too full
        update_interval = 1

        # running document count; gets reset every time we update the progress bar
        docs_cnt = 0

        with smart_open.open(source_path) as f:
            for ln in f:
                try:
                    row = decoder.decode(ln)
                except Exception as e:
                    raise DolmaError(
                        f"Failed to decode line {ln} in {source_path}; "
                        f"are you sure {source_path} is an attributes file?"
                    ) from e

                # update the length and score trackers for each attribute
                for attr_name, attr_values in row.attributes.items():
                    # if a regex is provided, skip attributes that don't match it
                    if name_regex and not name_regex.search(attr_name):
                        continue

                    # empty attributes count as zero
                    attr_values = attr_values or [(0, 0, 0.0)]
                    for start, end, score in attr_values:
                        if "__label__" in attr_name:
                            # annoying fix for fasttext: fasttext sometimes emits probabilities that are slightly
                            # above 1.0, which causes issues with histograms. Therefore, we shift values that are
                            # greater than 1.0 down to 1.0
                            #
                            # fasttext labels are of the form __label__<label>, so we can just check if the
                            # attribute name contains __label__
                            score = min(score, 1.0)

                        trackers.setdefault(f"{attr_name}/score", _make_tracker()).add(score)
                        trackers.setdefault(f"{attr_name}/length", _make_tracker()).add(end - start)

                # increment the number of documents processed so far
                docs_cnt += 1

                if docs_cnt % update_interval == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(queue, documents=docs_cnt)
                    docs_cnt = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

        with smart_open.open(destination_path, "w") as f:
            for attr_name, tracker in trackers.items():
                summary = SummarySpec.from_tracker(name=attr_name, tracker=tracker, n=num_bins)
                f.write(msgspec.json.encode(summary).decode("utf-8") + "\n")

        # update the progress bar one last time
        cls.increment_progressbar(queue, files=1, documents=docs_cnt)


def aggregate_summaries(summaries_path: str, num_bins: int = 1000) -> List[SummarySpec]:
    # keep track of the length and score of each attribute
    trackers: Dict[str, BaseBucketApi] = {}

    # instantiate a decoder for faster decoding
    decoder = Decoder(SummarySpec)

    # iterator with nice progress bar
    it = tqdm.tqdm(list(glob_path(summaries_path)), desc="Aggregating summaries", unit=" files", unit_scale=True)

    # load partial summaries and aggregate it
    for path in it:
        with smart_open.open(path, "rt") as f:
            for ln in f:
                summary = decoder.decode(ln)
                trackers.setdefault(summary.name, _make_tracker()).add_many(summary.bins, summary.counts)

    # convert trackers to summaries
    summaries = [
        SummarySpec.from_tracker(name=attr_name, tracker=attr_tracker, n=num_bins)
        for attr_name, attr_tracker in trackers.items()
    ]
    return summaries


def visualize_summaries(summaries: List[SummarySpec], digits: int = 4, num_viz_bins: int = 10):
    console = Console()
    console.print()

    def round_all(values: List[float], opt_sci: bool = False) -> List[str]:
        """Logic to round values depending on their range"""

        if values == [0, 1]:
            # just 0 and 1; no need to round or add decimal points
            return ["0", "1"]
        elif all(-1 <= val <= 1 for val in values):
            # all values are in the range [-1, 1]; let's attempt rounding with {digits} decimal points
            # unless some values are identical after rounding.
            attempt_rounding = [round(val, digits) for val in values]

            if len(set(attempt_rounding)) != len(values) and opt_sci:
                # oops, some values collide after rounding; let's use scientific notation instead
                # with  one decimal point (note that we do this only if `opt_sci` is True)
                return [f"{val:.1e}" for val in values]
            else:
                # phew, all good; let's use {digits} decimal points for all values
                return [f"{round(val, digits):.{digits}f}" for val in values]
        else:
            # all values are outside the range [-1, 1]; let's round them to the nearest integer
            return [f"{int(round(val, 0)):d}" for val in values]

    for summary in summaries:
        # we use fewer bins for visualization
        summary = SummarySpec(
            name=summary.name,
            counts=(re_binned := summary.to_tracker().summarize(n=num_viz_bins)).counts,
            bins=re_binned.bins,
        )

        # build the table here
        table = Table(title=summary.name, style="bold", min_width=len(summary.name))
        table.add_column("value", justify="left", style="cyan")
        table.add_column("dist", justify="left", style="magenta")
        table.add_column("count", justify="left", style="green")

        rounded_bins = round_all(summary.bins)
        ranges = (
            [f"[{lo}, {hi})" for lo, hi in zip(rounded_bins, rounded_bins[1:])]
            if len(summary.bins) > len(summary.counts)
            else rounded_bins
        )

        counts_sum = sum(summary.counts)
        counts_normed = round_all([(count / counts_sum) for count in summary.counts], opt_sci=False)

        for value, dist, count in zip(ranges, counts_normed, summary.counts):
            table.add_row(value, dist, f"{count:,}")

        console.print(table)
        console.print()


def write_output(summaries: List[SummarySpec], report: Optional[str] = None):
    if report is None:
        return

    mkdir_p(report)
    with smart_open.open(f"{report}/summaries.jsonl.gz", "w") as f:
        for summary in summaries:
            f.write(msgspec.json.encode(summary).decode("utf-8") + "\n")


def create_and_run_analyzer(
    attributes: List[str],
    summaries_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    report: Optional[str] = None,
    debug: bool = False,
    seed: int = 0,
    num_bins: int = 1000,
    num_processes: int = 1,
    name_regex: Optional[str] = None,
):
    """ """

    # create the report directory if it doesn't exist
    if report:
        mkdir_p(report)

    with ExitStack() as stack:
        # use temporary directories if no paths are provided
        summaries_path = summaries_path or stack.enter_context(TemporaryDirectory())
        metadata_path = metadata_path or stack.enter_context(TemporaryDirectory())

        # make sure these locations exist
        mkdir_p(summaries_path)
        mkdir_p(metadata_path)

        try:
            analyzer = AnalyzerProcessor(
                source_prefix=attributes,
                destination_prefix=summaries_path,
                metadata_prefix=metadata_path,
                debug=debug,
                seed=seed,
                ignore_existing=True,
                retries_on_error=0,
                num_processes=num_processes,
            )
            analyzer(num_bins=num_bins, name_regex=name_regex)

            summaries = aggregate_summaries(summaries_path=summaries_path, num_bins=num_bins)
            visualize_summaries(summaries=summaries)
            write_output(summaries=summaries, report=report)

        finally:
            shutil.rmtree(summaries_path)
            shutil.rmtree(metadata_path)
