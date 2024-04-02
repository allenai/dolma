import math
import multiprocessing
import re
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union

import msgspec
import smart_open
import tqdm
from msgspec.json import Decoder
from rich.console import Console
from rich.table import Table

from .binning import (
    BaseBucketApi,
    FixedBucketsValTracker,
    InferBucketsValTracker,
    SummaryTuple,
)
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
    total: int
    sum: Union[int, float]

    @classmethod
    def from_tracker(cls, name: str, tracker: "BaseBucketApi", n: int) -> "SummarySpec":
        sm = tracker.summarize(n=n)
        return SummarySpec(name=name, counts=sm.counts, bins=sm.bins, total=sm.total, sum=sm.sum)

    def to_tracker(self, tracker_type: str = "fixed", **tracker_kwargs) -> "BaseBucketApi":
        tracker = _make_tracker(type_=tracker_type, **tracker_kwargs)
        tracker.add(values=self.bins, counts=self.counts)
        return tracker

    def from_summary_tuple(self, summary: SummaryTuple) -> "SummarySpec":
        return SummarySpec(
            name=self.name, counts=summary.counts, bins=summary.bins, total=summary.total, sum=summary.sum
        )

    def to_summary_tuple(self) -> SummaryTuple:
        return SummaryTuple(counts=self.counts, bins=self.bins, total=self.total, sum=self.sum)


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
    it = tqdm.tqdm(
        list(glob_path(summaries_path, autoglob_dirs=True, recursive_dirs=True, yield_dirs=False)),
        desc="Aggregating summaries",
        unit=" files",
        unit_scale=True,
    )

    # load partial summaries and aggregate it
    for path in it:
        with smart_open.open(path, "rt") as f:
            for ln in f:
                summary = decoder.decode(ln)
                trackers.setdefault(summary.name, _make_tracker()).add_summary(summary.to_summary_tuple())

    # convert trackers to summaries
    summaries = [
        SummarySpec.from_tracker(name=attr_name, tracker=attr_tracker, n=num_bins)
        for attr_name, attr_tracker in trackers.items()
    ]
    return summaries


def round_values_for_visual(values: List[float], opt_sci: bool = False, max_decimal: int = 4) -> List[str]:
    """Logic to round values depending on their range"""

    # we try rounding as little as possible until all values are different
    # we reach the maximum number of decimal points
    for decimal in range(max_decimal):
        attempt_rounding = [round(val, decimal) for val in values]
        if len(set(attempt_rounding)) == len(values):
            # success! let's return the rounded values
            return [f"{val:.{decimal}f}" for val in values]

    # no luck; let's use scientific notation instead if we are allowed to or simply return the values
    if opt_sci:
        return [f"{val:.1e}" for val in values]
    else:
        return [f"{val:.{max_decimal}f}" for val in values]


def visualize_summaries(
    summaries: List[SummarySpec], max_decimal: int = 4, num_viz_bins: int = 10, show_total: bool = False
):
    console = Console()
    console.print()

    for summary in summaries:
        # we use fewer bins for visualization
        short_summary = SummarySpec(
            name=summary.name,
            counts=(re_binned := summary.to_tracker().summarize(n=num_viz_bins, mode="count")).counts,
            bins=re_binned.bins,
            total=summary.total,
            sum=summary.sum,
        )
        # build the table here
        table = Table(title=short_summary.name, style="bold", min_width=len(short_summary.name))
        table.add_column("value", justify="left", style="cyan")
        table.add_column("dist", justify="left", style="magenta")
        table.add_column("count", justify="left", style="green")

        # we round the bins and write them in [lo, hi) format ]
        rounded_bins = round_values_for_visual(values=short_summary.bins, max_decimal=max_decimal)
        ranges = (
            [
                f"[{lo}, {hi}" + ("]" if i == (len(short_summary.bins) - 2) else ")")
                for i, (lo, hi) in enumerate(zip(rounded_bins, rounded_bins[1:]))
            ]
            if len(short_summary.bins) > len(short_summary.counts)
            else rounded_bins
        )

        counts_sum = sum(short_summary.counts)
        counts_normed = round_values_for_visual(
            values=[(count / counts_sum) for count in short_summary.counts], opt_sci=False, max_decimal=max_decimal
        )

        for value, dist, count in zip(ranges, counts_normed, short_summary.counts):
            table.add_row(value, dist, f"{count:,}")

        # in the last row, we show the total sum and count.
        # we first have to round the sum to a reasonable number of decimal points
        if len(str(round(short_summary.sum))) > 10:
            # regardless whether it is an integer or not, we use scientific notation for large numbers
            tot_sum_round = f"{short_summary.sum:.2e}"
        elif short_summary.sum == round(short_summary.sum):
            # use comma formatting for integers
            tot_sum_round = f"{int(short_summary.sum):,}"
        else:
            # use two decimal points for floats
            tot_sum_round = f"{short_summary.sum:.2f}"

        if show_total:
            # add totals
            table.add_row(f"{tot_sum_round}", "← sum/total →", f"{short_summary.total:,}")

        # print the table, add a newline after
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
    show_total: bool = False,
):
    """Create and run the analyzer.

    This function creates and runs an analyzer for the given attributes. It generates summaries and metadata
    based on the provided paths and parameters.

    Args:
        attributes (List[str]): List of attributes.
        summaries_path (Optional[str], optional): Path to the summaries directory. Defaults to None.
        metadata_path (Optional[str], optional): Path to the metadata directory. Defaults to None.
        report (Optional[str], optional): Path to the report directory. Defaults to None.
        debug (bool, optional): Enable debug mode. Defaults to False.
        seed (int, optional): Seed value for randomization. Defaults to 0.
        num_bins (int, optional): Number of bins for analysis. Defaults to 1000.
        num_processes (int, optional): Number of processes to use for analysis. Defaults to 1.
        name_regex (Optional[str], optional): Regular expression for filtering attribute names. Defaults to None.
        show_total (bool, optional): Show total summary. Defaults to False.
    """
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
        visualize_summaries(summaries=summaries, show_total=show_total)
        write_output(summaries=summaries, report=report)
