import datetime
import multiprocessing
import tempfile
from contextlib import ExitStack
from itertools import chain
from time import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import msgspec
import smart_open
from courlan import clean_url
from necessary import necessary

from ..core.data_types import InputSpecWithMetadataAndAttributes
from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.paths import glob_path, join_path, split_ext
from ..core.registry import TaggerRegistry
from ..core.runtime import _make_paths_from_prefix
from ..core.utils import make_variable_name
from .linearizers import LinearizerRegistry
from .utils import raise_warc_dependency_error

with necessary("fastwarc", soft=True) as FASTWARC_AVAILABLE:
    if FASTWARC_AVAILABLE or TYPE_CHECKING:
        from fastwarc.stream_io import (  # pylint: disable=no-name-in-module
            GZipStream,
            LZ4Stream,
        )
        from fastwarc.warc import (  # pylint: disable=no-name-in-module
            ArchiveIterator,
            WarcRecordType,
        )


with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    if DATEPARSER_AVAILABLE or TYPE_CHECKING:
        import dateparser


DATE_FORMATS = ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]


class WarcProcessor(BaseParallelProcessor):
    """Processes WARC files, like the ones used by Common Crawl, in parallel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert FASTWARC_AVAILABLE, raise_warc_dependency_error("fastwarc")
        assert DATEPARSER_AVAILABLE, raise_warc_dependency_error("dateparser")

    @staticmethod
    def _format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
        """Format a timestamp as a string using near ISO-8601 format."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    @staticmethod
    def _parse_warc_timestamp(timestamp_str: Optional[str]) -> datetime.datetime:
        """Parse a WARC timestamp into a datetime object."""
        # import sys
        # sys.stdin = open("/dev/tty")
        # import ipdb; ipdb.set_trace()

        if not timestamp_str:
            return datetime.datetime.now()

        for fmt in DATE_FORMATS:
            try:
                return datetime.datetime.strptime(timestamp_str, fmt)
            except ValueError:
                pass

        return dateparser.parse(date_string=timestamp_str) or datetime.datetime.now()

    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: QueueType,  # queue must be the first argument, and it should be a positional-only argument
        /,
        files: int = 0,
        records: int = 0,
        extracted: int = 0,
    ) -> Dict[str, int]:
        """Records (documents) and records are the units we use to track progress."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, records=records, extracted=extracted)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs,
    ):
        """Lets extract from a single WARC file."""

        warc_date: Optional[datetime.datetime] = None
        warc_filename: Optional[str] = None
        date_now = datetime.datetime.now()
        date_now_str = cls._format_to_dolma_timestamp(date_now)

        # interval at which to update the progress bar; will double if it gets too full
        update_interval = 1

        # hold the number of records processed in this variable
        records_cnt = 0
        extracted_cnt = 0

        # encoder
        encoder = msgspec.json.Encoder()

        # get the name and version of this source
        source_name = kwargs.get("source_name", None)
        source_version = kwargs.get("source_version", "v0")
        if not isinstance(source_name, str):
            raise ValueError(f"source_name must be a string, not {source_name} ({type(source_name)})")

        # create any tagger that runs before html extraction
        pre_taggers_names: List[str] = kwargs.get("pre_taggers") or []
        pre_taggers = {make_variable_name(name): TaggerRegistry.get(name)() for name in pre_taggers_names}

        # create the html extractor
        linearizer_name: str = kwargs.get("linearizer_name") or "resiliparse"
        linearizer = LinearizerRegistry.get(linearizer_name)()

        # get compression format
        cpz_ext = kwargs.get("compression", None) or "zst"

        # check for duplicate URLs
        check_duplicate_urls = bool(kwargs.get("check_duplicate_urls", None) or False)
        seen_urls: Set[str] = set()

        # keep track of the time it takes to process each document
        elapsed_time = time()

        # create any tagger that runs after html extraction
        post_taggers_names: List[str] = kwargs.get("post_taggers") or []
        post_taggers = {make_variable_name(name): TaggerRegistry.get(name)() for name in post_taggers_names}

        # whether to store html in metadata after extraction
        store_html_in_metadata: bool = kwargs.get("store_html_in_metadata") or False

        # whether to skip this document if pre-taggers find nothing
        skip_no_pre_taggers: bool = kwargs.get("skip_no_pre_taggers") or False

        # whether to skip this document if post-taggers find nothing
        skip_no_post_taggers: bool = kwargs.get("skip_no_post_taggers") or False

        # derive the destination path if it is not provided by splitting out all the
        # extensions, removing gz and warc, and adding jsonl.gz
        if not destination_path.endswith(f".jsonl.{cpz_ext}"):
            prot, base_dst, extension = split_ext(destination_path)
            extension = extension.replace(f".{cpz_ext}", "").replace(".warc", "") + f".jsonl.{cpz_ext}"
            destination_path = join_path(prot, *base_dst[:-1], base_dst[-1] + extension)

        with ExitStack() as stack:
            output_file = stack.enter_context(smart_open.open(destination_path, "wb"))

            if source_path.endswith(".lz4"):
                warc_stream = stack.enter_context(smart_open.open(source_path, "rb", compression="disable"))
                warc_file = LZ4Stream(warc_stream)
            elif source_path.endswith(".gz"):
                warc_stream = stack.enter_context(smart_open.open(source_path, "rb", compression="disable"))
                warc_file = GZipStream(warc_stream)
            else:
                warc_file = stack.enter_context(smart_open.open(source_path, "rt"))

            it = ArchiveIterator(warc_file, record_types=WarcRecordType.response | WarcRecordType.warcinfo)
            for record in it:
                if record.record_type == WarcRecordType.warcinfo:
                    warc_date = record.record_date or None
                    warc_timestamp = cls._format_to_dolma_timestamp(warc_date) or ""
                    warc_filename = record.record_id or None
                    continue

                # content is in bytes here
                content = record.reader.read()

                # keep track of the number of records processed
                records_cnt += 1

                # url
                target_uri = record.headers.get("WARC-Target-URI")
                url = (clean_url(target_uri) or target_uri).split("//", 1)[-1]

                # check for duplicate URLs
                if check_duplicate_urls:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                # metadata
                http_headers = record.http_headers.asdict()
                ctype = http_headers.get("Content-Type", "").split(";", 1)[0]
                header_date = cls._parse_warc_timestamp(t) if (t := http_headers.get("Date")) else None
                payload_id = record.headers.get("WARC-Payload-Digest").split(":", 2)[1].lower()
                header_timestamp = cls._format_to_dolma_timestamp(header_date) if header_date else warc_timestamp

                metadata = dict(
                    warc_url=target_uri,
                    url=url,
                    html=content,
                    warc_date=warc_timestamp,
                    warc_filename=warc_filename or "",
                    content_type=ctype,
                    uncompressed_offset=record.stream_pos,
                )
                doc = InputSpecWithMetadataAndAttributes(
                    source=source_name,
                    version=source_version,
                    id=payload_id,
                    text="",  # this will come later
                    metadata=metadata,
                    added=date_now_str,
                    created=header_timestamp,
                )

                # these are the properties extracted from
                pre_attributes = {name: tagger.tag(doc) for name, tagger in pre_taggers.items()}
                if skip_no_pre_taggers and not sum(map(len, pre_attributes.values())):
                    continue

                # extract text
                doc.text = linearizer.linearize(content=content, encoding=record.http_charset)

                # these are the properties extracted from the HTML content
                post_attributes = {name: tagger.tag(doc) for name, tagger in post_taggers.items()}
                if skip_no_post_taggers and not sum(map(len, post_attributes.values())):
                    continue

                doc.attributes = {
                    f"{t_name}__{t_name}__{make_variable_name(a_name)}": attr_values
                    for t_name, attributes in chain(pre_attributes.items(), post_attributes.items())
                    for a_name, attr_values in attributes.items()
                }

                if not store_html_in_metadata:
                    doc.metadata.pop("html", None)  # type: ignore

                output_file.write(encoder.encode(doc) + b"\n")  # pyright: ignore

                extracted_cnt += 1

                if extracted_cnt % update_interval == 0:
                    # update the progress bar every update_interval documents to prevent buffering
                    cls.increment_progressbar(queue, records=records_cnt, extracted=extracted_cnt)

                    delta_time = -(elapsed_time - (elapsed_time := time()))

                    # reset the counters
                    extracted_cnt = 0
                    records_cnt = 0

                    # no need to update the progress bar if too full or frequency is above 10 Hz
                    if queue.qsize() >= multiprocessing.cpu_count() or delta_time < 1e-1:
                        # double the update interval if the queue is full
                        update_interval *= 2

        cls.increment_progressbar(queue, files=1, records=records_cnt, extracted=extracted_cnt)


def create_and_run_warc_pipeline(
    documents: Union[str, List[str]],
    destination: Union[str, List[str]],
    source_name: str,
    metadata: Union[None, str, List[str]] = None,
    debug: bool = False,
    seed: int = 0,
    ignore_existing: bool = False,
    skip_on_failure: bool = False,
    num_processes: int = 1,
    pre_taggers: Optional[List[str]] = None,
    linearizer_name: str = "resiliparse",
    post_taggers: Optional[List[str]] = None,
    store_html_in_metadata: bool = False,
    skip_no_pre_taggers: bool = False,
    skip_no_post_taggers: bool = False,
    skip_source_glob: bool = False,
    backoff_max_time: Optional[int] = None,
    backoff_max_tries: Optional[int] = 10,
    compression: Optional[str] = "zst",
    check_duplicate_urls: bool = False,
):
    with ExitStack() as stack:
        if metadata is None:
            if isinstance(destination, str):
                metadata = stack.enter_context(tempfile.TemporaryDirectory())
            else:
                metadata = [stack.enter_context(tempfile.TemporaryDirectory()) for _ in range(len(destination))]

        all_src_paths = []
        all_dst_paths = []
        all_meta_paths = []

        if isinstance(destination, str) and isinstance(metadata, str):
            for src_pattern in [documents] if isinstance(documents, str) else documents:
                all_src_paths.extend([src_pattern] if skip_source_glob else list(glob_path(src_pattern)))

            all_dst_paths.extend(_make_paths_from_prefix(paths=all_src_paths, prefix=destination))
            all_meta_paths.extend(_make_paths_from_prefix(paths=all_src_paths, prefix=metadata))

        elif isinstance(destination, list) and isinstance(metadata, list):
            if not isinstance(documents, list):
                raise ValueError("documents must be a list of strings")
            if not isinstance(metadata, list):
                raise ValueError("metadata must be a list of strings")
            if len(documents) != len(destination):
                raise ValueError("documents and destination must have the same length")
            if len(metadata) != len(destination):
                raise ValueError("metadata and destination must have the same length")

            for src_pattern, dst_pattern, meta_pattern in zip(documents, destination, metadata):
                src_paths = [src_pattern] if skip_source_glob else list(glob_path(src_pattern))
                all_src_paths.extend(src_paths)
                all_dst_paths.extend(_make_paths_from_prefix(paths=src_paths, prefix=dst_pattern))
                all_meta_paths.extend(_make_paths_from_prefix(paths=src_paths, prefix=meta_pattern))
        else:
            raise ValueError("destination must be a string or a list of strings")

        processor = WarcProcessor(
            source_prefix=all_src_paths,
            destination_prefix=all_dst_paths,
            metadata_prefix=all_meta_paths,
            debug=debug,
            seed=seed,
            skip_source_glob=skip_source_glob,
            ignore_existing=ignore_existing,
            backoff_max_tries=backoff_max_tries,
            backoff_max_time=backoff_max_time,
            backoff_exceptions=(Exception,),
            num_processes=num_processes,
            shuffle_src_paths=False,
        )

        from dolma.core.runtime import profiler

        with profiler("temp/test.prof", human_readable=False):
            processor(
                skip_on_failure=skip_on_failure,
                store_html_in_metadata=store_html_in_metadata,
                linearizer_name=linearizer_name,
                pre_taggers=pre_taggers,
                post_taggers=post_taggers,
                skip_no_pre_taggers=skip_no_pre_taggers,
                skip_no_post_taggers=skip_no_post_taggers,
                source_name=source_name,
                compression=compression,
                debug=debug,
                check_duplicate_urls=check_duplicate_urls,
            )
