import datetime
import hashlib
import random
import tempfile
from contextlib import ExitStack
from functools import reduce
from itertools import chain
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Union

import msgspec
import smart_open
from necessary import necessary

from ..core.data_types import DocumentWithMetadataAndAttributes
from ..core.parallel import BaseParallelProcessor
from ..core.paths import glob_path, join_path, make_relative, split_ext, split_path
from ..core.progressbar import BaseProgressBar, QueueType
from ..core.registry import TaggerRegistry
from ..core.runtime import _make_paths_from_prefix
from ..core.utils import format_span_key, format_span_output, make_variable_name
from .iterator import SimpleWarcIterator
from .linearizers import LinearizerRegistry
from .utils import raise_warc_dependency_error

with necessary("fastwarc", soft=True) as FASTWARC_AVAILABLE:
    if FASTWARC_AVAILABLE or TYPE_CHECKING:
        from fastwarc.warc import WarcRecordType  # pylint: disable=no-name-in-module

with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    if DATEPARSER_AVAILABLE or TYPE_CHECKING:
        import dateparser

with necessary("courlan", soft=True) as COURLAN_AVAILABLE:
    if COURLAN_AVAILABLE or TYPE_CHECKING:
        from courlan import clean_url  # pyright: ignore


DATE_FORMATS = ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]


class WarcProgressBar(BaseProgressBar):
    records: int = 0
    duplicates: int = 0
    extracted: int = 0
    files: int = 0
    # attempts: int = 0


class WarcProcessor(BaseParallelProcessor):
    """Processes WARC files, like the ones used by Common Crawl, in parallel."""

    PROGRESS_BAR_CLS = WarcProgressBar

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert FASTWARC_AVAILABLE, raise_warc_dependency_error("fastwarc")
        assert DATEPARSER_AVAILABLE, raise_warc_dependency_error("dateparser")
        assert COURLAN_AVAILABLE, raise_warc_dependency_error("courlan")

    @staticmethod
    def _format_to_dolma_timestamp(timestamp: Optional[datetime.datetime] = None) -> str:
        """Format a timestamp as a string using near ISO-8601 format."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"

    @staticmethod
    def _parse_warc_timestamp(timestamp_str: Optional[str]) -> datetime.datetime:
        """Parse a WARC timestamp into a datetime object."""
        if not timestamp_str:
            return datetime.datetime.now()

        for fmt in DATE_FORMATS:
            try:
                return datetime.datetime.strptime(timestamp_str, fmt)
            except ValueError:
                pass

        return dateparser.parse(date_string=timestamp_str) or datetime.datetime.now()

    @staticmethod
    def _get_destination_path(paths: List[str], new_ext: Optional[str] = None) -> str:
        """Given a set of paths, compute the actual destination paths.
        If `paths` contain a single file, return the same path; otherwise, find the common prefix
        shared by all files, and create a destination path by appending hash of the paths to the prefix and
        keeping the same extension as the first path.
        """

        if new_ext:
            # replace the extension of all paths with the new extension if provided
            paths = [join_path((s := split_ext(p))[0], *s[1]) + new_ext for p in paths]

        # if there is only one destination path, return it
        if len(paths) == 1:
            return paths[0]

        # get the common prefix
        common_prefix, rest = make_relative(paths)
        common_prot, common_parts = split_path(common_prefix)

        # get the extension of the first path
        _, _, extension = split_ext(rest[0])

        # create the destination path
        hash_str = reduce(
            lambda h, p: h.update(p.encode()) or h, paths, hashlib.sha1()  # type: ignore
        ).hexdigest()
        destination_path = join_path(common_prot, *common_parts, f"{hash_str}{extension}")

        return destination_path

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: QueueType,
        **kwargs,
    ):
        """Extracting documents from a single WARC file is done by calling process_batch with a single file."""
        return cls.process_batch(
            source_paths=[source_path],
            destination_paths=[destination_path],
            queue=queue,
            kwargs=[kwargs],
        )

    @classmethod
    def process_batch(
        cls,
        source_paths: List[str],
        destination_paths: List[str],
        queue: QueueType,
        kwargs: List[Dict[str, Any]],
    ):
        warc_date: Optional[datetime.datetime] = None
        warc_filename: Optional[str] = None
        date_now = datetime.datetime.now()
        date_now_str = cls._format_to_dolma_timestamp(date_now)

        # encoder
        encoder = msgspec.json.Encoder()

        with ExitStack() as stack:
            pbar = stack.enter_context(WarcProgressBar(queue))

            # delay start if requested
            delay: float = kwargs[0].get("delay_start", 0.0)
            if delay > 0:
                time.sleep(random.random() * delay)

            # get compression format; it's slightly awkward that we have to check that is the same for all
            # the single kwargs, but decent sanity check.
            all_compression_ext = {kw.get("compression", None) or "zst" for kw in kwargs}
            assert len(all_compression_ext) == 1, "All compression formats must be the same"
            cpz_ext = all_compression_ext.pop()

            # we need to figure out where to write the output of extraction. We operate slightly differently
            # depending on whether we are working with a single destination or multiple destinations.
            destination_path = cls._get_destination_path(paths=destination_paths, new_ext=f".jsonl.{cpz_ext}")

            # this is the file where we will write the output
            output_file = stack.enter_context(smart_open.open(destination_path, "wb"))

            for src_path, src_kwargs in zip(source_paths, kwargs):
                # get the name and version of this source
                source_name = src_kwargs.get("source_name", None)
                source_version = src_kwargs.get("source_version", "v0")
                if not isinstance(source_name, str):
                    raise ValueError(f"source_name must be a string, not {source_name} ({type(source_name)})")

                # create any tagger that runs before html extraction
                pre_taggers_names: List[str] = src_kwargs.get("pre_taggers") or []
                pre_taggers = {make_variable_name(name): TaggerRegistry.get(name)() for name in pre_taggers_names}
                pre_taggers_mode = src_kwargs.get("pre_taggers_mode") or "any"
                assert pre_taggers_mode in ["any", "all"], "pre_mode must be 'any' or 'all'"

                # create the html extractor
                linearizer_name: str = src_kwargs.get("linearizer_name") or "resiliparse"
                linearizer = LinearizerRegistry.get(linearizer_name)()

                # faster extractor for first stage
                fast_linearizer_name: str = src_kwargs.get("fast_linearizer_name") or linearizer_name
                fast_linearizer = LinearizerRegistry.get(fast_linearizer_name)()

                # minimum content lengths
                min_raw_length = int(src_kwargs.get("min_raw_length") or 0)
                min_text_length = int(src_kwargs.get("min_text_length") or 0)

                # check for duplicate URLs
                skip_duplicate_urls = bool(src_kwargs.get("skip_duplicate_urls", None) or False)
                seen_urls: Set[str] = set()

                # create any tagger that runs after html extraction
                post_taggers_names: List[str] = src_kwargs.get("post_taggers") or []
                post_taggers = {
                    make_variable_name(name): TaggerRegistry.get(name)() for name in post_taggers_names
                }
                post_taggers_mode = src_kwargs.get("post_taggers_mode") or "any"
                assert post_taggers_mode in ["any", "all"], "post_mode must be 'any' or 'all'"

                # whether to store html in metadata after extraction
                store_html_in_meta: bool = src_kwargs.get("store_html_in_metadata") or False

                # whether to store attribute spans in metadata after extraction
                store_spans_in_meta = int(src_kwargs.get("store_attribute_spans_in_metadata", -1))

                # whether to skip this document if pre-taggers find nothing
                skip_no_pre_taggers: bool = src_kwargs.get("skip_no_pre_taggers") or False

                # whether to skip this document if post-taggers find nothing
                skip_no_post_taggers: bool = src_kwargs.get("skip_no_post_taggers") or False

                # open the WARC file
                it = stack.enter_context(SimpleWarcIterator(path=src_path))

                # in case there's no warcinfo record, we will set these to None
                warc_date = warc_filename = None
                warc_timestamp = ""

                for record in it:
                    if record.record_type == WarcRecordType.warcinfo:
                        warc_date = record.record_date or None
                        warc_timestamp = cls._format_to_dolma_timestamp(warc_date) or ""
                        warc_filename = record.record_id or None
                        continue

                    # content is in bytes here
                    ct = record.reader.read()

                    # keep track of the number of records processed
                    pbar.records += 1

                    # below min length
                    if len(ct) < min_raw_length:
                        continue

                    # url
                    target_uri = record.headers.get("WARC-Target-URI")
                    url = (clean_url(target_uri) or target_uri).split("//", 1)[-1]

                    # check for duplicate URLs
                    if skip_duplicate_urls:
                        if url in seen_urls:
                            pbar.duplicates += 1
                            continue
                        seen_urls.add(url)

                    # metadata
                    http_headers = record.http_headers.asdict()
                    ctype = http_headers.get("Content-Type", "").split(";", 1)[0]
                    header_date = cls._parse_warc_timestamp(t) if (t := http_headers.get("Date")) else None
                    payload_id = record.headers.get("WARC-Payload-Digest").split(":", 2)[1].lower()
                    header_timestamp = (
                        cls._format_to_dolma_timestamp(header_date) if header_date else warc_timestamp
                    )

                    metadata = dict(
                        warc_url=target_uri,
                        url=url,
                        html=ct,
                        warc_date=warc_timestamp,
                        warc_filename=warc_filename or "",
                        content_type=ctype,
                        uncompressed_offset=record.stream_pos,
                    )
                    doc = DocumentWithMetadataAndAttributes(
                        source=source_name,
                        version=source_version,
                        id=payload_id,
                        text="",  # this will come later
                        metadata=metadata,
                        attributes={},  # this will come later
                        added=date_now_str,
                        created=header_timestamp,
                    )

                    # these are the properties extracted from the HTML content
                    pre_attributes = {name: tagger.predict(doc) for name, tagger in pre_taggers.items()}
                    if not skip_no_pre_taggers:
                        pass
                    elif pre_taggers_mode == "any" and not any(r.spans for r in pre_attributes.values()):
                        continue
                    elif pre_taggers_mode == "all" and not all(r.spans for r in pre_attributes.values()):
                        continue

                    # extract text
                    doc.text = fast_linearizer.linearize(content=ct, encoding=record.http_charset)

                    # below min length
                    if len(doc.text) < min_text_length:
                        continue

                    # these are the properties extracted from the HTML content
                    post_attributes = {name: tagger.predict(doc) for name, tagger in post_taggers.items()}
                    if not skip_no_post_taggers:
                        pass
                    elif post_taggers_mode == "any" and not any(r.spans for r in post_attributes.values()):
                        continue
                    elif post_taggers_mode == "all" and not all(r.spans for r in post_attributes.values()):
                        continue

                    for attr_name, attr_result in chain(pre_attributes.items(), post_attributes.items()):
                        for attr_span in attr_result.spans:
                            attr_key = format_span_key(attr_name, attr_name, attr_span)
                            attr_val = format_span_output(attr_span)
                            doc.attributes.setdefault(attr_key, []).append(attr_val)

                            # in case we want to store the exact attribute span
                            if store_spans_in_meta >= 0:
                                mct = attr_span.select(doc, left=store_spans_in_meta, right=store_spans_in_meta)
                                # if it is a bunch of bytes, we decode to a string else we keep it as is
                                mct = mct.decode("utf-8", errors="ignore") if isinstance(mct, bytes) else mct
                                doc.metadata.setdefault("attribute_spans", {}).setdefault(attr_key, []).append(mct)

                    if not store_html_in_meta:
                        doc.metadata.pop("html", None)

                    if fast_linearizer_name != linearizer_name:
                        doc.metadata["fast_linearizer_text"] = doc.text
                        doc.text = linearizer.linearize(content=ct, encoding=record.http_charset)

                        if len(doc.text) < min_text_length:
                            # check again if the text is below the minimum length
                            continue

                    output_file.write(encoder.encode(doc.to_spec()) + b"\n")  # pyright: ignore
                    pbar.extracted += 1
                pbar.files += 1


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
    backoff_max_time: Optional[float] = None,
    backoff_max_tries: int = 10,
    batch_size: int = 1,
    compression: str = "zst",
    fast_linearizer_name: Optional[str] = None,
    linearizer_name: str = "resiliparse",
    min_raw_length: int = 0,
    min_text_length: int = 0,
    post_taggers_mode: str = "any",
    post_taggers: Optional[List[str]] = None,
    pre_taggers_mode: str = "any",
    pre_taggers: Optional[List[str]] = None,
    progress_bar_mode: Literal["tqdm", "logger"] = "tqdm",
    skip_duplicate_urls: bool = False,
    skip_no_post_taggers: bool = False,
    skip_no_pre_taggers: bool = False,
    skip_source_glob: bool = False,
    store_attribute_spans_in_metadata: int = -1,
    store_html_in_metadata: bool = False,
    delay_start: float = 0.0,
):
    """Create and run pipeline for extracting documents from WARC files.

    Args:
        documents (str | List[str]): One or more paths to WARC files. Can be a glob pattern.
        destination (str | List[str]): One or more locations where the extracted documents will be saved;
            if only one destination is provided, it will be used for all documents; otherwise, the number of
            destinations must match the number of documents.
        source_name (str): Name to assign to the source.
        metadata (str | List[str], optional): One or more locations where the metadata will be saved;
            if not provided, metadata will be saved in temporary directories. Defaults to None.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        ignore_existing (bool, optional): Whether to ignore existing outputs and re-run the taggers.
            Defaults to False, meaning that existing outputs will be skipped.
        skip_on_failure (bool, optional): Whether to skip the document if taggers return no output.
            Defaults to False.
        num_processes (int, optional): Number of parallel processes to use. Defaults to 1.
        backoff_max_time (float, optional): How long to wait until retrying succeeds. Defaults to None, meaning
            that the maximum time is dictated by the maximum number of tries.
        backoff_max_tries (int, optional): Maximum number of tries before giving up. Defaults to 10.
        batch_size (int, optional): Number of documents to process in each batch. Defaults to 1.
        compression (str, optional): Compression format to use for the output files. Defaults to "zst".
        fast_linearizer_name (str, optional): If provided, this linearizer will be used for first stage of
            extraction. Defaults to None, meaning that the same linearizer will be used for both stages.
        linearizer_name (str, optional): Name of the HTML linearizer to use. Run `dolma list --filter linearizer`
            to get a list of all available linearizers. Defaults to "resiliparse".
        post_taggers (List[str], optional): List of taggers to run after HTML extraction. These taggers will run
            on the extracted text from the linearizer. Defaults to None.
        pre_taggers (List[str], optional): List of taggers to run before HTML extraction. These taggers will run
            on byte HTML content. Defaults to None.
        progress_bar_mode ("tqdm" | "logger", optional): Mode for the progress bar. Defaults to "tqdm".
        skip_duplicate_urls (bool, optional): Whether to skip duplicate URLs. Defaults to False.
        skip_no_post_taggers (bool, optional): Wether to skip the document if post-taggers find nothing.
            Defaults to False.
        skip_no_pre_taggers (bool, optional): Wether to skip the document if pre-taggers find nothing.
            Defaults to False.
        skip_source_glob (bool, optional): Whether to skip globbing the source path in case documents are paths
            to individual files. Defaults to False.
        store_attribute_spans_in_metadata (int, optional): Whether to store the attribute spans in the metadata
            field. Defaults to -1, meaning no attribute spans are stored. The exact attribute span is stored.
            Any value N greater than 0 indicates that N characters before and after the tagged span should be
            saved in metadata. Defaults to -1.
        store_html_in_metadata (bool, optional): Whether to store the HTML content in the metadata field.
            Defaults to False.
        delay_start (int, optional): Delay in seconds before starting the pipeline. Defaults to 0.
    """

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

        for tagger_name in chain(pre_taggers or [], post_taggers or []):
            # cache taggers
            tagger = TaggerRegistry.get(tagger_name)()
            del tagger

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
            batch_size=batch_size,
            progress_bar_mode=progress_bar_mode,
        )

        processor(
            compression=compression,
            debug=debug,
            fast_linearizer_name=fast_linearizer_name,
            linearizer_name=linearizer_name,
            min_raw_length=min_raw_length,
            min_text_length=min_text_length,
            post_taggers_mode=post_taggers_mode,
            post_taggers=post_taggers,
            pre_taggers_mode=pre_taggers_mode,
            pre_taggers=pre_taggers,
            skip_duplicate_urls=skip_duplicate_urls,
            skip_no_post_taggers=skip_no_post_taggers,
            skip_no_pre_taggers=skip_no_pre_taggers,
            skip_on_failure=skip_on_failure,
            source_name=source_name,
            store_attribute_spans_in_metadata=store_attribute_spans_in_metadata,
            store_html_in_metadata=store_html_in_metadata,
            delay_start=delay_start,
        )
