import datetime
import multiprocessing
from typing import TYPE_CHECKING, Dict, List, Optional

import msgspec
import smart_open
from charset_normalizer import detect
from necessary import necessary

from dolma.core.paths import join_path, split_ext

from ..core.parallel import BaseParallelProcessor, QueueType
from .documents import WarcDocument, WarcDocumentMetadata
from .extractors import ExtractorInputType, partition_extractors
from .registries import ExtractorRegistry, LinearizerRegistry
from .utils import UrlNormalizer, raise_warc_dependency_error

with necessary("fastwarc", soft=True) as FASTWARC_AVAILABLE:
    if FASTWARC_AVAILABLE or TYPE_CHECKING:
        from fastwarc.warc import ArchiveIterator, WarcRecordType

with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    if DATEPARSER_AVAILABLE or TYPE_CHECKING:
        import dateparser


DATE_FORMATS = ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]


class WarcProcessor(BaseParallelProcessor):
    """Processes WARC files, like the ones used by Common Crawl, in parallel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not FASTWARC_AVAILABLE:
            raise_warc_dependency_error("fastwarc")
        if not DATEPARSER_AVAILABLE:
            raise_warc_dependency_error("dateparser")

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

        return dateparser.parse(date_string=timestamp_str, date_formats=DATE_FORMATS) or datetime.datetime.now()

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

        # interval at which to update the progress bar; will double if it gets too full
        update_interval = 1

        # hold the number of records processed in this variable
        records_cnt = 0
        extracted_cnt = 0

        # encoder
        encoder = msgspec.json.Encoder()

        # skip license if unknown
        skip_if_empty_heuristics: bool = kwargs.get("skip_if_empty_heuristics") or False
        store_html_in_metadata: bool = kwargs.get("store_html_in_metadata") or False

        # get the name of this source
        source_name = kwargs.get("source_name", None)
        if not isinstance(source_name, str):
            raise ValueError(f"source_name must be a string, not {source_name} ({type(source_name)})")

        # create the html extractor
        linearizer_name: str = kwargs.get("linearizer") or "resiliparse"
        linearizer = LinearizerRegistry.get(linearizer_name)()

        # create the license extractor
        extractors_names: List[str] = kwargs.get("extractors") or []
        extractors = partition_extractors([ExtractorRegistry.get(name)() for name in extractors_names])

        # url normalizer
        url_normalizer = UrlNormalizer()

        # derive the destination path if it is not provided by splitting out all the
        # extensions, removing gz and warc, and adding jsonl.gz
        if not destination_path.endswith(".jsonl.gz"):
            prot, base_dst, extension = split_ext(destination_path)
            extension = extension.replace(".gz", "").replace(".warc", "") + ".jsonl.gz"
            destination_path = join_path(prot, *base_dst[:-1], base_dst[-1] + extension)

        with smart_open.open(source_path, "rb") as warc_file, smart_open.open(
            destination_path, "wb"
        ) as output_file:
            it = ArchiveIterator(warc_file, record_types=WarcRecordType.response | WarcRecordType.warcinfo)
            for record in it:
                if record.record_type == WarcRecordType.warcinfo:
                    warc_date = record.record_date or None
                    warc_filename = record.record_id or None
                    continue

                # content is in bytes here
                content = record.reader.read()

                # handling decoding here; we try to decode the content using the charset (fast),
                # and only if that fails, we use the chardet library to detect the encoding (slow)
                decoded_content = ""
                if record.http_charset:
                    try:
                        decoded_content = content.decode(record.http_charset).strip()
                    except UnicodeDecodeError:
                        decoded_content = ""
                if not decoded_content and (encoding := detect(content)["encoding"]):
                    decoded_content = content.decode(str(encoding)).strip()
                if not decoded_content:
                    continue

                # keep track of the number of records processed
                records_cnt += 1

                # these are the properties extracted from the HTML content
                properties = [
                    prop
                    for extractor in extractors[ExtractorInputType.HTML]
                    for prop in extractor.extract(content=decoded_content)
                ]

                if skip_if_empty_heuristics and not properties:
                    continue

                text = linearizer.linearize(content=decoded_content)

                # these are the properties extracted from the linearized plain text
                properties.extend(
                    [
                        prop
                        for extractor in extractors[ExtractorInputType.PLAIN]
                        for prop in extractor.extract(content=decoded_content)
                    ]
                )

                # metadata
                ctype, *_ = (record.http_headers.get("Content-Type") or "").split(";")
                date = cls._parse_warc_timestamp(record.http_headers.get("Date"))
                target_uri = record.headers.get("WARC-Target-URI")
                payload_id = record.headers.get("WARC-Payload-Digest").split(":")[1].lower()

                metadata = WarcDocumentMetadata(
                    url=target_uri,
                    norm_url=url_normalizer(target_uri),
                    html=decoded_content if store_html_in_metadata else None,
                    warc_date=cls._format_to_dolma_timestamp(warc_date),
                    warc_filename=warc_filename or "",
                    content_type=ctype,
                    properties=properties,
                )

                document = WarcDocument(
                    source="warc",
                    id=payload_id,
                    created=cls._format_to_dolma_timestamp(date),
                    added=cls._format_to_dolma_timestamp(date_now),
                    text=text or "",
                    metadata=metadata,
                )

                output_file.write(encoder.encode(document) + b"\n")  # pyright: ignore

                extracted_cnt += 1

                if extracted_cnt % update_interval == 0:
                    # update the progress bar every update_interval documents to prevent buffering
                    cls.increment_progressbar(queue, records=records_cnt, extracted=extracted_cnt)

                    # reset the counters
                    extracted_cnt = 0
                    records_cnt = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

        cls.increment_progressbar(queue, files=1, records=records_cnt, extracted=extracted_cnt)
