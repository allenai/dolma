import datetime
import io
import multiprocessing
import re
from typing import Dict, Generator, Optional, Type, Union, TYPE_CHECKING

import msgspec
import smart_open
from necessary import necessary


from .types import WarcDocument, WarcDocumentMetadata
from .utils import raise_dependency_error
from .html import HTML_EXTRACTORS, BaseHtmlExtractor
from ..parallel import BaseParallelProcessor, QueueType

with necessary("warcio", soft=True) as WARCIO_AVAILABLE:
    if WARCIO_AVAILABLE or TYPE_CHECKING:
        from warcio.archiveiterator import ArchiveIterator
        from warcio.recordloader import ArcWarcRecord

with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    if DATEPARSER_AVAILABLE or TYPE_CHECKING:
        import dateparser


DATE_FORMATS = ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]


class WarcProcessor(BaseParallelProcessor):
    """Processes WARC files, like the ones used by Common Crawl, in parallel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert WARCIO_AVAILABLE, raise_dependency_error("warcio")
        assert DATEPARSER_AVAILABLE, raise_dependency_error("dateparser")

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
    ) -> Dict[str, int]:
        """Records (documents) and records are the units we use to track progress."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, records=records)

    @classmethod
    def _record_iterator(
        cls, stream: Union[io.TextIOWrapper, io.BytesIO]
    ) -> Generator["ArcWarcRecord", None, None]:
        """Iterate over the records in a WARC file."""

        for record in ArchiveIterator(stream):
            yield record

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

        # encoder
        encoder = msgspec.json.Encoder()

        # create the html extractor
        extractor_name: str = kwargs.get("extractor", "trafilatura")
        extractor_cls: Union[Type[BaseHtmlExtractor], None] = HTML_EXTRACTORS.get(extractor_name)
        if extractor_cls is None:
            raise ValueError(f"Extractor {kwargs.get('extractor', 'trafilatura')} is not supported.")
        assert extractor_name in HTML_EXTRACTORS, f"Extractor {extractor_name} is not supported."
        extractor = extractor_cls(**kwargs.get("extractor_kwargs", {}))

        # play with extensions
        if not destination_path.endswith(".jsonl.gz"):
            destination_path = re.sub(r"(\.warc)?\.gz$", "", destination_path) + ".jsonl.gz"

        with smart_open.open(source_path, "rb") as warc_file, smart_open.open(
            destination_path, "wb"
        ) as output_file:
            for record in cls._record_iterator(warc_file):
                if record.rec_type == "warcinfo":
                    warc_date = cls._parse_warc_timestamp(record.rec_headers.get_header("WARC-Date"))
                    warc_filename = record.rec_headers.get_header("WARC-Filename")

                elif record.rec_type == "response":
                    cont_type, *_ = record.http_headers.get_header("Content-Type", record.content_type).split(";")

                    date = record.http_headers.get_header("Date")

                    target_uri = record.rec_headers.get_header("WARC-Target-URI")
                    payload_id = record.rec_headers.get_header("WARC-Payload-Digest").split(":")[1].lower()

                    content = record.content_stream().read()
                    text = extractor(content=content)

                    metadata = WarcDocumentMetadata(
                        url=target_uri,
                        content=content,
                        warc_date=cls._format_to_dolma_timestamp(warc_date),
                        warc_filename=warc_filename or "",
                        content_type=cont_type,
                    )

                    document = WarcDocument(
                        source="warc",
                        id=payload_id,
                        created=cls._format_to_dolma_timestamp(cls._parse_warc_timestamp(date)),
                        added=cls._format_to_dolma_timestamp(date_now),
                        text=text or "",
                        metadata=metadata,
                    )

                    output_file.write(encoder.encode(document) + b"\n")  # pyright: ignore

                    records_cnt += 1

                    if records_cnt % update_interval == 0:
                        # update the progress bar every update_interval documents to prevent buffering
                        cls.increment_progressbar(queue, records=update_interval)

                        if queue.qsize() >= multiprocessing.cpu_count():
                            # double the update interval if the queue is full
                            update_interval *= 2

        cls.increment_progressbar(queue, files=1, records=records_cnt % update_interval)
