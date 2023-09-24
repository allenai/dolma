import datetime
import io
import multiprocessing
import re
from typing import TYPE_CHECKING, Dict, Generator, Optional, Type, Union

import msgspec
import smart_open
from charset_normalizer import detect
from necessary import necessary

from ..parallel import BaseParallelProcessor, QueueType
from .html import HTML_EXTRACTORS, BaseHtmlExtractor
from .license import LICENSE_EXTRACTORS, BaseLicenseExtractor
from .types import WarcDocument, WarcDocumentMetadata
from .utils import raise_dependency_error

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
        extracted: int = 0,
    ) -> Dict[str, int]:
        """Records (documents) and records are the units we use to track progress."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, records=records, extracted=extracted)

    @classmethod
    def _record_iterator(
        cls, stream: Union[io.TextIOWrapper, io.BytesIO]
    ) -> Generator["ArcWarcRecord", None, None]:
        """Iterate over the records in a WARC file."""

        for record in ArchiveIterator(stream):
            yield record

    @classmethod
    def _decode_content(cls, content: bytes) -> Union[str, None]:
        if not (encoding := detect(content)["encoding"]):
            return None
        return content.decode(str(encoding))

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
        skip_unknown_license: bool = kwargs.get("skip_unknown_license", False)

        # create the html extractor
        html_extractor_name: str = kwargs.get("html_extractor", "trafilatura")
        html_extractor_cls: Union[Type[BaseHtmlExtractor], None] = HTML_EXTRACTORS.get(html_extractor_name)
        if html_extractor_cls is None:
            raise ValueError(f"Extractor {kwargs.get('extractor', 'trafilatura')} is not supported.")
        html_extractor = html_extractor_cls(**kwargs.get("extractor_kwargs", {}))

        # create the license extractor
        license_extr_name: str = kwargs.get("license_extractor", "cc_regex")
        license_extr_cls: Union[Type[BaseLicenseExtractor], None] = LICENSE_EXTRACTORS.get(license_extr_name)
        if license_extr_cls is None:
            raise ValueError(f"License extractor {license_extr_name} is not supported.")
        license_extractor = license_extr_cls()

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
                    content = record.content_stream().read()
                    cc_license = license_extractor(content=content)

                    records_cnt += 1

                    if skip_unknown_license and cc_license.type_ == "unk":
                        continue

                    str_content = cls._decode_content(content)
                    if str_content is None:
                        continue

                    text = html_extractor(content=str_content)

                    # metadata
                    content_type, *_ = record.http_headers.get_header("Content-Type", record.content_type).split(
                        ";"
                    )
                    date = record.http_headers.get_header("Date")
                    target_uri = record.rec_headers.get_header("WARC-Target-URI")
                    payload_id = record.rec_headers.get_header("WARC-Payload-Digest").split(":")[1].lower()

                    metadata = WarcDocumentMetadata(
                        url=target_uri,
                        content=str_content,
                        warc_date=cls._format_to_dolma_timestamp(warc_date),
                        warc_filename=warc_filename or "",
                        content_type=content_type,
                        cc_license=cc_license,
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
