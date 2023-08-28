import datetime
import io
import logging
import multiprocessing
import os
import re
import tempfile
from typing import Dict, Generator, Literal, Optional, Union

import msgspec
import smart_open
from necessary import necessary

from .errors import DolmaFatalError
from .parallel import BaseParallelProcessor, QueueType

with necessary("warcio", soft=True) as WARCIO_AVAILABLE:
    from warcio.archiveiterator import ArchiveIterator
    from warcio.recordloader import ArcWarcRecord

with necessary("justext", soft=True) as JUSTEXT_AVAILABLE:
    from justext import justext

with necessary("newspaper3k", soft=True) as NEWSPAPER_AVAILABLE:
    from newspaper import Article

with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    import dateparser

with necessary("trafilatura", soft=True) as TRAFILATURA_AVAILABLE:
    from trafilatura import extract as trafilatura_extract


DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %Z",
    "%Y-%m-%dT%H:%M:%SZ",
]


class WarcDocumentMetadata(msgspec.Struct):
    content: bytes
    url: str
    content_type: str
    warc_date: str
    warc_filename: str


class WarcDocument(msgspec.Struct):
    """A document extracted from a WARC file."""

    source: str
    id: str
    text: str
    added: str
    created: str
    metadata: WarcDocumentMetadata


def format_timestamp(ts: Optional[datetime.datetime] = None) -> str:
    if ts is None:
        ts = datetime.datetime.now()
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def parse_warc_timestamp(ts: Optional[str]) -> datetime.datetime:
    if not ts:
        return datetime.datetime.now()

    return dateparser.parse(date_string=ts, date_formats=DATE_FORMATS) or datetime.datetime.now()


def raise_dependency_error(package: str):
    raise DolmaFatalError(
        f"Package {package} is required to run this processor. "
        f"Please install it with `pip install dolma[warc]`."
    )


class WarcProcessor(BaseParallelProcessor):
    """Processes WARC files, like the ones used by Common Crawl, in parallel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert WARCIO_AVAILABLE, raise_dependency_error("warcio")
        assert TRAFILATURA_AVAILABLE, raise_dependency_error("trafilatura")
        assert JUSTEXT_AVAILABLE, raise_dependency_error("justext")
        assert NEWSPAPER_AVAILABLE, raise_dependency_error("newspaper3k")
        assert DATEPARSER_AVAILABLE, raise_dependency_error("dateparser")

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
    def _extract_text(
        cls, content: bytes, backend: Union[Literal["trafilatura"], Literal["justext"], Literal["newspaper3k"]]
    ) -> Union[str, None]:
        if not content.strip():
            return None

        if backend == "trafilatura":
            output = trafilatura_extract(
                content, include_comments=False, include_links=False, include_tables=False, no_fallback=True
            )
        elif backend == "justext":
            output = "\n".join(
                paragraph.text for paragraph in justext(content, frozenset()) if not paragraph.is_boilerplate
            )
        elif backend == "newspaper3k":
            (article := Article("")).set_html(content)
            article.parse()
            output = article.text
        else:
            raise DolmaFatalError(f"Unknown backend: {backend}")

        return output

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

        # make the trafilatura logger quiet
        logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

        # play with extensions
        if not destination_path.endswith(".jsonl.gz"):
            destination_path = re.sub(r"(\.warc)?\.gz$", "", destination_path) + ".jsonl.gz"

        with smart_open.open(source_path, "rb") as warc_file, smart_open.open(
            destination_path, "wb"
        ) as output_file:
            for record in cls._record_iterator(warc_file):
                if record.rec_type == "warcinfo":
                    warc_date = parse_warc_timestamp(record.rec_headers.get_header("WARC-Date"))
                    warc_filename = record.rec_headers.get_header("WARC-Filename")

                elif record.rec_type == "response":
                    content_type, *_ = record.http_headers.get_header("Content-Type", record.content_type).split(
                        ";"
                    )

                    date = record.http_headers.get_header("Date")

                    target_uri = record.rec_headers.get_header("WARC-Target-URI")
                    payload_id = record.rec_headers.get_header("WARC-Payload-Digest").split(":")[1].lower()

                    content = record.content_stream().read()
                    text = cls._extract_text(content=content, backend="trafilatura")

                    metadata = WarcDocumentMetadata(
                        url=target_uri,
                        content=content,
                        warc_date=format_timestamp(warc_date),
                        warc_filename=warc_filename or "",
                        content_type=content_type,
                    )

                    document = WarcDocument(
                        source="warc",
                        id=payload_id,
                        created=format_timestamp(parse_warc_timestamp(date)),
                        added=format_timestamp(date_now),
                        text=text or "",
                        metadata=metadata,
                    )

                    output_file.write(encoder.encode(document) + b"\n")  # type: ignore

                    records_cnt += 1

                    if records_cnt % update_interval == 0:
                        # update the progress bar every update_interval documents to prevent buffering
                        cls.increment_progressbar(queue, records=update_interval)

                        if queue.qsize() >= multiprocessing.cpu_count():
                            # double the update interval if the queue is full
                            update_interval *= 2

        cls.increment_progressbar(queue, files=1, records=records_cnt % update_interval)


if __name__ == "__main__":
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    temp = "s3://ai2-russella/crawl-data/CC-MAIN-2019-18/segments/1555578517558.8/warc/CC-MAIN-20190418101243-20190418122311-00016.warc.gz"

    with tempfile.TemporaryDirectory() as tempdir:
        processor = WarcProcessor(
            source_prefix=temp,
            destination_prefix="s3://ai2-llm/experimental/cc-main-2019-18/v0/documents",
            metadata_prefix=tempdir,
            debug=False,
            num_processes=8,
        )
        processor()
