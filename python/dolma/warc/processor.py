import datetime
import multiprocessing
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import msgspec
import smart_open
from charset_normalizer import detect
from necessary import necessary

from dolma.core.data_types import Document

from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.taggers import BaseTagger
from ..taggers.language import (
    BaseLanguageTagger,
    Cld2EnglishLanguageTagger,
    Cld3LanguageTagger,
    FastTextAllLanguagesDocumentTagger,
)
from .html import HTML_EXTRACTORS, BaseHtmlExtractor, UrlNormalizer
from .license import LICENSE_EXTRACTORS, BaseLicenseExtractor
from .types import WarcDocument, WarcDocumentMetadata, WarcDocumentMetadataLanguage
from .utils import raise_warc_dependency_error

with necessary("fastwarc", soft=True) as FASTWARC_AVAILABLE:
    if FASTWARC_AVAILABLE or TYPE_CHECKING:
        from fastwarc.warc import ArchiveIterator, WarcRecordType

with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    if DATEPARSER_AVAILABLE or TYPE_CHECKING:
        import dateparser


DATE_FORMATS = ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]

LANGUAGE_TAGGERS: Dict[Union[str, None], Type[BaseTagger]] = {
    "fasttext": FastTextAllLanguagesDocumentTagger,
    "cld2": Cld2EnglishLanguageTagger,
    "cld3": Cld3LanguageTagger,
    None: BaseLanguageTagger,
}


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
        skip_unknown_license: bool = kwargs.get("skip_unknown_license") or False
        keep_html_in_metadata: bool = kwargs.get("keep_html_in_metadata") or False

        # get the name of this source
        source_name = kwargs.get("source_name", None)
        if not isinstance(source_name, str):
            raise ValueError(f"source_name must be a string, not {source_name} ({type(source_name)})")

        # create the html extractor
        html_extractor_name: str = kwargs.get("html_extractor") or "resiliparse"
        html_extractor_kwargs: Dict[str, Any] = kwargs.get("html_kwargs") or {}
        html_extractor_cls: Union[Type[BaseHtmlExtractor], None] = HTML_EXTRACTORS.get(html_extractor_name)
        if html_extractor_cls is None:
            raise ValueError(f"Extractor `{html_extractor_name}` is not supported.")
        html_extractor = html_extractor_cls(**html_extractor_kwargs)

        # create the license extractor
        license_extr_name: str = kwargs.get("license_extractor") or "null"
        license_extr_kwargs: Dict[str, Any] = kwargs.get("license_kwargs") or {}
        license_extr_cls: Union[Type[BaseLicenseExtractor], None] = LICENSE_EXTRACTORS.get(license_extr_name)
        if license_extr_cls is None:
            raise ValueError(f"License extractor {license_extr_name} is not supported.")
        license_extractor = license_extr_cls(**license_extr_kwargs)

        # Create the language tagger
        language_tagger_name: Union[str, None] = kwargs.get("language_tagger") or None
        language_tagger_kwargs: Dict[str, Any] = kwargs.get("language_tagger_kwargs") or {}
        language_tagger_cls = LANGUAGE_TAGGERS.get(language_tagger_name, None) or None
        if language_tagger_cls is None:
            raise ValueError(f"Language tagger {language_tagger_name} is not supported.")
        language_tagger = language_tagger_cls(**language_tagger_kwargs)

        # url normalizer
        url_normalizer = UrlNormalizer()

        # derive the destination path if it is not provided and the source path contains a WARC extension
        if not destination_path.endswith(".jsonl.gz"):
            destination_path = re.sub(r"(\.warc)?\.gz$", "", destination_path) + ".jsonl.gz"

        with smart_open.open(source_path, "rb") as warc_file, smart_open.open(
            destination_path, "wb"
        ) as output_file:
            for record in ArchiveIterator(
                warc_file, record_types=WarcRecordType.response | WarcRecordType.warcinfo
            ):
                if record.record_type == WarcRecordType.warcinfo:
                    warc_date = record.record_date or None
                    warc_filename = record.record_id or None
                    continue

                content = record.reader.read()

                records_cnt += 1

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

                licenses = license_extractor(content=decoded_content)

                if skip_unknown_license and not licenses:
                    continue
                text = html_extractor(content=decoded_content)

                # metadata
                ctype, *_ = (record.http_headers.get("Content-Type") or "").split(";")
                date = cls._parse_warc_timestamp(record.http_headers.get("Date"))
                target_uri = record.headers.get("WARC-Target-URI")
                payload_id = record.headers.get("WARC-Payload-Digest").split(":")[1].lower()

                # sort the predicted languages by score
                predicted_languages = [
                    WarcDocumentMetadataLanguage(code=lang.type, conf=lang.score)
                    for lang in sorted(
                        language_tagger.predict(doc=Document(text=text, id="", source="")).spans,
                        key=lambda x: x.score,
                        reverse=True,
                    )
                ]

                metadata = WarcDocumentMetadata(
                    url=target_uri,
                    norm_url=url_normalizer(target_uri),
                    content=decoded_content if keep_html_in_metadata else None,
                    warc_date=cls._format_to_dolma_timestamp(warc_date),
                    warc_filename=warc_filename or "",
                    content_type=ctype,
                    licenses=licenses,
                    languages=predicted_languages,
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
