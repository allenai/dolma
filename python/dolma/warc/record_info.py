import datetime
from typing import TYPE_CHECKING, Optional

from fastwarc.warc import WarcRecordType
from necessary import necessary

with necessary("dateparser", soft=True) as DATEPARSER_AVAILABLE:
    if DATEPARSER_AVAILABLE or TYPE_CHECKING:
        import dateparser


DATE_FORMATS = ["%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ"]


class WarcRecordInfo:
    def __init__(self, record):
        self.record = record

        if not self.is_valid:
            return None

        self.payload_id = record.headers.get("WARC-Payload-Digest").split(":")[1].lower()
        self.target_uri = record.headers.get("WARC-Target-URI")

        if record.record_type == WarcRecordType.response:
            ctype, *_ = (record.http_headers.get("Content-Type") or "").split(";")
            self.ctype = ctype
            self.date = WarcRecordInfo._parse_warc_timestamp(record.http_headers.get("Date"))
        elif record.record_type == WarcRecordType.resource:
            self.ctype, *_ = (record.headers.get("Content-Type") or "").split(";")
            self.date = WarcRecordInfo._parse_warc_timestamp(record.headers.get("WARC-Date"))
        else:
            raise ValueError(f"Unsupported record type: {record.record_type}")

    @property
    def is_valid(self) -> bool:
        if not self.record.headers.get("WARC-Payload-Digest"):
            return False

        if not self.record.headers.get("WARC-Target-URI"):
            return False

        return True

    @staticmethod
    def _parse_warc_timestamp(timestamp_str: Optional[str]) -> datetime.datetime:
        """Parse a WARC timestamp into a datetime object."""
        if not timestamp_str:
            return datetime.datetime.now()

        return dateparser.parse(date_string=timestamp_str, date_formats=DATE_FORMATS) or datetime.datetime.now()
