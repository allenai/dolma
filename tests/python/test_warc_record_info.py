import datetime
from unittest import TestCase
from unittest.mock import MagicMock, patch

from fastwarc.warc import WarcRecord, WarcRecordType

from dolma.warc.record_info import WarcRecordInfo


class TestWarcRecordInfo(TestCase):
    def test_response_record(self):
        record_mock = MagicMock(spec=WarcRecord)
        record_mock.record_type = WarcRecordType.response
        record_mock.headers = {"WARC-Payload-Digest": "sha1:payload_id", "WARC-Target-URI": "http://example.com"}
        record_mock.http_headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Date": "Thu, 20 Apr 2023 12:00:00 GMT",
        }

        record_info = WarcRecordInfo(record_mock)

        self.assertEqual(record_info.payload_id, "payload_id")
        self.assertEqual(record_info.target_uri, "http://example.com")
        self.assertEqual(record_info.ctype, "text/html")
        self.assertEqual(record_info.date, datetime.datetime(2023, 4, 20, 12, 0, 0))

    def test_resource_record(self):
        record_mock = MagicMock(spec=WarcRecord)
        record_mock.record_type = WarcRecordType.resource
        record_mock.headers = {
            "WARC-Payload-Digest": "sha1:payload_id",
            "WARC-Target-URI": "http://example.com",
            "Content-Type": "application/json",
            "WARC-Date": "2023-04-20T12:00:00Z",
        }
        record_mock.http_headers = {}

        record_info = WarcRecordInfo(record_mock)

        self.assertEqual(record_info.payload_id, "payload_id")
        self.assertEqual(record_info.target_uri, "http://example.com")
        self.assertEqual(record_info.ctype, "application/json")
        self.assertEqual(record_info.date, datetime.datetime(2023, 4, 20, 12, 0, 0))

    def test_unsupported_record_type(self):
        record_mock = MagicMock(spec=WarcRecord)
        record_mock.record_type = "unsupported"
        record_mock.headers = {"WARC-Payload-Digest": "sha1:payload_id", "WARC-Target-URI": "http://example.com"}
        record_mock.http_headers = {}

        with self.assertRaises(ValueError):
            WarcRecordInfo(record_mock)

    def test_missing_headers(self):
        record_mock = MagicMock(spec=WarcRecord)
        record_mock.record_type = WarcRecordType.response
        record_mock.headers = {}
        record_mock.http_headers = {}

        with patch("dolma.warc.record_info.datetime") as datetime_mock:
            now = datetime.datetime.now()
            datetime_mock.datetime.now.return_value = now
            record_info = WarcRecordInfo(record_mock)
            assert not record_info.is_valid

    def test_content_type_with_extra_info(self):
        record_mock = MagicMock(spec=WarcRecord)
        record_mock.record_type = WarcRecordType.response
        record_mock.headers = {"WARC-Payload-Digest": "sha1:payload_id", "WARC-Target-URI": "http://example.com"}
        record_mock.http_headers = {
            "Content-Type": "text/html; charset=utf-8; boundary=---123",
            "Date": "Thu, 20 Apr 2023 12:00:00 GMT",
        }

        record_info = WarcRecordInfo(record_mock)

        self.assertEqual(record_info.ctype, "text/html")

    def test_invalid_date_format(self):
        record_mock = MagicMock(spec=WarcRecord)
        record_mock.record_type = WarcRecordType.response
        record_mock.headers = {"WARC-Payload-Digest": "sha1:payload_id", "WARC-Target-URI": "http://example.com"}
        record_mock.http_headers = {"Content-Type": "text/html; charset=utf-8", "Date": "Invalid Date"}

        with patch("dolma.warc.record_info.datetime") as datetime_mock:
            now = datetime.datetime.now()
            datetime_mock.datetime.now.return_value = now
            record_info = WarcRecordInfo(record_mock)

        # Assert that the date is "close enough" to now, since it's hard to mock perfectly
        self.assertAlmostEqual(record_info.date.timestamp(), now.timestamp(), delta=1)
