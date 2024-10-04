import random
import time
from functools import reduce
from io import BytesIO, TextIOWrapper
from typing import TYPE_CHECKING, Generator, List, Optional, Union

import smart_open
from necessary import necessary

from ..core.loggers import get_logger

with necessary("fastwarc", soft=True) as FASTWARC_AVAILABLE:
    if FASTWARC_AVAILABLE or TYPE_CHECKING:
        from fastwarc.stream_io import (  # pylint: disable=no-name-in-module
            GZipStream,
            LZ4Stream,
        )
        from fastwarc.warc import (  # pylint: disable=no-name-in-module
            ArchiveIterator,
            WarcRecord,
            WarcRecordType,
        )


class BackoffWarcIterator:
    def __init__(
        self,
        path: str,
        max_time: Optional[float] = None,
        min_wait: float = 1.0,
        max_tries: int = 10,
        max_wait: Optional[float] = None,
        record_types: Optional[List[Union[str, "WarcRecordType"]]] = None,
    ):
        self.path = path
        self.max_time = max_time
        self.max_tries = max_tries
        self.max_wait = max_wait or float("inf")
        self.min_wait = min_wait
        self.logger = get_logger(self.__class__.__name__)

        self._file_object: Optional[Union[TextIOWrapper, BytesIO]] = None
        self._start_time = float("-inf")
        self._attempt = 0
        self._location = 0

        record_types = record_types or ["response", "warcinfo"]
        self.record_types = [WarcRecordType[r] if isinstance(r, str) else r for r in record_types]

    def _open(self):
        self._attempt += 1

        # close any previous file object
        self.close()

        if self.path.endswith(".lz4"):
            warc_stream = smart_open.open(self.path, "rb", compression="disable")
            warc_stream.seek(self._location)
            self._file_object = LZ4Stream(warc_stream)
        elif self.path.endswith(".gz"):
            warc_stream = smart_open.open(self.path, "rb", compression="disable")
            warc_stream.seek(self._location)
            self._file_object = GZipStream(warc_stream)
        else:
            obj = smart_open.open(self.path, "rt")
            obj.seek(self._location)
            self._file_object = obj

    def close(self):
        if self._file_object is not None:
            self._file_object.close()

    def wait(self):
        # exponential backoff with jitter
        # https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
        wait_time = random.uniform(0, min(2**self._attempt * self.min_wait, self.max_wait))

        # inform user, then sleep
        self.logger.warning("Failed to read %s, retrying in %.2f seconds.", self.path, wait_time)
        time.sleep(wait_time)

    def should_raise(self):
        if self._attempt >= self.max_tries:
            self.logger.error("Failed to read %s after %d tries.", self.path, self._attempt)
            return True

        if self.max_time and (self._start_time + time.time()) > self.max_time:
            self.logger.error("Failed to read %s after %.2f seconds.", self.path, self.max_time)
            return True

    def __enter__(self):
        self._open()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self._file_object = None
        self._location = 0
        self._start_time = float("-inf")
        self._attempt = 0

    def __iter__(self) -> Generator["WarcRecord", None, None]:
        if self._file_object is None:
            raise OSError("File object must be opened before iterating.")

        while True:
            try:
                it = ArchiveIterator(self._file_object, record_types=reduce(lambda a, b: a | b, self.record_types))
                for record in it:
                    self._location = self._file_object.tell()
                    yield record
                return
            except Exception as exp:
                if self.should_raise():
                    self.logger.exception("Failed to read %s.", self.path)
                    raise exp

                self.wait()
                self._open()


class SimpleWarcIterator:
    def __init__(self, path: str, record_types: Optional[List[Union[str, "WarcRecordType"]]] = None):
        self.path = path
        self.record_types = [
            WarcRecordType[r] if isinstance(r, str) else r for r in (record_types or ["response", "warcinfo"])
        ]
        self._fobj: Optional[Union[BytesIO, TextIOWrapper]] = None
        self._it: Optional["ArchiveIterator"] = None

    def __enter__(self):
        if self.path.endswith(".lz4"):
            warc_stream = smart_open.open(self.path, "rb", compression="disable")
            self._fobj = LZ4Stream(warc_stream)
        elif self.path.endswith(".gz"):
            warc_stream = smart_open.open(self.path, "rb", compression="disable")
            self._fobj = GZipStream(warc_stream)
        else:
            self._fobj = smart_open.open(self.path, "rt")
        self._it = ArchiveIterator(self._fobj, record_types=reduce(lambda a, b: a | b, self.record_types))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fobj is not None:
            self._fobj.close()
        self._fobj = None
        self._it = None

    def __iter__(self) -> Generator["WarcRecord", None, None]:
        if self._it is None:
            raise OSError("File object must be opened before iterating.")
        yield from self._it
