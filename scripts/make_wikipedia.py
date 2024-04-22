"""
Minimal example to download a Wikipedia dump, process via WikiExtractor, and transform to Dolma format.

Author: Luca Soldaini (@soldni)
"""


import argparse
import datetime
import json
import logging
import multiprocessing
import re
import string
import sys
from contextlib import ExitStack
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Union

import requests
import smart_open
from dolma.core.parallel import BaseParallelProcessor, QueueType
from uniseg.wordbreak import words as uniseg_get_words

CMD_INSTALL = "pip install git+https://github.com/santhoshtr/wikiextractor.git requests smart_open tqdm"

try:
    from wikiextractor import WikiExtractor
except ImportError:
    print(f"Please install wikiextractor with `{CMD_INSTALL}`")
    sys.exit(1)

try:
    import requests  # noqa
except ImportError:
    print(f"Please install requests with `{CMD_INSTALL}`")
    sys.exit(1)

try:
    import smart_open  # noqa
except ImportError:
    print(f"Please install smart_open with `{CMD_INSTALL}`")
    sys.exit(1)

try:
    import tqdm  # noqa
except ImportError:
    print(f"Please install tqdm with `{CMD_INSTALL}`")
    sys.exit(1)


DUMP_URL = "https://dumps.wikimedia.org/{lang}wiki/{date}/{lang}wiki-{date}-pages-articles-multistream.xml.bz2"
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


def get_words(text: str) -> List[str]:
    return [word for word in uniseg_get_words(text) if not all(char in string.whitespace for char in word)]


def download_file(url, filename):
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise ValueError(f"File {url} not found; try a more recent date")

    file_size = int(response.headers["Content-Length"])

    with open(filename, "wb") as f, tqdm.tqdm(
        desc="Downloading", total=file_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(len(chunk))


def download_wiki(date: str, lang: str, output_path: Union[str, Path], overwrite: bool = False):
    assert re.match(r"\d{8}", date), "Date must be in YYYYMMDD format"
    dump_url = DUMP_URL.format(date=date, lang=lang)

    output_path = Path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        LOGGER.warning(f"Output file {output_path} exists. Use --overwrite to overwrite it.")
        return

    if not output_path.suffix.endswith(".bz2"):
        LOGGER.warning("Output file does not end with .bz2. This is not recommended.")

    try:
        LOGGER.info(f"Downloading {dump_url} to {output_path}")
        download_file(dump_url, output_path)
    except Exception as exception:
        if output_path.exists():
            output_path.unlink()
        raise exception


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Date in YYYYMMDD format", default="20231001")
    ap.add_argument("--lang", help="Language code", default="simple")
    ap.add_argument("--output", help="Output dir", required=True)
    ap.add_argument("--overwrite", help="Overwrite existing file", action="store_true")
    ap.add_argument("--processes", help="Number of processes", type=int, default=1)
    return ap.parse_args()


class NewOutputSplitter:
    file: TextIOWrapper

    def __init__(self, nextFile, max_file_size=0, compress=True):
        """
        :param nextFile: a NextFile object from which to obtain filenames
            to use.
        :param max_file_size: the maximum size of each file.
        :para compress: whether to write data with bzip compression.
        """
        self.nextFile = nextFile
        self.compress = compress
        self.max_file_size = max_file_size
        self.stack = ExitStack()
        self.ext = ".gz" if compress else ""
        self.written = 0
        self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reserve(self, size):
        if self.written + size > self.max_file_size:
            self.close()
            self.open()
            self.written = 0

    def write(self, data):
        self.reserve(len(data))
        self.file.write(data)

    def close(self):
        self.stack.pop_all().close()

    def open(self):
        self.file = self.stack.enter_context(smart_open.open(self.nextFile.next() + self.ext, "wt"))


def wiki_extract(output_gzip: Path, output_json: Path, processes: int = 1, overwrite: bool = False):
    if output_json.exists() and not overwrite:
        LOGGER.warning(f"Output file {output_json} exists. Use --overwrite to overwrite it.")
        return

    WikiExtractor.Extractor.keepLinks = False
    WikiExtractor.Extractor.HtmlFormatting = False
    WikiExtractor.ignoreTag("a")
    WikiExtractor.Extractor.to_json = True  # type: ignore
    WikiExtractor.acceptedNamespaces = ["Article"]
    # input_file = output_gzip

    # # output_path = output_json
    # def _open(*args, **kwargs):
    #     return tqdm.tqdm(smart_open.open(*args, **kwargs))

    WikiExtractor.decode_open = smart_open.open
    WikiExtractor.OutputSplitter = NewOutputSplitter

    file_size = 1024 * 1024 * 1024  # 1 GB

    output_json.mkdir(exist_ok=True, parents=True)

    WikiExtractor.process_dump(
        input_file=str(output_gzip),
        template_file=None,
        out_file=str(output_json),
        file_size=file_size,
        file_compress=True,
        process_count=processes,
        html_safe=False,
    )


class WikiExtractorParallel(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore[override]
        cls, queue, /, files: int = 0, documents: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(
            queue,
            files=files,
            documents=documents,
        )

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        date = kwargs.pop("date", None)
        lang = kwargs.pop("lang", None)
        assert date is not None, "date should be provided"
        assert lang is not None, "lang should be provided"

        created_time = convert_timestamp(datetime.datetime.strptime(date, "%Y%m%d"))
        current_time = convert_timestamp(datetime.datetime.now())
        logger = cls.get_logger()

        documents_count = 0
        update_interval = 1

        with smart_open.open(source_path) as f, smart_open.open(destination_path, "w") as g:
            try:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exp:
                        logger.warning("Failed to parse %s:%s `%s...`: %s", source_path, i, line[:80], exp)
                        continue

                    id_ = data.pop("id")
                    title = data.pop("title", "").strip()
                    body = data.pop("text", "").strip()

                    if not id_ or not title or not body:
                        # logger.warning("Skipping %s:%s: missing id, title or body", source_path, i)
                        continue

                    text = f"{title}\n\n{body}".strip()

                    json_data = {
                        "id": id_,
                        "source": "wikipedia",
                        "version": "v0",
                        "text": text,
                        "created": created_time,
                        "added": current_time,
                        "metadata": {**data, "length": len(get_words(text))},
                    }

                    g.write(json.dumps(json_data) + "\n")
                    documents_count += 1

                    if documents_count % update_interval == 0:
                        # update the progress bar every 1000 documents to prevent
                        # buffering
                        cls.increment_progressbar(queue, documents=documents_count)

                        if queue.qsize() >= multiprocessing.cpu_count():
                            # double the update interval if the queue is full
                            update_interval *= 2

                        documents_count = 0

            except Exception as exp:
                logger.warning("Failed to process %s: %s", source_path, exp)
                return

            cls.increment_progressbar(queue, files=1, documents=documents_count)


def main():
    args = get_arguments()
    output_gzip = Path(args.output) / f"wiki_{args.date}_{args.lang}.xml.bz2"
    output_json = Path(args.output) / f"wiki_{args.date}_{args.lang}"
    output_final = Path(args.output) / "v0/documents"

    download_wiki(date=args.date, lang=args.lang, output_path=output_gzip, overwrite=args.overwrite)
    wiki_extract(output_gzip=output_gzip, output_json=output_json, processes=args.processes)

    with TemporaryDirectory() as tempdir:
        processor = WikiExtractorParallel(
            source_prefix=f"{output_json}/*/*.gz",
            destination_prefix=str(output_final),
            metadata_prefix=tempdir,
            num_processes=args.processes,
        )
        processor(date=args.date, lang=args.lang)


if __name__ == "__main__":
    # setting multiprocessing start method to spawn
    multiprocessing.set_start_method("spawn")

    main()
