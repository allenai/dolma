import datetime
from itertools import chain
import json
from tempfile import mkdtemp
from typing import Any, Dict
from multiprocessing import cpu_count
from bs4 import BeautifulSoup
from dolma.core.parallel import AllPathsTuple, BaseParallelProcessor, QueueType
from dolma.core.paths import glob_path, join_path, split_path
import smart_open
import feedparser
import trafilatura
import uniseg.wordbreak


SRC = "s3://ai2-llm/pretraining-data/sources/smallweb/raw/feeds/*"
DST = "s3://ai2-llm/pretraining-data/sources/smallweb/v0/documents/feeds"
GROUP = 16


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class SmallWebProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(cls, queue: "QueueType", /, words: int = 0, files: int = 0, docs: int = 0) -> Dict[str, int]:
        return super().increment_progressbar(queue, words=words, files=files, docs=docs)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        all_files = [f for p in kwargs.get("src", []) for f in glob_path(f"{p}/*")]

        with smart_open.open(destination_path + ".jsonl.gz", "w") as dest_file:
            for fn in all_files:
                with smart_open.open(fn, "r") as input_file:
                    data = input_file.read()

                try:
                    content = feedparser.parse(data)
                except Exception:
                    content = {"entries": []}

                for entry in content.get("entries", []):
                    title = entry.get('title', '')
                    url = entry.get("link", None)
                    if url is None:
                        continue

                    published_parsed = entry.get('published_parsed', entry.get('updated_parsed', None))
                    created = datetime.datetime(
                        year=published_parsed.tm_year,
                        month=published_parsed.tm_mon,
                        day=published_parsed.tm_mday,
                        hour=published_parsed.tm_hour,
                        minute=published_parsed.tm_min,
                        second=published_parsed.tm_sec,
                    ) if published_parsed else datetime.datetime.now()

                    document = {
                        "metadata": {k: v for k, v in entry.items() if k != 'content'},
                        'created': convert_timestamp(created),
                        'added': convert_timestamp(datetime.datetime.now()),
                        'id': url,
                        "source": cls.__name__,
                    }

                    paragraphs = []
                    for content in entry.get("content", []):
                        if 'html' in content.type:
                            try:
                                text = trafilatura.extract(content.value, favor_precision=True, deduplicate=True)
                            except Exception:
                                text = None
                            if not text:
                                text = BeautifulSoup(content.value).text.strip()
                        else:
                            text = content.value.strip()
                        if text and len(text) > 25:
                            paragraphs.append(text.strip())

                    if len(paragraphs) == 0:
                        if (summary := entry.get("summary", None)) is not None:
                            try:
                                text = trafilatura.extract(summary, favor_precision=True, deduplicate=True)
                            except Exception:
                                text = None
                            if not text:
                                text = BeautifulSoup(summary).text
                            if len(text) > 25:
                                paragraphs.append(text.strip())

                    text = (f"{title.strip()}\n" if title.strip() else "") + "\n".join(paragraphs)

                    words_cnt = sum(1 for _ in uniseg.wordbreak.words(text))

                    if (text := text.strip()):
                        document['text'] = text
                        dest_file.write(json.dumps(document) + "\n")
                        cls.increment_progressbar(queue, docs=1, words=words_cnt)

                cls.increment_progressbar(queue, files=1)

    def _get_all_paths(self) -> AllPathsTuple:
        paths = super()._get_all_paths()

        new_tup = AllPathsTuple.empty()

        for i in range(0, len(paths.src), GROUP):
            new_tup.src.append(paths.src[i])
            new_tup.dst.append(paths.dst[i])
            new_tup.meta.append(paths.meta[i])
            new_tup.kwargs.append({**paths.kwargs[i], 'src': paths.src[i:i+GROUP]})
        return new_tup

    @classmethod
    def new(cls) -> "SmallWebProcessor":
        prot, parts = split_path(DST)
        paths = list(glob_path(SRC))
        get_fn = lambda x: split_path(x)[-1][-1]
        dest = [join_path(prot, *parts, get_fn(x)) for x in paths]

        return cls(
            source_prefix=paths,
            destination_prefix=dest,
            metadata_prefix=mkdtemp(),
            num_processes=cpu_count(),
            debug=False
        )


if __name__ == "__main__":
    SmallWebProcessor.new()()
