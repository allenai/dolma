import datetime
import multiprocessing
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List, Optional, Tuple

import smart_open
from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.paths import glob_path, make_relative, mkdir_p
from msgspec import Struct, field
from msgspec.json import Decoder, Encoder


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class PassageSpec(Struct):
    text: List[str] = field(default_factory=list)


class EntrySpec(Struct):
    id: str
    passage: PassageSpec
    source_url: Optional[str] = None
    source_text: Optional[str] = None
    source_lang: Optional[str] = None


class MegaWikaSpec(Struct):
    article_title: str
    article_text: str
    entries: List[EntrySpec]


class MegaWikaRawProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls, queue: "QueueType", /, files: int = 0, documents: int = 0, entries: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents, entries=entries)

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        all_grouped_paths: List[List[str]] = kwargs.pop("all_grouped_paths")
        process_paths = all_grouped_paths[int(source_path)]

        parser = Decoder(MegaWikaSpec)
        writer = Encoder()
        created = convert_timestamp(datetime.datetime(year=2023, month=7, day=19))

        update_interval = 1
        docs_cnt = entries_cnt = 0

        with smart_open.open(f"{destination_path}", "wb") as wf:
            for path in process_paths:
                with smart_open.open(path, "rt") as rf:
                    for raw in rf:
                        doc = parser.decode(raw)

                        for entry in doc.entries:
                            wiki_lang, _ = entry.id.split("-", 1)
                            if entry.source_text is None or entry.source_lang is None or entry.source_url is None:
                                continue

                            output = {
                                "id": entry.id,
                                "created": created,
                                "added": convert_timestamp(datetime.datetime.now()),
                                "source": "megawika",
                                "text": entry.source_text,
                                "metadata": {
                                    "url": entry.source_url,
                                    "language": entry.source_lang,
                                    "wikipedia": {
                                        "title": doc.article_title,
                                        "text": doc.article_text,
                                        "passage": entry.passage.text,
                                        "language": wiki_lang,
                                    },
                                },
                            }
                            wf.write(writer.encode(output) + b"\n")     # pyright: ignore
                            entries_cnt += 1

                        docs_cnt += 1

                        if docs_cnt % update_interval == 0:
                            # update the progress bar every 1000 documents to prevent buffering
                            cls.increment_progressbar(queue, documents=docs_cnt, entries=entries_cnt)
                            docs_cnt = entries_cnt = 0

                            if queue.qsize() >= multiprocessing.cpu_count():
                                # double the update interval if the queue is full
                                update_interval *= 2

                cls.increment_progressbar(queue, files=1)
            cls.increment_progressbar(queue, documents=docs_cnt, entries=entries_cnt)

    def group_paths(self, base_src: str) -> List[List[str]]:
        dest: Dict[str, List[str]] = {}
        for path in glob_path(base_src):
            grouped_path, *_ = path.rsplit("/", 1)
            dest.setdefault(grouped_path, []).append(path)
        return list(dest.values())

    def make_paths(
        self,
        all_grouped_paths: List[List[str]],
        dst_prefix: str,
        meta_prefix: str,
        step: int = 100,
    ) -> Generator[Tuple[List[str], str, str], None, None]:
        root, _ = make_relative([p for paths in all_grouped_paths for p in paths])
        for group in all_grouped_paths:
            dest_root, dest_groups = make_relative(group)
            if dest_groups == ["."]:
                dest_root, _ = dest_root.rsplit("/", 1)

            group_dst_prefix = f"{dst_prefix}{dest_root.replace(root, '')}"
            group_meta_prefix = f"{meta_prefix}{dest_root.replace(root, '')}"
            mkdir_p(group_dst_prefix)
            mkdir_p(group_meta_prefix)

            last = (last_set := set(group[0].split("/")[-1].split("-")[0] for e in group)).pop()
            assert len(last_set) == 0, f"last_set: {last_set}"
            for i in range(0, len(group), step):
                end = min(i + step, len(group))
                out = (
                    group[i : i + step],
                    f"{group_dst_prefix}/{last}-{i:05d}-to-{end:05d}.jsonl.gz",
                    f"{group_meta_prefix}/{last}-{i:05d}-to-{end:05d}.jsonl.gz",
                )
                yield out

    def __call__(self, **kwargs):
        all_grouped_paths = self.group_paths(self.src_prefixes[0])
        all_grouped_paths, all_dst_paths, all_meta_paths = zip(
            *self.make_paths(all_grouped_paths, self.dst_prefixes[0], self.meta_prefixes[0])
        )
        all_source_paths = [str(i) for i in range(len(all_grouped_paths))]

        self.num_processes = min(len(all_source_paths), multiprocessing.cpu_count() - 1)
        fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=all_source_paths,
            all_destination_paths=all_dst_paths,
            all_metadata_paths=all_meta_paths,
            all_grouped_paths=all_grouped_paths,
        )


def main():
    with TemporaryDirectory() as tmp_dir:
        MegaWikaRawProcessor(
            source_prefix=f"{os.path.expanduser('~')}/megawika/data/*/*.jsonl",
            destination_prefix="s3://ai2-llm/pretraining-data/sources/megawika/v0/documents",
            metadata_prefix=tmp_dir,
            num_processes=multiprocessing.cpu_count(),
        )()


if __name__ == "__main__":
    main()
