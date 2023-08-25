import hashlib
from dataclasses import dataclass, field
from queue import Queue
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, Union

import msgspec
from omegaconf import MISSING
from omegaconf.omegaconf import OmegaConf as om
import smart_open

from dolma.core.parallel import BaseParallelProcessor
from dolma.core.paths import glob_path, make_relative


class TrafilaturaReformatter(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore[override]
        cls,
        queue: Queue[Union[Tuple[int, ...], None]],
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue[Union[Tuple[int, ...], None]],
        **kwargs: Any
    ):
        documents = 0
        interval = 10_000

        root, (_, rel_dst) = make_relative([source_path, destination_path])
        rel_dst = rel_dst.replace('/', '_') + '.gz'
        destination_path = f'{root}/{rel_dst}'

        with smart_open.open(source_path, "rb") as source_file, \
                smart_open.open(destination_path, "wb") as destination_file:
            for line in source_file:
                document = msgspec.json.decode(line)
                documents += 1

                transformed = {
                    # use hash of the whole document as the id
                    "id": hashlib.md5(line).hexdigest(),
                    "text": document["content"],
                    "metadata": document
                }

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

                destination_file.write(msgspec.json.encode(transformed) + b"\n")

        cls.increment_progressbar(queue, files=1, documents=documents % interval)


@dataclass
class Config:
    src: str = field(default_factory=lambda: MISSING)
    dst: str = field(default_factory=lambda: MISSING)
    proc: int = field(default=1)
    debug: bool = field(default=False)


def reformat_files(config):
    with TemporaryDirectory() as tempdir:
        processor = TrafilaturaReformatter(
            source_prefix=config.src,
            destination_prefix=config.dst,
            metadata_prefix=tempdir,
            num_processes=config.proc,
            debug=config.debug,
        )
        processor()


if __name__ == "__main__":
    config = om.merge(om.structured(Config), om.from_cli(sys.argv[1:]))
    reformat_files(config)
