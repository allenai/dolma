import hashlib
import json
import sys
from dataclasses import dataclass, field
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple, Union, cast

import smart_open
from omegaconf import MISSING
from omegaconf.omegaconf import OmegaConf as om

from dolma.core.parallel import BaseParallelProcessor


class TrafilaturaReformatter(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: "Queue[Union[Tuple[int, ...], None]]",
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        documents = 0
        interval = 10_000

        destination_prefix = kwargs.get("destination_prefix", None)

        new_destination_suffix = (
            destination_path.replace(destination_prefix, "").lstrip("/").replace("/", "_") + ".gz"
        )
        destination_path = f"{destination_prefix.rstrip()}/{new_destination_suffix}"

        with smart_open.open(source_path, "rt") as source_file, smart_open.open(
            destination_path, "wt"
        ) as destination_file:
            for line in source_file:
                document = json.loads(line)
                documents += 1

                transformed = {
                    # use hash of the whole document as the id
                    "id": hashlib.md5(line.encode("utf-8")).hexdigest(),
                    "text": document["content"],
                    "source": "trafilatura",
                    "metadata": document,
                }

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

                destination_file.write(json.dumps(transformed) + "\n")

        cls.increment_progressbar(queue, files=1, documents=documents % interval)


@dataclass
class Config:
    src: str = field(default_factory=lambda: MISSING)
    dst: str = field(default_factory=lambda: MISSING)
    proc: int = field(default=1)
    debug: bool = field(default=False)


def reformat_files(config: Config):
    with TemporaryDirectory() as tempdir:
        processor = TrafilaturaReformatter(
            source_prefix=config.src,
            destination_prefix=config.dst,
            metadata_prefix=tempdir,
            num_processes=config.proc,
            debug=config.debug,
        )
        processor(destination_prefix=config.dst)


if __name__ == "__main__":
    config = om.merge(om.structured(Config), om.from_cli(sys.argv[1:]))
    reformat_files(cast(Config, config))
