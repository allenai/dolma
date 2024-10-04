from collections import Counter
from hashlib import md5
import itertools
import multiprocessing
from tempfile import TemporaryDirectory
from dolma.core.parallel import BaseParallelProcessor, BaseProgressBar, QueueType
from dolma.core.data_types import InputSpecWithMetadataAndAttributes
from dolma.core.paths import glob_path
from dolma.core.utils import add_compression
from typing import Counter as CounterType
import msgspec
import json
import smart_open


class LicensePbar(BaseProgressBar):
    documents: int = 0
    files: int = 0
    nc: int = 0
    nd: int = 0
    yc: int = 0


class LicenseCounter(BaseParallelProcessor):
    PROGRESS_BAR_CLS = LicensePbar

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs):
        add_compression()

        counter: "CounterType[str]" = Counter()
        decoder = msgspec.json.Decoder(InputSpecWithMetadataAndAttributes)

        with smart_open.open(source_path, "rb") as f_in, LicensePbar(queue) as pbar:
            for line in f_in:
                data = decoder.decode(line)
                pbar.documents += 1

                if not data.attributes:
                    continue

                licenses = {t.rsplit("__", 1)[-1] for t in data.attributes.keys() if t.startswith("cc_re")}
                if any("nc" in ln for ln in licenses):
                    pbar.nc += 1
                elif any("nd" in ln for ln in licenses):
                    pbar.nd += 1
                else:
                    pbar.yc += 1

                counter.update(licenses)

            pbar.files += 1

        destination_dir, _ = destination_path.rsplit("/", 1)
        destination_path = f"{destination_dir}/{md5(source_path.encode()).hexdigest()}.json"

        with smart_open.open(destination_path, "wt") as f_out:
            f_out.write(json.dumps(counter, indent=2))


def main():
    base_path = "s3://ai2-llm/pretraining-data/sources/cccc/v1/documents"
    base_dst = "s3://ai2-llm/stats/cccc/v1"

    # glob_params = dict(autoglob_dirs=False, recursive_dirs=False, yield_dirs=False)
    # it = itertools.chain(
    #     glob_path(f"{base_path}/CC-MAIN-*/*.jsonl.zst", **glob_params),
    #     glob_path(f"{base_path}/CC-MAIN-*/*/warc/*.jsonl.zst", **glob_params)
    # )
    # it = itertools.chain(
    #     glob_path(f"{base_path}/CC-MAIN-2021-17/*.jsonl.zst", **glob_params),
    # )

    with TemporaryDirectory() as tmpdir:
        src_paths, dst_paths, meta_paths = [], [], []
        for path in glob_path(base_path + '/*/*.gz'):
            snapshot = path.replace(base_path, "").lstrip("/").split("/")[0]
            src_paths.append(path)
            dst_paths.append(f"{base_dst}/{snapshot}")
            meta_paths.append(f"{tmpdir}/{snapshot}")

        print(f"Found {len(src_paths):,} files to process")

        counter = LicenseCounter(
            source_prefix=src_paths,
            destination_prefix=dst_paths,
            metadata_prefix=meta_paths,
            num_processes=multiprocessing.cpu_count() - 1,
            debug=False,
            skip_source_glob=True,
            ignore_existing=True,
        )
        counter()

    collated = Counter()
    for path in glob_path(f"{base_dst}/*/*.json"):
        with smart_open.open(path, "rt") as f:
            collated.update(json.load(f))

    sorted_collated = dict(sorted(collated.items(), key=lambda x: x[1], reverse=True))

    with smart_open.open(f"{base_dst}/collated.json", "wt") as f:
        f.write(json.dumps(sorted_collated, indent=2))

    print(json.dumps(sorted_collated, indent=2))


if __name__ == "__main__":
    main()
