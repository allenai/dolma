import argparse
import multiprocessing
import os
import re
import tempfile

from ..core.paths import glob_path
from ..core.runtime import _make_paths_from_prefix
from . import WarcProcessor


# CommonCrawl format
# s3://commoncrawl/crawl-data/CC-MAIN-2023-40/segments/*/warc/*.warc.gz

# Example
# AWS_PROFILE=llm python -m dolma.warc --src 's3://ai2-russella/crawl-data/CC-MAIN-2019-18/segments/*/warc/*.warc.gz' --dst s3://ai2-llm/pretraining-data/sources/licensed-cc/v0/documents/CC-MAIN-2019-18


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    return ap.parse_args()


def main():
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    opts = get_args()

    with tempfile.TemporaryDirectory() as tempdir:
        src_paths = list(glob_path(opts.src))
        dst_paths = _make_paths_from_prefix(paths=src_paths, prefix=opts.dst)
        meta_paths = _make_paths_from_prefix(paths=src_paths, prefix=tempdir)

        processor = WarcProcessor(
            source_prefix=src_paths,
            destination_prefix=dst_paths,
            metadata_prefix=meta_paths,
            debug=False,
            num_processes=multiprocessing.cpu_count(),
        )
        processor(skip_unknown_license=True)


if __name__ == "__main__":
    main()
