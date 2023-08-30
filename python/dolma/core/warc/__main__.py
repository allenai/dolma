import multiprocessing
import os
import tempfile

from ..paths import glob_path
from ..runtime import _make_paths_from_prefix
from . import WarcProcessor

if __name__ == "__main__":
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    with tempfile.TemporaryDirectory() as tempdir:
        src = "s3://ai2-russella/crawl-data/CC-MAIN-2019-18/segments/*/warc/*.warc.gz"
        dst = "s3://ai2-llm/experimental/cc-main-2019-18/v0/documents"

        src_paths = list(glob_path(src))
        dst_paths = _make_paths_from_prefix(paths=src_paths, prefix=dst)
        meta_paths = _make_paths_from_prefix(paths=src_paths, prefix=tempdir)

        processor = WarcProcessor(
            source_prefix=src_paths,
            destination_prefix=dst_paths,
            metadata_prefix=meta_paths,
            debug=False,
            num_processes=multiprocessing.cpu_count(),
        )
        processor(skip_unknown_license=True)
