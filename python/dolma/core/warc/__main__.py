import os
import tempfile

from . import WarcProcessor

if __name__ == "__main__":
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
    temp = "s3://ai2-russella/crawl-data/CC-MAIN-2019-18/segments/1555578517558.8/warc/CC-MAIN-20190418101243-20190418122311-00016.warc.gz"

    with tempfile.TemporaryDirectory() as tempdir:
        processor = WarcProcessor(
            source_prefix=temp,
            destination_prefix="s3://ai2-llm/experimental/cc-main-2019-18/v0/documents",
            metadata_prefix=tempdir,
            debug=False,
            num_processes=8,
        )
        processor()
