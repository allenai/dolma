"""Exercise real writer close with remote transport mapped to local files."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import smart_open

from dolma.tokenizer.data_types import TokenizerOutput
from dolma.tokenizer.memmap_writer import MemmapWriter


def test_remote_upload_uses_bounded_reads(tmp_path):
    original_open = smart_open.open
    reads = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size=-1):
            reads.append(size)
            assert 0 < size <= 8 * 1024 * 1024
            return self.handle.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

    def transport(path, mode="r", **kwargs):
        if str(path).startswith("s3://"):
            path = tmp_path / Path(str(path)).name
        handle = original_open(path, mode=mode, **kwargs)
        return Reader(handle) if mode == "rb" else handle

    with patch("smart_open.open", side_effect=transport):
        with MemmapWriter("s3://test/shard", np.dtype("uint32"), max_tokens=10) as writer:
            assert writer.write(TokenizerOutput.from_tokens("doc", "input", 1, [1, 2, 3]))
    assert reads
    assert np.fromfile(tmp_path / "shard.npy", dtype=np.uint32).tolist() == [1, 2, 3]
    with original_open(tmp_path / "shard.csv.gz", "rt") as metadata:
        assert metadata.read().strip() == "0,3,doc,input,1"
