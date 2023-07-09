from pathlib import Path
from unittest import TestCase

from dolma.core.runtime import _make_paths_from_prefix, _make_paths_from_substitution

LOCAL_DATA = Path(__file__).parent.parent / "data"


class TestRuntimeUtilities(TestCase):
    def test_make_paths_from_substitution(self):
        paths = [
            "s3://bucket/common-crawl/documents/cc_*/*.json.gz",
            "/local/path/to/documents/train/*",
        ]
        new_paths = _make_paths_from_substitution(
            paths=paths,
            find="documents",
            replace="attributes",
        )
        self.assertEqual(new_paths, ["s3://bucket/common-crawl/attributes", "/local/path/to/attributes/train"])

    def test_make_paths_from_prefix(self):
        paths = [
            "s3://bucket/common-crawl/documents/cc_head/*.json.gz",
            "s3://bucket/common-crawl/documents/cc_middle/*.json.gz",
            "s3://bucket/common-crawl/documents/cc_tail/*.json.gz",
        ]
        new_paths = _make_paths_from_prefix(
            paths=paths,
            prefix="s3://bucket/common-crawl/attributes/",
        )
        self.assertEqual(
            new_paths,
            [
                "s3://bucket/common-crawl/attributes/cc_head",
                "s3://bucket/common-crawl/attributes/cc_middle",
                "s3://bucket/common-crawl/attributes/cc_tail",
            ],
        )

        paths = [
            "s3://bucket/common-crawl/documents/*.json.gz",
            "s3://bucket2/c4/documents/**/data/*.json.gz",
        ]
        new_paths = _make_paths_from_prefix(
            paths=paths,
            prefix="/local/path/",
        )
        self.assertEqual(
            new_paths,
            [
                "/local/path/bucket/common-crawl/documents",
                "/local/path/bucket2/c4/documents",
            ],
        )
