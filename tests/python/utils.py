import json
import logging
import os
import re
import uuid
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple, Union
from unittest import TestCase
from urllib.parse import urlparse

import boto3
import smart_open
from smart_open import open

from dolma.core.paths import glob_path, mkdir_p

DOLMA_TESTS_S3_PREFIX_ENV_VAR = "DOLMA_TESTS_S3_PREFIX"
DOLMA_TESTS_SKIP_AWS_ENV_VAR = "DOLMA_TESTS_SKIP_AWS"
DOLMA_TESTS_S3_PREFIX_DEFAULT = "s3://dolma-tests"

LOGGER = logging.getLogger(__name__)


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse an s3 path into a bucket and key.

    Args:
        s3_path: The s3 path to parse.

    Returns:
        A tuple containing the bucket and key.
    """
    if not re.match(r"^s3://[\w-]+", s3_path):
        raise RuntimeError(f"Invalid s3 path: {s3_path}")

    # use urlparse to parse the s3 path
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip("/")


def get_test_prefix() -> str:
    # get the test prefix from the environment, or use the default if not set
    test_prefix = os.environ.get(DOLMA_TESTS_S3_PREFIX_ENV_VAR, DOLMA_TESTS_S3_PREFIX_DEFAULT)

    # this will check if it is a valid path
    bucket, _ = parse_s3_path(test_prefix)

    # check if the user has access to the test bucket using boto3
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        if not skip_aws_tests():
            raise RuntimeError(
                f"Unable to access test bucket '{test_prefix}'. To provide a different bucket, "
                f"set the '{DOLMA_TESTS_S3_PREFIX_ENV_VAR}' environment variable before running the tests."
            )

    # add a uuid to the test prefix to avoid collisions
    return f"{test_prefix.rstrip()}/{uuid.uuid4()}"


def skip_aws_tests() -> bool:
    dolma_tests_skip = os.environ.get(DOLMA_TESTS_SKIP_AWS_ENV_VAR)
    LOGGER.info(f"{DOLMA_TESTS_SKIP_AWS_ENV_VAR}: {dolma_tests_skip}")
    return (dolma_tests_skip or "false").lower() == "true"


def upload_test_documents(local_input: str, test_prefix: str) -> Tuple[str, str]:
    remote_input = f"{test_prefix}/input/documents"
    remote_output = f"{test_prefix}/output/documents"

    for i, local_fp in enumerate(glob_path(local_input)):
        remote_fp = f"{remote_input}/{i:05d}.json.gz"

        with open(local_fp, "rb") as f, open(remote_fp, "wb") as g:
            g.write(f.read())

    return remote_input, remote_output


def upload_test_attributes(local_attributes: str, test_prefix: str):
    remote_attributes = f"{test_prefix}/input/attributes"

    for i, local_fp in enumerate(glob_path(local_attributes)):
        matched = re.match(r"^(attributes|duplicate)-(\w+)", local_fp)
        if not matched:
            raise RuntimeError(f"Unexpected filename: {local_fp}")

        _, name = matched.groups()

        remote_fp = f"{remote_attributes}/{name}/{i:05d}.json.gz"

        with open(local_fp, "rb") as f, open(remote_fp, "wb") as g:
            g.write(f.read())


def clean_test_data(test_prefix: str):
    s3 = boto3.client("s3")

    bucket_name, prefix = parse_s3_path(test_prefix)

    for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", []):
        s3.delete_object(Bucket=bucket_name, Key=obj["Key"])


def download_s3_prefix(s3_prefix: str, local_prefix: str):
    s3 = boto3.client("s3")

    bucket_name, prefix = parse_s3_path(s3_prefix)

    for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", []):
        name = obj["Key"].replace(prefix, "").lstrip("/")
        local_fp = os.path.join(local_prefix, name)
        mkdir_p(os.path.dirname(local_fp))

        s3.download_file(Bucket=bucket_name, Key=obj["Key"], Filename=local_fp)


def upload_s3_prefix(s3_prefix: str, local_prefix: str):
    s3 = boto3.client("s3")

    bucket_name, prefix = parse_s3_path(s3_prefix)

    for local_fp in glob_path(local_prefix):
        name = local_fp.replace(local_prefix, "").lstrip("/")
        s3.upload_file(Bucket=bucket_name, Key=f"{prefix}/{name}", Filename=local_fp)


def load_jsonl(fp: Union[str, Path]) -> List[dict]:
    with smart_open.open(fp, "r") as f:
        return [json.loads(ln) for ln in f]


class TestCasePipeline(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def readUnits(self, paths: List[Union[Path, str]]) -> List[dict]:
        units = chain.from_iterable(load_jsonl(fp) for fp in paths)
        return sorted(units, key=lambda x: int(x["id"]))

    def writeUnits(
        self, units: List[dict], unit_type: str, partitions: int = 1, ext_dir: Optional[Path] = None
    ) -> List[str]:
        if len(units) < partitions:
            raise ValueError(f"Cannot partition {len(units)} {unit_type} over {partitions} partitions")

        dir_path = ext_dir or Path(self.makeUniquePath())
        file_paths = []
        for i in range(partitions):
            fp = dir_path / f"{unit_type}/{i}.jsonl.gz"
            fp.parent.mkdir(parents=True, exist_ok=True)
            with smart_open.open(fp, "w") as f:
                for doc in units[i::partitions]:
                    f.write(json.dumps(doc) + "\n")
            file_paths.append(fp)

        return [str(p) for p in file_paths]

    def writeDocs(self, docs: List[str], partitions: int = 1, ext_dir: Optional[Path] = None) -> List[str]:
        encoded_docs = [{"id": str(i), "text": d, "source": __file__} for i, d in enumerate(docs)]
        return self.writeUnits(units=encoded_docs, unit_type="documents", partitions=partitions, ext_dir=ext_dir)

    def writeAttributes(
        self,
        attributes: List[List[Tuple[int, int, float]]],
        attribute_name: str,
        partitions: int = 1,
        ext_dir: Optional[Path] = None,
    ) -> List[str]:
        encoded_attributes = [{"id": str(i), "attributes": {attribute_name: d}} for i, d in enumerate(attributes)]
        return self.writeUnits(
            units=encoded_attributes,
            unit_type=f"attributes/{attribute_name}",
            partitions=partitions,
            ext_dir=ext_dir,
        )

    def makeUniquePath(self, ext_dir: Optional[Path] = None, ext: str = "") -> str:
        ext = f".{ext.lstrip('.')}" if ext else ext
        return f"{ext_dir or self.temp_dir.name}/{uuid.uuid4()}{ext}"

    def writeConfig(self, config: dict, ext_dir: Optional[Path] = None) -> str:
        fp = Path(self.makeUniquePath(ext="json", ext_dir=ext_dir))
        with smart_open.open(fp, "wt") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        return str(fp)

    def combineIntoDoc(self, *lines: str, join: str = "\n") -> str:
        return join.join(lines)
