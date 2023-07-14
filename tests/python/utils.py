import os
import re
import uuid
import boto3

from typing import Tuple
from dolma.core.paths import glob_path, mkdir_p
from urllib.parse import urlparse

from smart_open import open


DOLMA_TEST_S3_PREFIX_ENV_VAR = "DOLMA_TESTS_S3_PREFIX"
DOLMA_TEST_S3_PREFIX_DEFAULT = "s3://dolma/tests"


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Parse an s3 path into a bucket and key.

    Args:
        s3_path: The s3 path to parse.

    Returns:
        A tuple containing the bucket and key.
    """
    if not re.match(r'^s3://[\w-]+', s3_path):
        raise RuntimeError(f"Invalid s3 path: {s3_path}")

    # use urlparse to parse the s3 path
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip('/')


def get_test_prefix() -> str:
    # get the test prefix from the environment, or use the default if not set
    test_prefix = os.environ.get(DOLMA_TEST_S3_PREFIX_ENV_VAR, DOLMA_TEST_S3_PREFIX_DEFAULT)

    # this will check if it is a valid path
    bucket, _ = parse_s3_path(test_prefix)

    # check if the user has access to the test bucket using boto3
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        raise RuntimeError(
            f"Unable to access test bucket '{test_prefix}'. To provide a different bucket, "
            f"set the '{DOLMA_TEST_S3_PREFIX_ENV_VAR}' environment variable before running the tests."
        )

    # add a uuid to the test prefix to avoid collisions
    return test_prefix.rstrip() + str(uuid.uuid4())


def upload_test_documents(local_input: str, test_prefix: str) -> Tuple[str, str]:

    remote_input = f'{test_prefix}/input/documents'
    remote_output = f'{test_prefix}/output/documents'

    for i, local_fp in enumerate(glob_path(local_input)):
        remote_fp = f'{remote_input}/{i:05d}.json.gz'

        with open(local_fp, 'rb') as f, open(remote_fp, 'wb') as g:
            g.write(f.read())

    return remote_input, remote_output


def upload_test_attributes(local_attributes: str, test_prefix: str):
    remote_attributes = f'{test_prefix}/input/attributes'

    for i, local_fp in enumerate(glob_path(local_attributes)):
        matched = re.match(r'^(attributes|duplicate)-(\w+)', local_fp)
        if not matched:
            raise RuntimeError(f'Unexpected filename: {local_fp}')

        _, name = matched.groups()

        remote_fp = f'{remote_attributes}/{name}/{i:05d}.json.gz'

        with open(local_fp, 'rb') as f, open(remote_fp, 'wb') as g:
            g.write(f.read())


def clean_test_data(test_prefix: str):
    s3 = boto3.client('s3')

    bucket_name, prefix = parse_s3_path(test_prefix)

    for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get('Contents', []):
        s3.delete_object(Bucket=bucket_name, Key=obj['Key'])


def copy_files(src: str, dst: str) -> None:
    if urlparse(src).scheme in ('file', ''):
        mkdir_p(os.path.dirname(dst.lstrip('file:/')))

    with open(src, 'rb') as f, open(dst, 'wb') as g:
        g.write(f.read())
