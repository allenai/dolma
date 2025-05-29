import re
from typing import Tuple, Generator, Optional
from botocore.exceptions import ClientError
import boto3

s3_client = boto3.client('s3')

def validate_s3_path(s3_path: str) -> Tuple[bool, Optional[str]]:
    """Validate an S3 path."""
    pattern = r'^s3://[\w.-]+/.*$'
    if not re.match(pattern, s3_path):
        return False, f"Invalid S3 path format: {s3_path}"
    return True, None

def check_s3_path_exists(s3_path: str) -> Tuple[bool, Optional[str]]:
    """Check if an S3 path exists and is accessible."""
    try:
        bucket, key = s3_path[5:].split('/', 1)
        if key.endswith('/'):
            # For directories, we just need to check if the prefix exists
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
            if 'Contents' not in response:
                return False, f"S3 path does not exist or is empty: {s3_path}"
        else:
            # For files, we can use head_object
            s3_client.head_object(Bucket=bucket, Key=key)
        return True, None
    except ClientError as e:
        return False, f"S3 path does not exist or is not accessible: {s3_path}. Error: {str(e)}"

def check_s3_path_writable(s3_path: str) -> Tuple[bool, Optional[str]]:
    """Check if an S3 path is writable."""
    try:
        bucket, key = s3_path[5:].split('/', 1)
        # Ensure the key ends with a '/' to treat it as a directory
        if not key.endswith('/'):
            key += '/'
        s3_client.put_object(Bucket=bucket, Key=f"{key}test_write", Body=b'')
        s3_client.delete_object(Bucket=bucket, Key=f"{key}test_write")
        return True, None
    except ClientError as e:
        return False, f"S3 path is not writable: {s3_path}. Error: {str(e)}"

def check_s3_parent_exists(s3_path: str) -> Tuple[bool, Optional[str]]:
    """Check if the parent directory of an S3 path exists."""
    parent_path = '/'.join(s3_path.split('/')[:-1]) + '/'
    return check_s3_path_exists(parent_path)

def list_s3_objects(s3_path: str) -> Generator[str, None, None]:
    """List objects in an S3 path, handling wildcards."""

    bucket, prefix = s3_path[5:].split('/', 1)

    # Remove '**/' from the prefix
    prefix = prefix.replace('**/', '')

    # Remove the filename pattern (e.g., '*.jsonl.gz') from the prefix
    prefix = '/'.join(prefix.split('/')[:-1]) + '/'

    paginator = s3_client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                yield f"s3://{bucket}/{obj['Key']}"

def get_base_path(s3_path: str) -> str:
    """Extract the base path from an S3 path with wildcards."""
    parts = s3_path.split('/')
    base_parts = []
    for part in parts:
        if part == '**':
            break
        base_parts.append(part)
    return '/'.join(base_parts)

def get_corresponding_attribute_path(doc_path: str, base_doc_path: str, base_attr_path: str, attr_type: str) -> str:
    """Get the corresponding attribute path for a given document path and attribute type."""
    relative_path = doc_path.replace(base_doc_path, '', 1)
    relative_path = relative_path.lstrip('/')
    return f"{base_attr_path.rstrip('/')}/{attr_type}/{relative_path}"
