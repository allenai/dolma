import os
import random
import re
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm
import boto3
import json
import itertools
import smart_open
from botocore.exceptions import ClientError

from s3_utils import s3_client, list_s3_objects, get_base_path, get_corresponding_attribute_path
from utils import vprint

class FileDownloadError(Exception):
    pass

def sample_files(s3_path: str, num_samples: int) -> List[str]:
    """Sample a subset of files from an S3 path."""
    all_files = list(list_s3_objects(s3_path))
    # Filter out directories (paths ending with '/')
    all_files = [f for f in all_files if not f.endswith('/')]
    chosen_files = random.sample(all_files, min(int(num_samples), len(all_files)))
    print(f"Sampled {len(chosen_files)} files from {len(all_files)} matching files")
    return chosen_files

def download_file(s3_path: str, local_path: str) -> None:
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    try:
        s3_client.download_file(bucket, key, local_path)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise FileDownloadError(f"File not found: {s3_path}")
        else:
            raise FileDownloadError(f"Error downloading file {s3_path}: {str(e)}")


def sample_and_download_files(stream: Dict[str, Any], num_samples: int) -> Tuple[List[str], Dict[str, List[str]]]:
    temp_dir = "temp_sample_files"
    
    # Create the temporary directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    try:
        doc_samples = sample_files(stream['documents'][0], num_samples)
        
        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)
        
        total_files = len(doc_samples) * (len(stream['attributes']) + 1)  # +1 for the document itself
        
        with tqdm(total=total_files, desc="Downloading files") as pbar:
            local_doc_samples = []
            local_attr_samples_dict = {attr_type: [] for attr_type in stream['attributes']}
            
            for doc_sample in doc_samples:
                try:
                    local_doc_path = os.path.join(temp_dir, os.path.basename(doc_sample))
                    download_file(doc_sample, local_doc_path)
                    local_doc_samples.append(local_doc_path)
                    pbar.update(1)
                    
                    # Extract the base name and extension
                    base_name, extension = os.path.splitext(os.path.basename(doc_sample))
                    if extension == '.gz':
                        # Handle double extensions like .jsonl.gz
                        base_name, inner_extension = os.path.splitext(base_name)
                        extension = inner_extension + extension
                    
                    for attr_type in stream['attributes']:
                        attr_sample = get_corresponding_attribute_path(doc_sample, base_doc_path, base_attr_path, attr_type)
                        # Construct the new filename with the attribute type before the extension, using a hyphen
                        new_filename = f"{base_name}-{attr_type}{extension}"
                        local_attr_path = os.path.join(temp_dir, new_filename)
                        download_file(attr_sample, local_attr_path)
                        local_attr_samples_dict[attr_type].append(local_attr_path)
                        pbar.update(1)
                except FileDownloadError as e:
                    print(f"Warning: {str(e)}. Skipping this file and its attributes.")
                    continue
            
        return local_doc_samples, local_attr_samples_dict
    
    except Exception as e:
        print(f"An error occurred during file sampling and downloading: {str(e)}")
        raise
    
def count_file_lines(file_path: str) -> int:
    """
    Count the number of lines in a file (local or S3, compressed or not).
    
    :param file_path: Path to the file (can be S3 or local)
    :return: Number of lines in the file, or -1 if an error occurred
    """
    # print(f"Counting lines in file: {file_path}")
    try:
        with smart_open.open(file_path, 'rb') as f:
            # print("successfully opened file in count_file_lines")
            line_count = sum(1 for _ in f)
        return line_count
    except Exception as e:
        print(f"Error counting lines in file {file_path}: {str(e)}")
        return -1

def check_attribute_name_typos(config_attributes: set, sample_attributes: set) -> None:
    """Check for typos in attribute names by comparing config and sample data."""
    missing_in_sample = config_attributes - sample_attributes
    extra_in_sample = sample_attributes - config_attributes
    
    if missing_in_sample:
        print("Warning: The following attributes are in the config but not in the sample data:")
        for attr in missing_in_sample:
            print(f"  - {attr}")
    
    if extra_in_sample:
        print("Info: The following attributes are in the sample data but not used in the config:")
        for attr in extra_in_sample:
            print(f"  - {attr}")

def sample_file_lines(file_path: str, num_lines: int = 1) -> Optional[List[str]]:
    """
    Sample N lines from a file, handling both local and S3 paths, and compression.
    
    Args:
    file_path (str): Path to the file (local or S3)
    num_lines (int): Number of lines to sample (default: 1)
    
    Returns:
    list: List of sampled lines, or None if an error occurred
    """
    try:
        if not isinstance(file_path, str):
            raise ValueError(f"Expected string for file_path, got {type(file_path)}")
        with smart_open.open(file_path, 'r') as f:
            # Use itertools.islice to efficiently read N lines
            sampled_lines = list(itertools.islice(f, num_lines))
        
        if not sampled_lines:
            print(f"Warning: File is empty or could not be read: {file_path}")
            return None
        
        # Strip whitespace from each line
        sampled_lines = [line.strip() for line in sampled_lines]
        
        if len(sampled_lines) < num_lines:
            print(f"Warning: Requested {num_lines} lines, but file only contains {len(sampled_lines)} lines: {file_path}")
        
        return sampled_lines

    except ValueError as ve:
        print(f"Error in sample_file_lines: {str(ve)}")
        return None
    except Exception as e:
        print(f"Error in sample_file_lines when reading file {file_path}: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"File path type: {type(file_path)}")
        return None
    

def sample_documents_with_attributes(doc_file_paths: List[str], attr_file_paths: List[str], num_samples: int = 100) -> List[Dict[str, Any]]:    
    sampled_docs = []
    for doc_path, attr_paths in zip(doc_file_paths, attr_file_paths):
        doc_lines = sample_file_lines(doc_path, num_samples)
        if not doc_lines:
            continue
        
        attr_samples = {}
        for attr_path in attr_paths:
            attr_lines = sample_file_lines(attr_path, num_samples)
            if attr_lines:
                attr_name = os.path.basename(attr_path).split('.')[0]  # Extract attribute name from file name
                attr_samples[attr_name] = attr_lines

        for i, doc_line in enumerate(doc_lines):
            doc = json.loads(doc_line)
            for attr_name, attr_lines in attr_samples.items():
                if i < len(attr_lines):
                    doc[attr_name] = json.loads(attr_lines[i])
            sampled_docs.append(doc)

    return sampled_docs


def validate_jsonl(file_path: str, expected_fields: set) -> Tuple[bool, List[str]]:
    """
    Validate that the file is a valid JSONL and contains expected fields.
    
    :param file_path: Path to the file (can be S3 or local)
    :param expected_fields: Set of field names expected in each JSON object
    :return: Tuple (is_valid, error_messages)
    """
    unexpected_fields = set()
    error_messages = []
    is_valid = True

    try:
        with smart_open.open(file_path, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    missing_fields = expected_fields - set(data.keys())
                    new_fields = set(data.keys()) - expected_fields
                    
                    if missing_fields:
                        error_messages.append(f"Line {i}: Missing expected fields: {missing_fields}")
                        is_valid = False
                    
                    if new_fields:
                        unexpected_fields.update(new_fields)
                        is_valid = False
                    
                except json.JSONDecodeError:
                    error_messages.append(f"Line {i}: Invalid JSON")
                    is_valid = False
    
    except Exception as e:
        error_messages.append(f"Error reading file {file_path}: {str(e)}")
        is_valid = False
    
    if unexpected_fields:
        error_messages.append(f"Additional fields found across the file: {unexpected_fields}")
    return is_valid, error_messages

