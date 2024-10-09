import yaml
import json
import sys
import jq
from jsonpath_ng.ext import parse as parse_jsonpath
import re
import boto3
import random
import tempfile
import os
import shutil
from botocore.exceptions import ClientError
import json
import gzip
import io
import itertools
from collections import defaultdict
import subprocess
import signal
import smart_open
from tqdm import tqdm

s3_client = boto3.client('s3')

def load_config(config_path):
    """Load the configuration file (YAML or JSON)."""
    try:
        with open(config_path, 'r') as file:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(file)
            elif config_path.endswith('.json'):
                return json.load(file)
            else:
                raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        sys.exit(1)

def keyboard_interrupt_handler(signal, frame):
    print("\n\nScript interrupted by user. Exiting ...")
    sys.exit(0)

# Register the custom handler
signal.signal(signal.SIGINT, keyboard_interrupt_handler)

def extract_attribute_names_from_filters(filters):
    attribute_names = set()
    for filter_expr in filters:
        # Extract attribute names from JSONPath expressions
        matches = re.findall(r'@\.([a-zA-Z0-9_]+)', filter_expr)
        attribute_names.update(matches)
    return attribute_names

def validate_config_structure(config):
    """Validate the basic structure of the configuration."""
    required_fields = ['streams', 'processes']
    errors = []

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    if 'streams' in config:
        if not isinstance(config['streams'], list):
            errors.append("'streams' should be a list")
        else:
            for i, stream in enumerate(config['streams']):
                stream_errors = validate_stream(stream, i)
                errors.extend(stream_errors)

    if 'processes' in config and not isinstance(config['processes'], int):
        errors.append("'processes' should be an integer")

    return errors

def validate_stream(stream, index):
    """Validate an individual stream configuration."""
    required_fields = ['name', 'documents', 'attributes', 'output']
    errors = []

    for field in required_fields:
        if field not in stream:
            errors.append(f"Stream {index}: Missing required field: {field}")

    if 'documents' in stream and not isinstance(stream['documents'], list):
        errors.append(f"Stream {index}: 'documents' should be a list")

    if 'attributes' in stream and not isinstance(stream['attributes'], list):
        errors.append(f"Stream {index}: 'attributes' should be a list")

    if 'output' in stream:
        output_errors = validate_output(stream['output'], index)
        errors.extend(output_errors)

    if 'filter' in stream:
        filter_errors = validate_filter_config(stream['filter'], index)
        errors.extend(filter_errors)

    return errors

def validate_output(output, stream_index):
    """Validate the output configuration of a stream."""
    required_fields = ['path', 'max_size_in_bytes']
    errors = []

    for field in required_fields:
        if field not in output:
            errors.append(f"Stream {stream_index} output: Missing required field: {field}")

    if 'max_size_in_bytes' in output and not isinstance(output['max_size_in_bytes'], int):
        errors.append(f"Stream {stream_index} output: 'max_size_in_bytes' should be an integer")

    return errors

def validate_filter_config(filter_config, stream_index):
    """Validate the filter configuration of a stream."""
    errors = []

    if 'include' in filter_config and not isinstance(filter_config['include'], list):
        errors.append(f"Stream {stream_index} filter: 'include' should be a list")

    if 'exclude' in filter_config and not isinstance(filter_config['exclude'], list):
        errors.append(f"Stream {stream_index} filter: 'exclude' should be a list")

    return errors

def validate_s3_path(s3_path):
    """Validate an S3 path."""
    pattern = r'^s3://[\w.-]+/.*$'
    if not re.match(pattern, s3_path):
        return False, f"Invalid S3 path format: {s3_path}"
    return True, None

def check_s3_path_exists(s3_path):
    """Check if an S3 path exists and is accessible."""
    # s3_client = boto3.client('s3')
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

def check_s3_path_writable(s3_path):
    """Check if an S3 path is writable."""
    # s3_client = boto3.client('s3')
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

def check_s3_parent_exists(s3_path):
    """Check if the parent directory of an S3 path exists."""
    parent_path = '/'.join(s3_path.split('/')[:-1]) + '/'
    return check_s3_path_exists(parent_path)

def list_s3_objects(s3_path):
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

def get_base_path(s3_path):
    """Extract the base path from an S3 path with wildcards."""
    parts = s3_path.split('/')
    base_parts = []
    for part in parts:
        if part == '**':
            break
        base_parts.append(part)
    return '/'.join(base_parts)

def validate_jq_expression(expr):
    """Validate a JQ expression."""
    try:
        jq.compile(expr)
        return True, None
    except ValueError as e:
        return False, str(e)
    
def validate_jsonpath_expression(expr):
    """Validate a JSONPath expression."""
    try:
        parse_jsonpath(expr)
        return True, None
    except Exception as e:
        return False, str(e)
    
def split_complex_jsonpath(expr):
    """Split a complex JSONPath expression into individual valid JSONPath expressions."""
    # Extract the base path and the filter condition
    match = re.match(r'(\$\.attributes\[?\??)\s*(.+?)\s*(\]?)$', expr)
    if not match:
        return [expr]  # Return the original expression if it doesn't match the expected pattern

    base_path, conditions, closing_bracket = match.groups()
    
    # Split the conditions
    split_conditions = re.split(r'\s*(?:&&|\|\|)\s*', conditions)
    
    # Reconstruct each condition into a valid JSONPath
    valid_jsonpaths = []
    for condition in split_conditions:
        # Remove any opening and closing parentheses from the condition
        condition = condition.strip('()')
        # Extract the comparison part if present
        comparison_match = re.search(r'([<>]=?)\s*([^)]+)$', condition)
        if comparison_match:
            comparison_op, comparison_value = comparison_match.groups()
            # Remove the comparison part from the condition
            condition = condition[:comparison_match.start()].rstrip()
            # Add back the comparison
            condition += f" {comparison_op} {comparison_value.rstrip(')')}"
        valid_jsonpaths.append(f"{base_path}{condition}{closing_bracket}")
    
    return valid_jsonpaths

def validate_filter_expressions(filter_config):
    """Validate filter expressions based on specified syntax."""
    errors = []
    warnings = []

    syntax = filter_config.get('syntax', 'jsonpath').lower()  # Default to JSONPath if not specified

    if syntax not in ['jq', 'jsonpath']:
        warnings.append(f"Unsupported syntax '{syntax}'. Defaulting to JSONPath for validation.")
        syntax = 'jsonpath'

    validate_function = validate_jq_expression if syntax == 'jq' else validate_jsonpath_expression

    for filter_type in ['include', 'exclude']:
        expressions = filter_config.get(filter_type, [])
        for expr in expressions:
            if syntax == 'jsonpath':
                conditions = split_complex_jsonpath(expr)
                for condition in conditions:
                    valid, error = validate_function(condition)
                    if not valid:
                        errors.append(f"Invalid {syntax.upper()} sub-expression in {filter_type}: {condition}. Error: {error}")
            else:
                is_valid, error = validate_function(expr)
                if not is_valid:
                    errors.append(f"Invalid {syntax.upper()} expression in {filter_type}: {expr}. Error: {error}")

    return errors, warnings

def sample_files(s3_path, num_samples):
    """Sample a subset of files from an S3 path."""
    all_files = list(list_s3_objects(s3_path))
    # Filter out directories (paths ending with '/')
    all_files = [f for f in all_files if not f.endswith('/')]
    chosen_files = random.sample(all_files, min(num_samples, len(all_files)))
    print(f"Sampled {len(chosen_files)} files from {len(all_files)} matching files")
    # print("Sampled files:")
    # print("\n".join(chosen_files))
    return chosen_files

def download_file(s3_path, local_path):
    # print("I'm in download_file")
    # s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3_client.download_file(bucket, key, local_path)
    print(f"Successfully downloaded: {s3_path} -> {local_path}")

def sample_and_download_files(stream, num_samples):

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
                local_doc_path = os.path.join(temp_dir, os.path.basename(doc_sample))
                download_file(doc_sample, local_doc_path)
                local_doc_samples.append(local_doc_path)
                pbar.update(1)
                
                for attr_type in stream['attributes']:
                    attr_sample = get_corresponding_attribute_path(doc_sample, base_doc_path, base_attr_path, attr_type)
                    local_attr_path = os.path.join(temp_dir, f"{os.path.basename(doc_sample)}.{attr_type}")
                    download_file(attr_sample, local_attr_path)
                    local_attr_samples_dict[attr_type].append(local_attr_path)
                    pbar.update(1)
        
        return local_doc_samples, local_attr_samples_dict
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

def get_corresponding_attribute_path(doc_path, base_doc_path, base_attr_path, attr_type):
    """Get the corresponding attribute path for a given document path and attribute type."""
    relative_path = doc_path.replace(base_doc_path, '', 1)
    relative_path = relative_path.lstrip('/')
    return f"{base_attr_path.rstrip('/')}/{attr_type}/{relative_path}"

def validate_jsonl(file_path, expected_fields):
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
        print(f"Validating file: {file_path} in validate_jsonl")
        with smart_open.open(file_path, 'r') as f:
            print(f"succesfully opened file: {file_path}")
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
    print(f"Validation result in validate_jsonl: {is_valid}")
    return is_valid, error_messages

# def count_file_lines(file_path):
#     """
#     Count the number of lines in a file (local or S3, compressed or not).
    
#     :param file_path: Path to the file (can be S3 or local)
#     :return: Number of lines in the file
#     """
#     print(f"Counting lines in file: {file_path}")
#     try:
#         if file_path.startswith('s3://'):
#             print("reading from s3")
#             content = read_s3_file(file_path)
#             f = io.StringIO(content)
#         else:
#             if file_path.endswith('.gz'):
#                 f = gzip.open(file_path, 'rt')
#             else:
#                 f = open(file_path, 'r')
#         # print("line count attempt")
#         line_count = sum(1 for _ in f)
        
#         if not file_path.startswith('s3://'):
#             f.close()
        
#         return line_count
    
#     except Exception as e:
#         print(f"Error counting lines in file {file_path}: {str(e)}")
#         return -1
    
def count_file_lines(file_path):
    """
    Count the number of lines in a file (local or S3, compressed or not).
    
    :param file_path: Path to the file (can be S3 or local)
    :return: Number of lines in the file, or -1 if an error occurred
    """
    # print(f"Counting lines in file: {file_path}")
    try:
        with smart_open.open(file_path, 'r') as f:
            print("successfully opened file in count_file_lines")
            line_count = sum(1 for _ in f)
        return line_count
    except Exception as e:
        print(f"Error counting lines in file {file_path}: {str(e)}")
        return -1
    
def get_line_count(filename, buf_size=1024 * 1024):
    """
    Efficiently count lines in a file using buffered reading.
    
    :param filename: Path to the file (can be S3 or local)
    :param buf_size: Size of the buffer for reading (default 1MB)
    :return: Number of lines in the file
    """
    try:
        with smart_open.open(filename, 'r') as f:
            lines = 0
            read_f = f.read  # For S3 files, we use smart_open's read method
            buf = read_f(buf_size)
            while buf:
                lines += buf.count(b'\n')
                buf = read_f(buf_size)
            return lines
    except Exception as e:
        print(f"Error counting lines in file {filename}: {str(e)}")
        return -1

def apply_jq_filter(data, filter_expr):
    """Apply a JQ filter to the data and return the result."""
    try:
        result = jq.compile(filter_expr).input(data).all()
        return result
    except Exception as e:
        print(f"Error applying JQ filter '{filter_expr}': {str(e)}")
        return None

def apply_jsonpath_filter(data, filter_expr):
    """Apply a JSONPath filter to the data and return the result."""
    try:
        jsonpath_expr = parse_jsonpath.parse(filter_expr)
        result = [match.value for match in jsonpath_expr.find(data)]
        return result
    except Exception as e:
        print(f"Error applying JSONPath filter '{filter_expr}': {str(e)}")
        return None

def check_attribute_name_typos(config_attributes, sample_attributes):
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

def validate_filters_and_check_typos(attr_file_paths, filter_config, stream_attributes):
    """Validate filters and check for attribute name typos across multiple attribute files."""
    print("Validating filters and checking typos across all attribute files...")
    
    # Extract filter attributes from config
    print("let's see the filter attributes")
    filter_attributes = extract_filter_attributes(filter_config)
    print(f"Extracted filter attributes from config: {filter_attributes}")
    
    # Sample and extract attributes from all files
    all_sampled_attributes = set()
    for attr_file_path in attr_file_paths:
        sampled_attributes = sample_and_extract_attributes(attr_file_path)
        all_sampled_attributes.update(sampled_attributes)
        print(f"Attributes found in {attr_file_path}: {sampled_attributes}")
    
    # Check if all mixer config filters are found
    missing_attributes = filter_attributes - all_sampled_attributes
    
    if not missing_attributes:
        print("All mixer config filters were found in the attribute files.")
    else:
        print("Warning: Some mixer config filters were not found in the attribute files.")
        print("Missing attributes:")
        for attr in missing_attributes:
            print(f"  - {attr}")
        
        print("\nAll attributes found in files:")
        for attr in sorted(all_sampled_attributes):
            print(f"  - {attr}")
        
        print("\nThis detailed list is provided to help identify potential typos or misconfigurations.")


def extract_filter_attributes(filter_config):
    """Extract attribute names from filter expressions."""
    filter_attributes = set()
    for filter_type in ['include', 'exclude']:
        for filter_expr in filter_config.get(filter_type, []):
            # Extract attribute names from JSONPath expressions
            matches = re.findall(r'@\.([a-zA-Z_][a-zA-Z0-9_]*)', filter_expr)
            filter_attributes.update(matches)
    return filter_attributes

def sample_and_extract_attributes(attr_file_path, num_samples=5):
    """Sample lines from the attribute file and extract unique attributes."""
    sampled_attributes = set()
    sampled_lines = sample_file_lines(attr_file_path, num_lines=num_samples)
    
    for line in sampled_lines:
        try:
            data = json.loads(line)
            sampled_attributes.update(data['attributes'].keys())
        except (json.JSONDecodeError, KeyError):
            print(f"Error: Invalid JSON or missing 'attributes' key in sampled line from {attr_file_path}")
    
    return sampled_attributes

def sample_file_lines(file_path, num_lines=1):
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
    

def execute_filter_commands(doc_samples, attr_samples_dict, filter_config, num_lines=100):
    """
    Execute filter commands on sampled lines from documents and their attributes.
    
    Args:
    doc_samples (list): List of paths to sampled document files
    attr_samples_dict (dict): Dictionary of attribute types to lists of attribute file paths
    filter_config (dict): Filter configuration containing include and exclude filters
    num_lines (int): Number of lines to sample from each file
    
    Returns:
    dict: Results of filter command execution
    """
    print(f"Executing filter commands on {len(doc_samples)} sampled documents, up to {num_lines} lines each")

    results = {
        'include': {'passed': 0, 'total': 0, 'errors': []},
        'exclude': {'passed': 0, 'total': 0, 'errors': []}
    }

    for doc_index, doc_path in enumerate(doc_samples):
        doc_lines = sample_file_lines(doc_path, num_lines)

        if doc_lines is None or len(doc_lines) == 0:
            raise ValueError(f"Unable to sample lines from document file: {doc_path}")
        print(f"sampled {len(doc_lines)} lines from {doc_path}")
        return "success"
    

        # for line_index, doc_line in enumerate(doc_lines):
        #     try:
        #         doc = json.loads(doc_line)
        #         combined_doc = doc.copy()
        #         combined_doc['attributes'] = {}

        #         # Process all attribute files
        #         for attr_type, attr_paths in attr_samples_dict.items():
        #             attr_path = attr_paths[doc_index]
        #             attr_lines = sample_file_lines(attr_path, num_lines)
        #             if attr_lines is None:
        #                 print(f"Skipping attribute {attr_path} due to sampling error")
        #                 continue
        #             if line_index < len(attr_lines):
        #                 attr = json.loads(attr_lines[line_index])
        #                 combined_doc['attributes'].update(attr['attributes'])

        #         for filter_type in ['include', 'exclude']:
        #             for filter_expr in filter_config.get(filter_type, []):
        #                 results[filter_type]['total'] += 1
        #                 try:
        #                     if filter_expr.startswith('$'):
        #                         # JSONPath
        #                         jsonpath_expr = parse_jsonpath(filter_expr)
        #                         matches = jsonpath_expr.find(combined_doc)
        #                     else:
        #                         # JQ
        #                         matches = jq.compile(filter_expr).input(combined_doc).all()

        #                     if matches:
        #                         results[filter_type]['passed'] += 1
        #                     else:
        #                         results[filter_type]['errors'].append(f"Filter '{filter_expr}' did not match for document {doc_path}, line {line_index+1}")
        #                 except Exception as e:
        #                     results[filter_type]['errors'].append(f"Error executing filter '{filter_expr}' on document {doc_path}, line {line_index+1}: {str(e)}")

        #     except json.JSONDecodeError as e:
        #         print(f"Error decoding JSON in file {doc_path}, line {line_index+1}: {str(e)}")
        #     except Exception as e:
        #         print(f"Error processing line {line_index+1} in file {doc_path}: {str(e)}")

    return results

def sample_documents_with_attributes(doc_file_paths, attr_file_paths, num_samples=100):
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


def main(config_path, num_samples):
    print("Loading configuration file...")
    config = load_config(config_path)

    print("Validating configuration structure...")
    errors = validate_config_structure(config)

    if errors:
        print("Configuration validation failed. Errors:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("Validating S3 paths and permissions...")
    for stream in config['streams']:
        # Assuming the first document pattern is the base path for attributes
        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)
        # print(f"Base document path: {base_doc_path}")
        # print(f"Base attribute path: {base_attr_path}")

        for doc_pattern in stream['documents']:
            is_valid, error = validate_s3_path(doc_pattern)
            if not is_valid:
                print(f"Error: {error}")
            else:
                # Check if at least one object matches the pattern
                matching_objects = list(list_s3_objects(doc_pattern))
                # Filter out folders (objects ending with '/')
                matching_objects = [obj for obj in matching_objects if not obj.endswith('/')]
                if not matching_objects:
                    print(f"Warning: No objects found matching pattern: {doc_pattern}")
                else:
                    print(f"Found {len(matching_objects)} objects matching pattern: {doc_pattern}")

        output_path = stream['output']['path']
        is_valid, error = validate_s3_path(output_path)
        if not is_valid:
            print(f"Error: {error}")
        else:
            parent_exists, error = check_s3_parent_exists(output_path)
            if not parent_exists:
                print(f"Error: Parent directory does not exist for output path: {output_path}")
            writable, error = check_s3_path_writable(output_path)
            if not writable:
                print(f"Error: {error}")
                sys.exit(1)
            # else:
            #     print(f"Output path is writable: {output_path}")

        for attr in stream['attributes']:
            attr_path = f"{base_attr_path}/{attr}/"
            is_valid, error = validate_s3_path(attr_path)
            if not is_valid:
                print(f"Error: {error}")
            else:
                exists, error = check_s3_path_exists(attr_path)
                if not exists:
                    print(f"Error: Attribute path does not exist: {attr_path}")
                    sys.exit(1)
                else:
                    print(f"Found attribute path: {attr_path}")
                    # Check if there are any objects in this path
                    objects = list(list_s3_objects(attr_path))
                    if not objects:
                        print(f"  Warning: No objects found in this attribute path")

    print("Validating filter expressions...")
    for stream in config['streams']:
        if 'filter' in stream:
            filter_config = stream['filter']
            filter_errors, filter_warnings = validate_filter_expressions(filter_config)
            
            if filter_warnings:
                print(f"Warnings in filter configuration for stream '{stream['name']}':")
                for warning in filter_warnings:
                    print(f"- Warning: {warning}")

            if filter_errors:
                print(f"Errors in filter expressions for stream '{stream['name']}':")
                for error in filter_errors:
                    print(f"- {error}")


    print(f"Sampling and validating document-attribute alignment, filters, and attribute names...")
    for stream in config['streams']:
        # Extract filter attributes once per stream
        doc_file_paths = stream['documents']
        attr_file_paths = []
        filter_attributes = set()
        if 'filter' in stream:
            include_filters = stream['filter'].get('include', [])
            exclude_filters = stream['filter'].get('exclude', [])
            filter_attributes = extract_attribute_names_from_filters(include_filters + exclude_filters)
        # print(f"Extracted filter attributes: {filter_attributes}")

        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)

        # Sample and download files
        local_doc_samples, local_attr_samples_dict = sample_and_download_files(stream, num_samples)

        for local_doc_sample in local_doc_samples:
            print(f"\nValidating file: {local_doc_sample}")
            
            # Count lines in the document
            doc_line_count = count_file_lines(local_doc_sample)
            if doc_line_count == -1:
                print("Failed to count lines in document file. Check the file and try again.")
                sys.exit(1)
            print(f"Document has {doc_line_count} lines")
            
            # Validate document JSONL
            doc_expected_fields = {'id', 'text', 'source', 'created', 'added', 'version', 'metadata', 'attributes'}
            is_valid, error_messages = validate_jsonl(local_doc_sample, doc_expected_fields)
            if not is_valid:
                print("Document validation failed:")
                for error in error_messages:
                    print(f"  {error}")
                sys.exit(1)
            
            for attr_type in stream['attributes']:
                local_attr_sample = local_attr_samples_dict[attr_type][local_doc_samples.index(local_doc_sample)]
                print(f"\nValidating attribute file: {local_attr_sample}")
                
                # Count lines in the attribute file
                print(f"Counting lines in attribute file: {local_attr_sample}")
                attr_line_count = count_file_lines(local_attr_sample)
                if attr_line_count == -1:
                    print("Failed to count lines in attribute file. Skipping further validation for this attribute.")
                    continue
                print(f"Attribute file has {attr_line_count} lines")
                
                # Check if the number of lines in document and attribute files match
                if doc_line_count != attr_line_count:
                    print(f"ERROR: Line count mismatch! Document has {doc_line_count} lines, but attribute file has {attr_line_count} lines.")
                    sys.exit(1)
                else:
                    print("Line count check passed: Document and attribute file have the same number of lines.")
                
                # Validate attribute JSONL
                attr_expected_fields = {'id', 'attributes'}
                is_valid, error_messages = validate_jsonl(local_attr_sample, attr_expected_fields)
                if not is_valid:
                    print("Warning: possible attribute validation mismatch:")
                    for error in error_messages:
                        print(f"  {error}")
                else:
                    print("Attribute validation passed")

        # Validate filters and check for attribute name typos
        if 'filter' in stream:
            validate_filters_and_check_typos([local_attr_sample for attr_samples in local_attr_samples_dict.values() for local_attr_sample in attr_samples], stream['filter'], 
    stream['attributes'])

        # Execute filter commands and analyze results
        filter_execution_results = execute_filter_commands(local_doc_samples, local_attr_samples_dict, stream['filter'], num_lines=100)
        print(filter_execution_results)
                # print("\nFilter Execution Results:")
                # for filter_type, results in filter_execution_results.items():
                #     print(f"  {filter_type.capitalize()} filters:")
                #     print(f"    Passed: {results['passed']}/{results['total']}")
                #     if results['errors']:
                #         print("    Errors:")
                #         for error in results['errors']:
                #             print(f"      - {error}")

            
    # print("Applying JQ/JSONPath filters to sampled data...")
    # print("Checking for filter runtime errors...")
    # print("Validating filter selection results...")
    # print("Verifying attribute names in filters match attribute files...")
    # print("Checking for existence of nested keys used in filter expressions...")
    # print("Verifying file encoding and checking for encoding errors...")

    print("Validation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python validate_mixer.py <path_to_config_file> [number_of_samples]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    num_samples = sys.argv[2] if len(sys.argv) > 2 else 1
    main(config_path, num_samples=1)