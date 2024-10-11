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
import importlib
import traceback

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
                # Check for and remove '@' only if it appears in "$@." pattern at the beginning
                if expr.startswith('$@.'):
                    expr = '$' + expr[2:]
                    warnings.append(f"Removed '@' from '$@.' at the beginning of the expression: {expr}")

                # conditions = split_complex_jsonpath(expr)
                conditions = split_complex_jsonpath(expr)
                for condition in conditions:
                    # Check for comparison operators
                    skip_operators = ["==", "<=", ">=", "<", ">"]
                    if any(op in condition for op in skip_operators):
                        operator = next(op for op in skip_operators if op in condition)
                        # warnings.append(f"Temporarily skipping expression because it contains '{operator}' operator: {condition}")
                        continue
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
    return chosen_files

def download_file(s3_path, local_path):
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3_client.download_file(bucket, key, local_path)

# For local testing
# def sample_and_download_files(stream, num_samples):
#     temp_dir = "temp_sample_files"
    
#     # Hardcoded file names
#     doc_file = "CC-MAIN-20240221102433-20240221132433-00460.jsonl.gz"
#     attr_files = {
#         "bff_duplicate_paragraph_spans": "CC-MAIN-20240221102433-20240221132433-00460-bff_duplicate_paragraph_spans.jsonl.gz",
#         "cc_tiny_subset_analysis": "CC-MAIN-20240221102433-20240221132433-00460-cc_tiny_subset_analysis.jsonl"
#     }
    
#     # Create full paths
#     local_doc_samples = [os.path.join(temp_dir, doc_file)]
#     local_attr_samples_dict = {
#         attr_type: [os.path.join(temp_dir, file_name)]
#         for attr_type, file_name in attr_files.items()
#     }
    
#     print("Using hardcoded paths for troubleshooting:")
#     print(f"Document: {local_doc_samples[0]}")
#     for attr_type, paths in local_attr_samples_dict.items():
#         print(f"{attr_type}: {paths[0]}")
    
#     return local_doc_samples, local_attr_samples_dict

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
    
def count_file_lines(file_path):
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
    filter_attributes = extract_filter_attributes(filter_config)
    
    # Sample and extract attributes from all files
    all_sampled_attributes = set()
    for attr_file_path in attr_file_paths:
        sampled_attributes = sample_and_extract_attributes(attr_file_path)
        all_sampled_attributes.update(sampled_attributes)
    
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
        
        print("\nThis detailed list is provided to help identify potential typos or misconfigurations\n")


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

def evaluate_comparison(value, op, comparison_value):
    if op == '==':
        return value == comparison_value
    elif op == '<':
        return value < comparison_value
    elif op == '>':
        return value > comparison_value
    elif op == '<=':
        return value <= comparison_value
    elif op == '>=':
        return value >= comparison_value
    else:
        raise ValueError(f"Unsupported operator: {op}")

def evaluate_jsonpath_condition(data, condition):
    # print(f"\nEvaluating condition: {condition}")
    try:
        # Check if the condition contains a comparison
        match = re.match(r'(.*?)\s*(==|<=|>=|<|>)\s*(.*)', condition)
        if match:
            path, op, comparison = match.groups()
            comparison_value = float(comparison.strip())
            
            # Extract all indices from the path
            indices = re.findall(r'\[(\d+)\]', path)
            indices = [int(index) for index in indices]
            
            # Remove all indices from the path for evaluation
            path = re.sub(r'\[\d+\]', '', path)
            
            # Evaluate the path part
            jsonpath_expr = parse_jsonpath(path.strip())
            matches = jsonpath_expr.find(data)
            
            if matches:
                value = matches[0].value
                # Apply all indices sequentially
                for index in indices:
                    if isinstance(value, list) and len(value) > index:
                        value = value[index]
                    else:
                        print(f"  Index {index} is out of range or value is not a list")
                        return False
                
                result = evaluate_comparison(value, op, comparison_value)
                # print(f"  Comparison: {value} {op} {comparison_value}")
                # print(f"  Result: {result}")
                return result
            else:
                print("  No matches found for the path")
                return False
        else:
            # Regular JSONPath without comparison
            jsonpath_expr = parse_jsonpath(condition)
            matches = jsonpath_expr.find(data)
            return len(matches) > 0
    except Exception as e:
        print(f"Error evaluating condition: {e}")
        return False
    
def split_complex_jsonpath(complex_expression):
    # Remove the outer quotes, brackets, and any leading/trailing whitespace
    expression = complex_expression.strip().strip('"- ').strip('[]')
    
    # Check for and remove '@' only if it appears in "$@." pattern at the beginning
    if expression.startswith('$@.'):
        expression = '$' + expression[2:]

    # Remove the initial "$.attributes[?(" and the trailing ")]"
    expression = re.sub(r'^\$\.attributes\[\?\(|\)\]$', '', expression)
    
    # Split the expression by '&&' but keep the JSONPath intact
    parts = re.split(r'\s*&&\s*', expression)
    
    conditions = []
    base_path = "$.attributes"
    
    for part in parts:
        # Remove the '@.' at the beginning of each condition
        part = part.replace('@.', '')
        
        # Remove any trailing parenthesis
        part = part.rstrip(')')
        
        # Check if it's a comparison condition
        if any(op in part for op in ['<', '>', '==', '<=', '>=']):
            conditions.append(f"{base_path}.{part}")
        else:
            # For existence checks, we don't need to change anything
            conditions.append(f"{base_path}.{part}")
    
    return conditions

def prepare_filter(filter_expr, syntax):
    if syntax == 'jsonpath':
        return split_complex_jsonpath(filter_expr)
    elif syntax == 'jq':
        return jq.compile(filter_expr)
    else:
        raise ValueError(f"Unsupported filter syntax: {syntax}. Supported options are 'jsonpath' and 'jq'.")

def execute_filter_commands(attr_samples_dict, attributes_list, filter_config, num_lines=100):
    """
    Execute filter commands on sampled lines from documents and their attributes.
    Supports both JQ and JSONPath expressions based on the specified syntax.
    """
    print(f"Executing filter commands on attribute files, sampling {num_lines} lines from each.")

    results = {
        'total_lines': 0,
        'lines_excluded': 0,
        'lines_passed': 0,
        'errors': 0
    }

    include_filters = filter_config.get('include', [])
    exclude_filters = filter_config.get('exclude', [])
    syntax = filter_config.get('syntax', 'jsonpath')

    # Compile JQ expressions or prepare JSONPath expressions based on the specified syntax
    try:
        include_filters_compiled = [prepare_filter(filter_expr, syntax) for filter_expr in include_filters]
        exclude_filters_compiled = [prepare_filter(filter_expr, syntax) for filter_expr in exclude_filters]
    except ValueError as e:
        print(f"Error preparing filters: {str(e)}")
        return results

    # Step 1: Sample lines from each attribute file and build a list of combined attribute data
    attr_lines_list = []
    for attr_name in attributes_list:
        attr_paths = attr_samples_dict.get(attr_name, [])
        if not attr_paths:
            print(f"No attribute files found for '{attr_name}'.")
            return results
        attr_path = attr_paths[0]
        attr_lines = sample_file_lines(attr_path, num_lines)
        if not attr_lines:
            print(f"No lines sampled from attribute file '{attr_path}'.")
            return results
        attr_lines_list.append(attr_lines)

    for lines_tuple in zip(*attr_lines_list):
        results['total_lines'] += 1
        combined_attr_data = {'attributes': {}}
        doc_id = None
        for line in lines_tuple:
            try:
                attr_data = json.loads(line)
                if doc_id is None:
                    doc_id = attr_data.get('id')
                    combined_attr_data['id'] = doc_id
                elif doc_id != attr_data.get('id'):
                    print(f"Mismatch in doc_ids: {doc_id} != {attr_data.get('id')}")
                    results['errors'] += 1
                    break
                combined_attr_data['attributes'].update(attr_data.get('attributes', {}))
            except json.JSONDecodeError:
                results['errors'] += 1
                break
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                results['errors'] += 1
                break
        else:
            # No break occurred, so process filters
            try:
                # Apply include filters
                if syntax == 'jsonpath':
                    include_passed = all(
                        all(evaluate_jsonpath_condition(combined_attr_data, condition) for condition in filter_expr)
                        for filter_expr in include_filters_compiled
                    )
                else:  # JQ
                    include_passed = all(
                        filter_expr.input(combined_attr_data).first()
                        for filter_expr in include_filters_compiled
                    )

                # Apply exclude filters if include filters passed
                if include_passed:
                    if syntax == 'jsonpath':
                        exclude_matched = any(
                            all(evaluate_jsonpath_condition(combined_attr_data, condition) for condition in filter_expr)
                            for filter_expr in exclude_filters_compiled
                        )
                    else:  # JQ
                        exclude_matched = any(
                            filter_expr.input(combined_attr_data).first()
                            for filter_expr in exclude_filters_compiled
                        )
                    
                    if exclude_matched:
                        results['lines_excluded'] += 1
                    else:
                        results['lines_passed'] += 1
                else:
                    results['lines_excluded'] += 1
            except Exception as e:
                print(f"Error applying filters: {str(e)}")
                results['errors'] += 1

    print("\nFilter execution results:")
    print(f"Total lines processed: {results['total_lines']}")
    print(f"Lines passed all filters: {results['lines_passed']}")
    print(f"Lines excluded by filters: {results['lines_excluded']}")
    print(f"Errors encountered: {results['errors']}")
    
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

def validate_s3_paths_and_permissions(config):
    print("Validating S3 paths and permissions...")
    for stream in config['streams']:
        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)

        # Validate document patterns
        for doc_pattern in stream['documents']:
            is_valid, error = validate_s3_path(doc_pattern)
            if not is_valid:
                print(f"Error: {error}")
            else:
                matching_objects = [obj for obj in list_s3_objects(doc_pattern) if not obj.endswith('/')]
                if not matching_objects:
                    print(f"Warning: No objects found matching pattern: {doc_pattern}")
                else:
                    print(f"Found {len(matching_objects)} objects matching pattern: {doc_pattern}")

        # Validate output path
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
                return False

        # Validate attribute paths
        for attr in stream['attributes']:
            attr_path = f"{base_attr_path}/{attr}/"
            is_valid, error = validate_s3_path(attr_path)
            if not is_valid:
                print(f"Error: {error}")
            else:
                exists, error = check_s3_path_exists(attr_path)
                if not exists:
                    print(f"Error: Attribute path does not exist: {attr_path}")
                else:
                    print(f"Found attribute path: {attr_path}")
                    objects = list(list_s3_objects(attr_path))
                    if not objects:
                        print(f"  Warning: No objects found in this attribute path")

    return True  # All validations passed

def validate_stream_filters(config):
    print("\nValidating filter expressions...")
    all_valid = True
    
    for stream in config['streams']:
        if 'filter' in stream:
            filter_config = stream['filter']
            filter_errors, filter_warnings = validate_filter_expressions(filter_config)
            
            if filter_warnings:
                print(f"Warnings in filter configuration for stream '{stream['name']}':")
                for warning in filter_warnings:
                    print(f"- Warning: {warning}")

            if filter_errors:
                all_valid = False
                print(f"Errors in filter expressions for stream '{stream['name']}':")
                for error in filter_errors:
                    print(f"- {error}")
    print("Filters validation complete.\n")
    
    return all_valid

def validate_documents_and_attributes(config, num_samples):
    print("Sampling and validating document-attribute alignment, filters, and attribute names...")
    for stream in config['streams']:
        filter_attributes = set()
        if 'filter' in stream:
            include_filters = stream['filter'].get('include', [])
            exclude_filters = stream['filter'].get('exclude', [])
            filter_attributes = extract_attribute_names_from_filters(include_filters + exclude_filters)

        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)

        doc_samples, attr_samples_dict = sample_and_download_files(stream, num_samples)

        for doc_sample in doc_samples:
            print(f"\nValidating file: {doc_sample}")
            
            doc_line_count = count_file_lines(doc_sample)
            if doc_line_count == -1:
                print("Failed to count lines in document file. Check the file and try again.")
                return False

            print(f"Document has {doc_line_count} lines")
            
            doc_expected_fields = {'id', 'text', 'source', 'created', 'added', 'version', 'metadata', 'attributes'}
            is_valid, error_messages = validate_jsonl(doc_sample, doc_expected_fields)
            if not is_valid:
                print("Document validation failed:")
                for error in error_messages:
                    print(f"  {error}")
                return False
            
            for attr_type in stream['attributes']:
                attr_sample = attr_samples_dict[attr_type][doc_samples.index(doc_sample)]
                print(f"\nValidating attribute file: {attr_sample}")
                
                attr_line_count = count_file_lines(attr_sample)
                if attr_line_count == -1:
                    print("Failed to count lines in attribute file. Skipping further validation for this attribute.")
                    continue

                print(f"Attribute file has {attr_line_count} lines")
                
                if doc_line_count != attr_line_count:
                    print(f"ERROR: Line count mismatch! Document has {doc_line_count} lines, but attribute file has {attr_line_count} lines.")
                    return False
                else:
                    print("Line count check passed: Document and attribute file have the same number of lines.")
                
                attr_expected_fields = {'id', 'attributes'}
                is_valid, error_messages = validate_jsonl(attr_sample, attr_expected_fields)
                if not is_valid:
                    print("Warning: possible attribute validation mismatch:")
                    for error in error_messages:
                        print(f"  {error}")
                else:
                    print("Attribute validation passed")

        if 'filter' in stream:
            validate_filters_and_check_typos([attr_sample for attr_samples in attr_samples_dict.values() for attr_sample in attr_samples], stream['filter'], stream['attributes'])

        filter_execution_results = execute_filter_commands(attr_samples_dict, stream['attributes'], stream['filter'], num_lines=100)
        print(filter_execution_results)

    return True

def load_and_validate_config(config_path):
    print("Loading configuration file...")
    config = load_config(config_path)

    print("Validating configuration structure...")
    errors = validate_config_structure(config)

    if errors:
        print("Configuration validation failed. Errors:")
        for error in errors:
            print(f"- {error}")
        return None
    else:
        print("Configuration validation successful.\n")
        return config

def main(config_path, num_samples):
    config = load_and_validate_config(config_path)
    if config is None:
        print("Configuration loading or validation failed")

    if not validate_s3_paths_and_permissions(config):
        print("S3 path validation failed")

    if not validate_stream_filters(config):
        print("Filter validation failed. \n")

    if not validate_documents_and_attributes(config, num_samples):
        print("Document and attribute validation failed")

    print("Validation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python validate_mixer.py <path_to_config_file> [number_of_file_samples, default=1]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    num_samples = sys.argv[2] if len(sys.argv) > 2 else 1
    main(config_path, num_samples=1)