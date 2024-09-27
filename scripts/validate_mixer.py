import yaml
import json
import sys
import jq
# import jsonpath_ng
# from jsonpath_ng.ext import parser
from jsonpath_ng.ext import parse as parse_jsonpath
import re
import boto3
import random
import tempfile
import os
import shutil
from botocore.exceptions import ClientError
from tqdm import tqdm
import json
import gzip
import io

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

def extract_attribute_names_from_filters(filters):
    attribute_names = set()
    for filter_expr in filters:
        # Extract attribute names from JSONPath expressions
        matches = re.findall(r'@\.([a-zA-Z0-9_]+)', filter_expr)
        attribute_names.update(matches)
    print(f"Extracted attribute names: {attribute_names}")
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
    s3_client = boto3.client('s3')
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
    s3_client = boto3.client('s3')
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
    
    # Remove '**/' from the prefix as we'll search recursively anyway
    prefix = prefix.replace('**/', '')
    
    # Remove the filename pattern (e.g., '*.jsonl.gz') from the prefix
    prefix = '/'.join(prefix.split('/')[:-1]) + '/'
    
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                yield f"s3://{bucket}/{obj['Key']}"

def read_s3_file(s3_path):
    """Read a file from S3."""
    bucket, key = s3_path[5:].split('/', 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read()
    
    if key.endswith('.gz'):
        content = gzip.decompress(content)
    
    return content.decode('utf-8')

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
        # print(f"Valid subset JSONPath: {base_path}{condition}{closing_bracket}")
    
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
    print("Sampled files:")
    print("\n".join(chosen_files))
    return chosen_files

def get_corresponding_attribute_path(doc_path, base_doc_path, base_attr_path, attr_type):
    """Get the corresponding attribute path for a given document path and attribute type."""
    relative_path = doc_path.replace(base_doc_path, '', 1)
    relative_path = relative_path.lstrip('/')
    # print(f"Relative path: {relative_path}")
    # print(f"Attr type: {attr_type}")
    # print("Full attribute path:", f"{base_attr_path.rstrip('/')}/{attr_type}/{relative_path}")
    return f"{base_attr_path.rstrip('/')}/{attr_type}/{relative_path}"

# def validate_document_attribute_alignment(config, document):
#     """Validate alignment between document and attribute files."""
#     print(f"Sampling and validating {document}...")
#     for stream in config['streams']:
#         base_doc_path = get_base_path(stream['documents'][0])
#         base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)

#         # Sample documents
#         sampled_docs = []
#         for doc_pattern in stream['documents']:
#             sampled_docs.extend(sample_files(doc_pattern, num_samples))

#         mismatches = []
#         matches = []
#         for doc_path in sampled_docs:
#             doc_content = read_s3_file(doc_path)
#             if doc_content is None:
#                 mismatches.append((doc_path, [("all", "Document file not found")]))
#                 continue

#             # Check for corresponding attribute files
#             doc_matches = []
#             doc_mismatches = []
#             for attr in stream['attributes']:
#                 attr_path = get_corresponding_attribute_path(doc_path, base_doc_path, base_attr_path, attr)
#                 attr_content = read_s3_file(attr_path)
                
#                 if attr_content is None:
#                     doc_mismatches.append((attr, "Attribute file not found"))
#                 else:
#                     span_mismatches = validate_attribute_spans(doc_content, attr_content)
#                     if span_mismatches:
#                         doc_mismatches.extend((attr, mismatch) for mismatch in span_mismatches)
#                     else:
#                         doc_matches.append((attr, attr_path))
            
#             if doc_mismatches:
#                 mismatches.append((doc_path, doc_mismatches))
#             if doc_matches:
#                 matches.append((doc_path, doc_matches))

#         print(f"\nResults for stream '{stream['name']}':")
#         print(f"Total sampled documents: {len(sampled_docs)}")
#         print(f"Documents with all attributes matched: {len(matches)}")
#         print(f"Documents with some or all attributes missing or misaligned: {len(mismatches)}")

#         if mismatches:
#             print("\nDocuments with missing or misaligned attributes:")
#             for doc, issues in mismatches:
#                 print(f"  Document: {doc}")
#                 for attr, issue in issues:
#                     print(f"    {attr}: {issue}")

#         if not mismatches:
#             print("\nAll sampled documents have corresponding and correctly aligned attribute files.")


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
        if file_path.startswith('s3://'):
            content = read_s3_file(file_path)
            f = io.StringIO(content)
        else:
            if file_path.endswith('.gz'):
                f = gzip.open(file_path, 'rt')
            else:
                f = open(file_path, 'r')
        
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                missing_fields = expected_fields - set(data.keys())
                new_fields = set(data.keys()) - expected_fields
                
                if missing_fields:
                    error_messages.append(f"Line {i}: Missing expected fields: {missing_fields}")
                    is_valid = False
                
                if new_fields:
                    # error_messages.append(f"Line {i}: Unexpected new fields: {new_fields}")
                    unexpected_fields.update(new_fields)
                    is_valid = False
                
            except json.JSONDecodeError:
                error_messages.append(f"Line {i}: Invalid JSON")
                is_valid = False
    
    except Exception as e:
        error_messages.append(f"Error reading file {file_path}: {str(e)}")
        is_valid = False
    
    finally:
        if not file_path.startswith('s3://'):
            f.close()
    
    if unexpected_fields:
        error_messages.append(f"Unexpected fields found across the file: {unexpected_fields}")
    
    return is_valid, error_messages

def count_file_lines(file_path):
    """
    Count the number of lines in a file (local or S3, compressed or not).
    
    :param file_path: Path to the file (can be S3 or local)
    :return: Number of lines in the file
    """
    try:
        if file_path.startswith('s3://'):
            content = read_s3_file(file_path)
            f = io.StringIO(content)
        else:
            if file_path.endswith('.gz'):
                f = gzip.open(file_path, 'rt')
            else:
                f = open(file_path, 'r')
        
        line_count = sum(1 for _ in f)
        
        if not file_path.startswith('s3://'):
            f.close()
        
        return line_count
    
    except Exception as e:
        print(f"Error counting lines in file {file_path}: {str(e)}")
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
            # Suggest similar attributes from the sample data
            similar = [s_attr for s_attr in sample_attributes if s_attr.startswith(attr[:5])]
            if similar:
                print(f"    Similar attributes in sample data: {', '.join(similar)}")
    
    if extra_in_sample:
        print("Info: The following attributes are in the sample data but not used in the config:")
        for attr in extra_in_sample:
            print(f"  - {attr}")

def validate_filters_and_check_typos(config, sample_file_path):
    """Validate filters and check for attribute name typos."""
    # Extract attribute names from config filters
    config_attributes = set()
    for stream in config['streams']:
        if 'filter' in stream:
            include_filters = stream['filter'].get('include', [])
            exclude_filters = stream['filter'].get('exclude', [])
            config_attributes.update(extract_attribute_names_from_filters(include_filters + exclude_filters))
    
    # Extract attribute names from sample data
    sample_attributes = set()
    with open(sample_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'attributes' in data:
                sample_attributes.update(data['attributes'].keys())
    
    # Check for typos
    check_attribute_name_typos(config_attributes, sample_attributes)
    
    # Validate filters (reusing the previous implementation)
    for stream in config['streams']:
        if 'filter' in stream:
            syntax = stream['filter'].get('syntax', 'jsonpath')
            include_filters = stream['filter'].get('include', [])
            exclude_filters = stream['filter'].get('exclude', [])
            validate_filters(sample_file_path, include_filters, exclude_filters, syntax)

def validate_filters_on_sample_data(file_path, include_filters, exclude_filters, syntax):
    """Validate include and exclude filters on sampled data."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            print(f"Validating filters for document {i}")
            
            for filter_expr in include_filters + exclude_filters:
                try:
                    jsonpath_expr = parse_jsonpath.parse(filter_expr)
                    result = [match.value for match in jsonpath_expr.find(data)]
                    print(f"Filter '{filter_expr}' result: {result}")
                except Exception as e:
                    print(f"Error applying filter '{filter_expr}': {str(e)}")

def validate_attribute_spans(doc_content, attr_content):
    """Validate that attribute spans align with the document content."""
    mismatches = []
    
    doc_lines = doc_content.splitlines()
    print(f"Number of lines in document: {len(doc_lines)}")
    print(f"Number of lines in attribute: {len(attr_content.splitlines())}")
    attr_lines = attr_content.splitlines()
    
    if len(doc_lines) != len(attr_lines):
        mismatches.append(f"Mismatch in number of lines: Document has {len(doc_lines)}, Attribute file has {len(attr_lines)}")
        return mismatches
    
    for i, (doc_line, attr_line) in enumerate(zip(doc_lines, attr_lines), 1):
        try:
            doc_data = json.loads(doc_line)
            attr_data = json.loads(attr_line)
            
            if doc_data['id'] != attr_data['id']:
                mismatches.append(f"Line {i}: Mismatched IDs: Doc ID {doc_data['id']}, Attr ID {attr_data['id']}")
            
            doc_len = len(doc_data['text'])
            for attr_name, attr_values in attr_data['attributes'].items():
                for start, end, _ in attr_values:
                    if end > doc_len:
                        mismatches.append(f"Line {i}: Attribute '{attr_name}' span [{start}, {end}] exceeds document length {doc_len}")
        except json.JSONDecodeError:
            mismatches.append(f"Line {i}: Invalid JSON in document or attribute file")
    
    return mismatches

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
    # for stream in tqdm(config['streams'], desc="Validating streams"):
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
            else:
                print(f"Output path is writable: {output_path}")

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

    print(f"Sampling and validating document-attribute alignment...")
    for stream in config['streams']:
        doc_samples = sample_files(stream['documents'][0], num_samples)
        for doc_sample in doc_samples:
            print(f"\nValidating file: {doc_sample}")
            
            # Count lines in the document
            doc_line_count = count_file_lines(doc_sample)
            if doc_line_count == -1:
                print("Failed to count lines in document file. Skipping further validation for this file.")
                continue
            print(f"Document has {doc_line_count} lines")
            
            # Validate document JSONL
            doc_expected_fields = {'id', 'text', 'source', 'created', 'added', 'version', 'metadata', 'attributes'}
            is_valid, error_messages = validate_jsonl(doc_sample, doc_expected_fields)
            if not is_valid:
                print("Document validation failed:")
                for error in error_messages:
                    print(f"  {error}")
            else:
                print("Document validation passed")
            
            for attr_type in stream['attributes']:
                attr_sample = get_corresponding_attribute_path(doc_sample, base_doc_path, base_attr_path, attr_type)
                print(f"\nValidating attribute file: {attr_sample}")
                
                # Count lines in the attribute file
                attr_line_count = count_file_lines(attr_sample)
                if attr_line_count == -1:
                    print("Failed to count lines in attribute file. Skipping further validation for this attribute.")
                    continue
                print(f"Attribute file has {attr_line_count} lines")
                
                # Check if the number of lines in document and attribute files match
                if doc_line_count != attr_line_count:
                    print(f"ERROR: Line count mismatch! Document has {doc_line_count} lines, but attribute file has {attr_line_count} lines.")
                else:
                    print("Line count check passed: Document and attribute file have the same number of lines.")
                
                # Validate attribute JSONL (you'll need to define the expected fields for attributes)
                attr_expected_fields = {'id', 'attributes'}
                is_valid, error_messages = validate_jsonl(attr_sample, attr_expected_fields)
                if not is_valid:
                    print("Attribute validation failed:")
                    for error in error_messages:
                        print(f"  {error}")
                else:
                    print("Attribute validation passed")

                # attr_expected_fields = {'id', 'attributes'}
            
            # for attr_type in stream['attributes']:
            #     attr_sample = get_corresponding_attribute_path(doc_sample, base_doc_path, base_attr_path, attr_type)
            #     print(f"Validating attribute file: {attr_sample}")
            #     # validate_jsonl(attr_sample)
            #     print("Validate that attribute spans align with the document content...")
                # Not working properly yet
                # mismatches = validate_attribute_spans(doc_sample, attr_sample)
                # print(f"Found {len(mismatches)} mismatches:")
                # print("\n".join(mismatches))
        
        # print("Applying filters to sampled data...")
        # if 'filter' in stream:
        #     syntax = stream['filter'].get('syntax', 'jsonpath')
        #     include_filters = stream['filter'].get('include', [])
        #     exclude_filters = stream['filter'].get('exclude', [])
            # validate_filters_on_sample_data(doc_samples[0], include_filters, exclude_filters, syntax)
            # TO DO: UPDATE function to read from s3 not local file
    
    # print("Validating filters and checking for attribute name typos...")
    # validate_filters_and_check_typos(config, doc_samples[0])

    print("Validation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python validate_mixer.py <path_to_config_file> [number_of_samples]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    num_samples = sys.argv[2] if len(sys.argv) > 2 else 1
    main(config_path, num_samples=1)