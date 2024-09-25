import yaml
import json
import sys
import jq
import jsonpath_ng
from jsonpath_ng.ext import parser
import re
import boto3
import random
from botocore.exceptions import ClientError
from tqdm import tqdm

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
        filter_errors = validate_filter(stream['filter'], index)
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

def validate_filter(filter_config, stream_index):
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
    s3_client = boto3.client('s3')
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

def validate_filter_expressions(filter_config):
    """Validate filter expressions based on specified syntax."""
    errors = []
    warnings = []

    syntax = filter_config.get('syntax')
    if syntax is None:
        warnings.append("No 'syntax' key specified. JSONPath expressions are not being evaluated.")
        return errors, warnings

    if syntax.lower() != 'jq':
        warnings.append(f"Unsupported syntax '{syntax}'. Only 'jq' is currently supported for validation.")
        return errors, warnings

    for filter_type in ['include', 'exclude']:
        expressions = filter_config.get(filter_type, [])
        for expr in expressions:
            is_valid, error = validate_jq_expression(expr)
            if not is_valid:
                errors.append(f"Invalid JQ expression in {filter_type}: {expr}. Error: {error}")
            else:
                print(f"Valid JQ expression in {filter_type}: {expr}")

    return errors, warnings

def sample_files(s3_path, num_samples):
    """Sample a subset of files from an S3 path."""
    all_files = list(list_s3_objects(s3_path))
    # Filter out directories (paths ending with '/')
    all_files = [f for f in all_files if not f.endswith('/')]
    return random.sample(all_files, min(num_samples, len(all_files)))

def get_corresponding_attribute_path(doc_path, base_doc_path, base_attr_path):
    """Get the corresponding attribute path for a given document path."""
    relative_path = doc_path.replace(base_doc_path, '', 1)
    # Remove leading slash if present
    relative_path = relative_path.lstrip('/')
    return f"{base_attr_path.rstrip('/')}/{relative_path}"

def validate_document_attribute_alignment(config, num_samples):
    """Validate alignment between document and attribute files."""
    print(f"Sampling and validating {num_samples} files for each stream...")
    for stream in tqdm(config['streams'], desc="Validating streams"):
        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)

        # Sample documents
        sampled_docs = []
        for doc_pattern in stream['documents']:
            sampled_docs.extend(sample_files(doc_pattern, num_samples))

        mismatches = []
        matches = []
        for doc_path in sampled_docs:
            # Check for corresponding attribute files
            doc_matches = []
            doc_mismatches = []
            for attr in stream['attributes']:
                attr_path = get_corresponding_attribute_path(doc_path, base_doc_path, f"{base_attr_path}/{attr}/")
                exists, error = check_s3_path_exists(attr_path)
                if exists:
                    doc_matches.append((attr, attr_path))
                else:
                    doc_mismatches.append((attr, attr_path))
            
            if doc_mismatches:
                mismatches.append((doc_path, doc_mismatches))
            if doc_matches:
                matches.append((doc_path, doc_matches))

        print(f"\nResults for stream '{stream['name']}':")
        print(f"Total sampled documents: {len(sampled_docs)}")
        print(f"Documents with all attributes matched: {len(matches)}")
        print(f"Documents with some or all attributes missing: {len(mismatches)}")

        if matches:
            print("\nSample of matching documents:")
            for doc, attrs in matches[:3]:  # Show up to 3 examples
                print(f"  Document: {doc}")
                for attr, path in attrs:
                    print(f"    Matched {attr}: {path}")
            if len(matches) > 3:
                print("  ...")

        if mismatches:
            print("\nDocuments with missing attributes:")
            for doc, missing_attrs in mismatches:
                print(f"  Document: {doc}")
                for attr, path in missing_attrs:
                    print(f"    Missing {attr}: {path}")

        if not mismatches:
            print("\nAll sampled documents have corresponding attribute files.")


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
    for stream in tqdm(config['streams'], desc="Validating streams"):
        # Assuming the first document pattern is the base path for attributes
        base_doc_path = get_base_path(stream['documents'][0])
        base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)
        
        print(f"Base document path: {base_doc_path}")
        print(f"Base attribute path: {base_attr_path}")

        for doc_pattern in stream['documents']:
            is_valid, error = validate_s3_path(doc_pattern)
            if not is_valid:
                print(f"Error: {error}")
            else:
                # Check if at least one object matches the pattern
                matching_objects = list(list_s3_objects(doc_pattern))
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
                    else:
                        print(f"  The attribute path contains various subdirectories and files")
                        print(f"  Here are a few examples of the contents:")
                        for obj in objects[:3]:  # Print first 3 objects
                            print(f"    - {obj}")
                        if len(objects) > 3:
                            print(f"    ... and more")

    print("Validating filter expressions...")
    for stream in tqdm(config['streams'], desc="Validating filters"):
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
    validate_document_attribute_alignment(config, num_samples)

    print("Validation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python validate_mixer.py <path_to_config_file> [number_of_samples]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    num_samples = sys.argv[2] if len(sys.argv) > 2 else 1
    main(config_path, num_samples=1)