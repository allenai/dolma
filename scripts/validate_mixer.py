import yaml
import json
import sys
import jq
import re
import boto3
import os
import gzip
import struct
from botocore.exceptions import ClientError

# Initialize the S3 client
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

def validate_jq_expression(expr):
    """Validate a JQ expression."""
    try:
        jq.compile(expr)
        return True, None
    except ValueError as e:
        return False, str(e)
    
    
def jq_dryrun(expressions, sample_file_path, sample_size=10):
    """
    Perform a dry run of JQ expressions on a sample gzipped JSONL file.
    
    :param expressions: List of JQ expressions to test
    :param sample_file_path: Path to a sample gzipped JSONL file to test against
    :param sample_size: Number of lines to sample from the file (default 10)
    :return: List of tuples (expression, success, error_message)
    """
    results = []

    try:
        # Read a sample of lines from the gzipped file
        with gzip.open(sample_file_path, 'rt') as file:
            sample_lines = [next(file) for _ in range(sample_size)]
        
        # Debugging: Print the first few lines of the sample
        print("Sample file content (first 3 lines):")
        for i, line in enumerate(sample_lines[:3]):
            print(f"Line {i+1}: {line[:100]}...")  # Print first 100 characters of each line
    except Exception as e:
        return [(expr, False, f"Error reading sample file: {str(e)}") for expr in expressions]

    for expr in expressions:
        try:
            # Compile the JQ expression
            jq_program = jq.compile(expr)
            
            # Try to apply the JQ expression to each line
            for i, line in enumerate(sample_lines):
                try:
                    json_data = json.loads(line)
                    result = jq_program.input(json_data).all()
                    # You might want to add some logic here to check if the result is what you expect
                except json.JSONDecodeError:
                    results.append((expr, False, f"Invalid JSON on line {i+1}"))
                    break
                except Exception as e:
                    results.append((expr, False, f"JQ expression failed on line {i+1}: {str(e)}"))
                    break
            else:
                # This else clause runs if the for loop completes without breaking
                results.append((expr, True, "Dry run successful"))
        
        except Exception as e:
            results.append((expr, False, f"Error during dry run: {str(e)}"))

    return results


def parse_s3_path(s3_path):
    """Parse an S3 path into bucket name and key."""
    pattern = r'^s3://(?P<bucket>[\w.-]+)/(?P<key>.+)$'
    match = re.match(pattern, s3_path)
    if not match:
        raise ValueError(f"Invalid S3 path: {s3_path}")
    return match.group('bucket'), match.group('key')

def validate_s3_path(s3_path):
    """Validate an S3 path and check if it exists."""
    try:
        bucket, key = parse_s3_path(s3_path)
    except ValueError as e:
        return False, str(e)

    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return False, f"Bucket does not exist: {bucket}"
        elif error_code == '403':
            return False, f"Access denied to bucket: {bucket}"
        else:
            return False, f"Error checking bucket: {str(e)}"

    # Handle wildcard paths
    if '*' in key:
        prefix = key.split('*')[0]
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
            if 'Contents' not in response:
                return False, f"No objects found matching: {s3_path}"
        except ClientError as e:
            return False, f"Error listing objects: {str(e)}"
    else:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            return False, f"Object does not exist: {s3_path}"

    return True, None

def diagnose_file(file_path):
    """
    Diagnose issues with a potentially gzipped file.
    
    :param file_path: Path to the file to diagnose
    :return: A diagnostic message
    """
    if not os.path.exists(file_path):
        return f"File does not exist: {file_path}"

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return f"File is empty: {file_path}"

    with open(file_path, 'rb') as f:
        file_start = f.read(10)  # Read the first 10 bytes

    if file_start.startswith(b'\x1f\x8b'):
        # It's a gzip file, let's try to read it
        try:
            with gzip.open(file_path, 'rb') as gz:
                gz.read(1)  # Try to read 1 byte
            return f"File appears to be a valid gzip file: {file_path}"
        except gzip.BadGzipFile:
            return f"File starts with gzip header but is corrupt: {file_path}"
        except Exception as e:
            return f"Error reading gzip file: {file_path}. Error: {str(e)}"
    else:
        # It's not a gzip file
        return f"File is not gzipped (first 10 bytes: {file_start.hex()}): {file_path}"

def get_sample_file(documents):
    """
    Get a sample file path from the list of documents in the config.
    
    :param documents: List of document paths from the config
    :return: Path to a sample file, or None if no suitable file is found
    """
    for doc in documents:
        if doc.startswith('s3://'):
            try:
                bucket, key = parse_s3_path(doc)
            except ValueError:
                continue

            # Handle wildcard paths
            if '*' in key:
                prefix = key.split('*')[0]
                try:
                    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
                    if 'Contents' in response:
                        key = response['Contents'][0]['Key']
                except ClientError as e:
                    print(f"Error listing objects: {str(e)}")
                    continue

            local_path = f"/tmp/sample_{os.path.basename(key)}"
            try:
                s3_client.download_file(bucket, key, local_path)
                print(diagnose_file(local_path))  # Diagnose the downloaded file
                return local_path
            except ClientError as e:
                print(f"Error downloading sample file: {str(e)}")
        else:
            # Handle local paths
            if os.path.isfile(doc):
                print(diagnose_file(doc))  # Diagnose the local file
                return doc
            elif doc.endswith('*'):
                directory = os.path.dirname(doc)
                for filename in os.listdir(directory):
                    if filename.endswith('.gz') or filename.endswith('.json'):
                        full_path = os.path.join(directory, filename)
                        print(diagnose_file(full_path))  # Diagnose the found file
                        return full_path
    
    return None

def print_s3_debug_info(s3_path):
    try:
        bucket, key = parse_s3_path(s3_path)
        print(f"Debugging S3 path: {s3_path}")
        print(f"Bucket: {bucket}")
        print(f"Key: {key}")
        
        if '*' in key:
            prefix = key.split('*')[0]
            try:
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
                if 'Contents' in response:
                    print("Found the following objects:")
                    for obj in response['Contents']:
                        print(f"- {obj['Key']}")
                else:
                    print("No objects found with this prefix.")
            except ClientError as e:
                print(f"Error listing objects: {str(e)}")
        else:
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                print("Object exists.")
            except ClientError as e:
                print(f"Error checking object: {str(e)}")
    except ValueError as e:
        print(f"Error parsing S3 path: {str(e)}")


def validate_config_structure(config):
    """Validate the basic structure of the configuration."""
    errors = []

    if 'streams' not in config:
        errors.append("Missing 'streams' key in config")
    elif not isinstance(config['streams'], list):
        errors.append("'streams' should be a list")
    else:
        for i, stream in enumerate(config['streams']):
            stream_errors = validate_stream(stream, i)
            errors.extend(stream_errors)

    if 'processes' not in config:
        errors.append("Missing 'processes' key in config")
    elif not isinstance(config['processes'], int):
        errors.append("'processes' should be an integer")

    return errors

def validate_stream(stream, index):
    """Validate an individual stream configuration."""
    errors = []

    required_keys = ['name', 'documents', 'attributes', 'output']
    for key in required_keys:
        if key not in stream:
            errors.append(f"Stream {index}: Missing required key '{key}'")

    if 'documents' in stream:
        if not isinstance(stream['documents'], list):
            errors.append(f"Stream {index}: 'documents' should be a list")
        else:
            for doc in stream['documents']:
                is_valid, error_msg = validate_s3_path(doc)
                if not is_valid:
                    errors.append(f"Stream {index}: Invalid S3 path in 'documents': {error_msg}")

    if 'attributes' in stream and not isinstance(stream['attributes'], list):
        errors.append(f"Stream {index}: 'attributes' should be a list")

    if 'output' in stream:
        output_errors = validate_output(stream['output'], index)
        errors.extend(output_errors)

    if 'filter' in stream:
        filter_errors = validate_filter(stream['filter'], index, stream.get('documents', []))
        errors.extend(filter_errors)

    return errors

def validate_output(output, stream_index):
    """Validate the output configuration of a stream."""
    errors = []

    required_keys = ['max_size_in_bytes', 'path']
    for key in required_keys:
        if key not in output:
            errors.append(f"Stream {stream_index} output: Missing required key '{key}'")

    if 'max_size_in_bytes' in output and not isinstance(output['max_size_in_bytes'], int):
        errors.append(f"Stream {stream_index} output: 'max_size_in_bytes' should be an integer")

    if 'path' in output:
        is_valid, error_msg = validate_s3_path(output['path'])
        if not is_valid:
            errors.append(f"Stream {stream_index} output: Invalid S3 path: {error_msg}")

    return errors

def validate_filter(filter_config, stream_index, documents):
    """Validate the filter configuration of a stream."""
    errors = []

    if 'syntax' in filter_config:
        if filter_config['syntax'] == 'jq':
            expressions = []
            for filter_type in ['include', 'exclude']:
                if filter_type in filter_config:
                    expressions.extend(filter_config[filter_type])
            
            sample_file_path = get_sample_file(documents)
            if sample_file_path:
                dryrun_results = jq_dryrun(expressions, sample_file_path)
                
                for expr, success, message in dryrun_results:
                    if not success:
                        errors.append(f"Stream {stream_index} filter: JQ dry run failed for expression '{expr}': {message}")
                
                # Clean up the sample file if it was downloaded from S3
                if sample_file_path.startswith('/tmp/'):
                    os.remove(sample_file_path)
            else:
                errors.append(f"Stream {stream_index} filter: Unable to find a sample file for JQ validation")
        
        elif filter_config['syntax'] != 'jsonpath':
            errors.append(f"Stream {stream_index} filter: Unsupported syntax '{filter_config['syntax']}'. Use 'jq' or 'jsonpath'")

    return errors

def main(config_path):
    config = load_config(config_path)
    errors = validate_config_structure(config)

    # Print debug information for S3 paths
    for stream in config.get('streams', []):
        for doc in stream.get('documents', []):
            print_s3_debug_info(doc)

    if errors:
        print("Configuration validation failed. Errors:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("Configuration validation passed successfully!")
        print("\nValidation Summary:")
        print(f"- Streams validated: {len(config['streams'])}")
        print(f"- S3 paths checked: {sum(len(stream['documents']) for stream in config['streams'])}")
        print(f"- JQ expressions validated: {sum(len(stream.get('filter', {}).get('include', [])) + len(stream.get('filter', {}).get('exclude', [])) for stream in config['streams'])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_mixer.py <path_to_config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)