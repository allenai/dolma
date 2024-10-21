import re
import os
import shutil
import sys
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

from s3_utils import (
    validate_s3_path,
    check_s3_path_exists,
    check_s3_parent_exists,
    check_s3_path_writable,
    list_s3_objects,
    get_base_path
)
from file_operations import (
    sample_and_download_files,
    count_file_lines,
    validate_jsonl
)
from filter_operations import (
    validate_filter_expressions,
    execute_filter_commands,
    extract_attribute_names_from_filters,
    validate_filters_and_check_typos
)
from config_handler import load_config, validate_config_structure

from env_handler import load_env_variables

from utils import vprint

def load_and_validate_config(config_path):
    #loading environment variables
    load_env_variables()
    
    vprint("Validating configuration file...")
    try:
        config = load_config(config_path)

    except FileNotFoundError as e:
        print(str(e))
        print("Please check the file path and try again.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error loading or validating config: {str(e)}")
        sys.exit(1)

    vprint("Validating configuration structure...")
    errors = validate_config_structure(config)

    if errors:
        print("Configuration validation FAILED. Errors:")
        for error in errors:
            print(f"- {error}")
        return None
    else:
        print("Configuration validation SUCCESSFUL.\n")
        return config

def validate_s3_paths_and_permissions(config: Dict[str, Any]) -> bool:
    vprint("Validating S3 paths and permissions...")
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
                    vprint(f"Found {len(matching_objects)} objects matching pattern: {doc_pattern}")

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
                    vprint(f"Found attribute path: {attr_path}")
                    objects = list(list_s3_objects(attr_path))
                    if not objects:
                        print(f"  Warning: No objects found in this attribute path")

    return True  # All validations passed

def validate_stream_filters(config: Dict[str, Any]) -> bool:
    vprint("\nValidating filter expressions...")
    all_valid = True
    
    for stream in config['streams']:
        if 'filter' in stream:
            filter_config = stream['filter']
            filter_errors, filter_warnings = validate_filter_expressions(filter_config)
            
            if filter_warnings:
                vprint(f"Warnings in filter configuration for stream '{stream['name']}':")
                for warning in filter_warnings:
                    vprint(f"- Warning: {warning}")

            if filter_errors:
                all_valid = False
                print(f"Errors in filter expressions for stream '{stream['name']}':")
                for error in filter_errors:
                    print(f"- {error}")
    print("Filters validation COMPLETE.\n")
    
    return all_valid

def validate_documents_and_attributes(config: Dict[str, Any], num_samples: int) -> bool:
    vprint("Sampling files...")
    temp_dir = "temp_sample_files"
    try:
        for stream in config['streams']:
            filter_attributes = set()
            if 'filter' in stream:
                include_filters = stream['filter'].get('include', [])
                exclude_filters = stream['filter'].get('exclude', [])
                filter_attributes = extract_attribute_names_from_filters(include_filters + exclude_filters)

            base_doc_path = get_base_path(stream['documents'][0])
            base_attr_path = re.sub(r'/documents($|/)', r'/attributes\1', base_doc_path)
            
            try:
                doc_samples, attr_samples_dict = sample_and_download_files(stream, num_samples)
            except Exception as e:
                print(f"Error during file sampling and downloading: {str(e)}")
                return False
            
            if not doc_samples:
                print("No document samples were successfully downloaded. Skipping further validation for this stream.")
                continue

            for doc_sample in doc_samples:
                vprint(f"\nValidating file: {doc_sample}")
                
                doc_line_count = count_file_lines(doc_sample)
                if doc_line_count == -1:
                    print(f"Failed to count lines in document file {doc_sample}. Skipping the file")
                    continue

                vprint(f"Document has {doc_line_count} lines")
                
                doc_expected_fields = {'id', 'text', 'source', 'created', 'added', 'version', 'metadata', 'attributes'}
                is_valid, error_messages = validate_jsonl(doc_sample, doc_expected_fields)
                if not is_valid:
                    print("Document validation failed:")
                    for error in error_messages:
                        print(f"  {error}")
                    return False
                
                for attr_type in stream['attributes']:
                    if attr_type not in attr_samples_dict or not attr_samples_dict[attr_type]:
                        print(f"Warning: No attribute samples found for {attr_type}. Skipping validation for this attribute type.")
                        continue

                    try:
                        doc_index = doc_samples.index(doc_sample)
                        if doc_index >= len(attr_samples_dict[attr_type]):
                            print(f"Warning: No corresponding attribute file for document {doc_sample} and attribute type {attr_type}. Skipping validation for this attribute.")
                            continue
                        attr_sample = attr_samples_dict[attr_type][doc_index]
                    except ValueError:
                        print(f"Warning: Document {doc_sample} not found in samples. Skipping validation for this document.")
                        continue

                    vprint(f"\nValidating attribute file: {attr_sample}")
                    
                    attr_line_count = count_file_lines(attr_sample)
                    if attr_line_count == -1:
                        print("Failed to count lines in attribute file. Skipping further validation for this attribute.")
                        continue

                    vprint(f"Attribute file has {attr_line_count} lines")
                    
                    if doc_line_count != attr_line_count:
                        print(f"Document: {doc_sample}")
                        print(f"Attribute: {attr_sample}")
                        print(f"ERROR: Line count mismatch! Document has {doc_line_count} lines, but attribute file has {attr_line_count} lines.")
                        return False
                    else:
                        vprint("Line count check PASSED: Document and attribute file have the same number of lines.")
                    
                    attr_expected_fields = {'id', 'attributes'}
                    is_valid, error_messages = validate_jsonl(attr_sample, attr_expected_fields)
                    if not is_valid:
                        print("Warning: possible attribute validation mismatch:")
                        for error in error_messages:
                            print(f"  {error}")
                    else:
                        print("Attribute validation PASSED\n")

            if 'filter' in stream:
                validate_filters_and_check_typos([attr_sample for attr_samples in attr_samples_dict.values() for attr_sample in attr_samples], stream['filter'], stream['attributes'])

            filter_execution_results = execute_filter_commands(attr_samples_dict, stream['attributes'], stream['filter'], num_lines=100)
            vprint(filter_execution_results)

        return True
    finally:
        # Clean up: remove the temporary directory and its contents
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                # print(f"Temporary directory '{temp_dir}' has been removed.")
            except Exception as e:
                print(f"Error while removing temporary directory '{temp_dir}': {str(e)}")

