import yaml
import json
import sys
import jq

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
    errors = []

    required_keys = ['max_size_in_bytes', 'path']
    for key in required_keys:
        if key not in output:
            errors.append(f"Stream {stream_index} output: Missing required key '{key}'")

    if 'max_size_in_bytes' in output and not isinstance(output['max_size_in_bytes'], int):
        errors.append(f"Stream {stream_index} output: 'max_size_in_bytes' should be an integer")

    return errors

def validate_filter(filter_config, stream_index):
    """Validate the filter configuration of a stream."""
    errors = []

    if 'syntax' in filter_config:
        if filter_config['syntax'] == 'jq':
            for filter_type in ['include', 'exclude']:
                if filter_type in filter_config:
                    for i, expr in enumerate(filter_config[filter_type]):
                        is_valid, error_msg = validate_jq_expression(expr)
                        if not is_valid:
                            errors.append(f"Stream {stream_index} filter: Invalid JQ expression in {filter_type}[{i}]: {error_msg}")
        elif filter_config['syntax'] != 'jsonpath':
            errors.append(f"Stream {stream_index} filter: Unsupported syntax '{filter_config['syntax']}'. Use 'jq' or 'jsonpath'")

    return errors

def main(config_path):
    config = load_config(config_path)
    errors = validate_config_structure(config)

    if errors:
        print("Configuration validation failed. Errors:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    else:
        print("Configuration validation passed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dolma_config_validator.py <path_to_config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)