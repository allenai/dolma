import yaml
import json
import os
from typing import Dict, Any, List, Union, Type
from env_handler import expand_env_vars_in_config

def load_config(config_path: str) -> Dict[str, Any]:
    """Load the configuration file (YAML or JSON)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")
    try:
        with open(config_path, 'r') as file:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(file)
            elif config_path.endswith('.json'):
                config = json.load(file)
            else:
                raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
            
        config = expand_env_vars_in_config(config)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config file: {str(e)}")

def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """Validate the basic structure of the configuration."""
    required_fields = ['streams', 'processes']
    errors = []

    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
        elif field == 'streams':
            errors.extend(validate_streams(config[field]))
        elif field == 'processes':
            errors.extend(validate_processes(config[field]))

    return errors

def validate_streams(streams: Any) -> List[str]:
    errors = []
    if not isinstance(streams, list):
        errors.append("'streams' should be a list")
    else:
        for i, stream in enumerate(streams):
            stream_errors = validate_stream(stream, i)
            errors.extend(stream_errors)
    return errors

def validate_processes(processes: Any) -> List[str]:
    if not isinstance(processes, int):
        return ["'processes' should be an integer"]
    return []

def validate_stream(stream: Dict[str, Any], index: int) -> List[str]:
    """Validate an individual stream configuration."""
    required_fields = ['name', 'documents', 'attributes', 'output']
    expected_type = {
            'name': str,
            'documents': list,
            'attributes': list,
            'output': dict
        }
    errors = []

    for field in required_fields:
        errors.extend(validate_field(stream, field, expected_type[field], index))

    if 'output' in stream:
        output_errors = validate_output(stream['output'], index)
        errors.extend(output_errors)

    if 'filter' in stream:
        filter_errors = validate_filter_config(stream['filter'], index)
        errors.extend(filter_errors)
    return errors

def validate_field(stream: Dict[str, Any], field: str, expected_type: Union[Type, List[Type]], stream_index: int) -> List[str]:
    """Check if a field is present in the stream and has the expected type."""
    errors = []
    if field not in stream:
        errors.append(f"Stream {stream_index}: Missing required field: {field}")
    elif not isinstance(stream[field], expected_type):
        type_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
        errors.append(f"Stream {stream_index}: '{field}' should be a {type_name}")
    return errors

def validate_output(output: Dict[str, Any], stream_index: int) -> List[str]:
    """Validate the output configuration of a stream."""
    required_fields = ['path', 'max_size_in_bytes']
    errors = []

    for field in required_fields:
        if field not in output:
            errors.append(f"Stream {stream_index} output: Missing required field: {field}")

    if 'max_size_in_bytes' in output and not isinstance(output['max_size_in_bytes'], int):
        errors.append(f"Stream {stream_index} output: 'max_size_in_bytes' should be an integer")

    return errors

def validate_filter_config(filter_config: Dict[str, Any], stream_index: int) -> List[str]:
    """Validate the filter configuration of a stream."""
    errors = []

    if 'include' in filter_config and not isinstance(filter_config['include'], list):
        errors.append(f"Stream {stream_index} filter: 'include' should be a list")

    if 'exclude' in filter_config and not isinstance(filter_config['exclude'], list):
        errors.append(f"Stream {stream_index} filter: 'exclude' should be a list")

    return errors