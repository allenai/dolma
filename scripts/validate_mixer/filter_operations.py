import re
import json
from typing import Optional, Union, List, Dict, Any, Tuple
import jq
from jsonpath_ng.ext import parse as parse_jsonpath
from file_operations import sample_file_lines
from utils import vprint

def validate_jq_expression(expr: str) -> Tuple[bool, Optional[str]]:
    """Validate a JQ expression."""
    try:
        jq.compile(expr)
        return True, None
    except ValueError as e:
        return False, str(e)
    
def validate_jsonpath_expression(expr: str) -> Tuple[bool, Optional[str]]:
    """Validate a JSONPath expression."""
    try:
        parse_jsonpath(expr)
        return True, None
    except Exception as e:
        return False, str(e)

def validate_filter_expressions(filter_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
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


def validate_filters_and_check_typos(attr_file_paths: List[str], filter_config: Dict[str, Any], stream_attributes: List[str]):
    """Validate filters and check for attribute name typos across multiple attribute files."""
    vprint("Validating filters and checking typos across all attribute files...")
    
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
        print("All mixer config filters are FOUND in the attribute files.\n")
    else:
        print("Warning: Some mixer config filters were not found in the attribute files.")
        print("Missing attributes:")
        for attr in missing_attributes:
            print(f"  - {attr}")
        
        vprint("\nAll attributes found in files:")
        for attr in sorted(all_sampled_attributes):
            vprint(f"  - {attr}")
        
        vprint("\nThis detailed list is provided to help identify potential typos or misconfigurations\n")

def sample_and_extract_attributes(attr_file_path: str, num_samples: int = 5) -> set:
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

def evaluate_comparison(value: Union[int, float], op: str, comparison_value: Union[int, float]) -> bool:
    if op == '==':
        return value == comparison_value
    elif op == '!=':
        return value != comparison_value
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

def evaluate_jsonpath_condition(data: Dict[str, Any], condition: str) -> bool:
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
    
def split_complex_jsonpath(complex_expression: str) -> List[str]:
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

def prepare_filter(filter_expr: str, syntax: str) -> Union[List[str], Any]:
    if syntax == 'jsonpath':
        return split_complex_jsonpath(filter_expr)
    elif syntax == 'jq':
        return jq.compile(filter_expr)
    else:
        raise ValueError(f"Unsupported filter syntax: {syntax}. Supported options are 'jsonpath' and 'jq'.")

def execute_filter_commands(attr_samples_dict: Dict[str, List[str]], attributes_list: List[str], filter_config: Dict[str, Any], num_lines: int = 100) -> Dict[str, int]:
    """
    Execute filter commands on sampled lines from documents and their attributes.
    Supports both JQ and JSONPath expressions based on the specified syntax.
    """
    print(f"Executing filter commands on attribute files, sampling {num_lines} lines for 1 file and their attributes.")

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
    print(f"Errors encountered: {results['errors']}\n")
    
    return results

def extract_attribute_names_from_filters(filters: List[str]) -> set:
    attribute_names = set()
    for filter_expr in filters:
        # Extract attribute names from JSONPath expressions
        matches = re.findall(r'@\.([a-zA-Z0-9_]+)', filter_expr)
        attribute_names.update(matches)
    return attribute_names

def extract_filter_attributes(filter_config: Dict[str, Any]) -> set:
    """Extract attribute names from filter expressions."""
    filter_attributes = set()
    for filter_type in ['include', 'exclude']:
        for filter_expr in filter_config.get(filter_type, []):
            # Extract attribute names from JSONPath expressions
            matches = re.findall(r'@\.([a-zA-Z_][a-zA-Z0-9_]*)', filter_expr)
            filter_attributes.update(matches)
    return filter_attributes

