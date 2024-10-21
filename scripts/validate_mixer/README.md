# Dolma Mixer Configuration Validator

This script validates the configuration for the Dolma Mixer, ensuring that all necessary components are correctly set up before running the main process.

## Features

The validator performs the following checks:

1. Verifies the presence and format of required fields in the configuration.
2. Validates the syntax of the configuration file (YAML or JSON).
3. Checks for duplicate keys in the configuration.
4. Validates JQ or JSONPath expressions for syntax and compilation.
5. Verifies S3 path syntax and accessibility.
6. Confirms write permissions for output paths.
7. Checks the existence and accessibility of attribute files.
8. Samples a subset of files for detailed validation.
9. Ensures alignment between document and attribute files.
10. Validates the format and content of sampled files.
11. Executes JQ or JSONPath commands on sampled files.
12. Validates nested key existence in filter expressions.

## Usage

Run the validator using the following command:

```
python scripts/validate_mixer/main.py <path_to_config_file> [--num_samples <number>] [--verbose]
```

- `<path_to_config_file>`: Path to your Dolma Mixer configuration file (required)
- `--num_samples <number>`: (Optional) Number of file samples to validate (default: 1)
- `--verbose`: (Optional) Enable verbose output

## Output

The script provides detailed progress information and error messages for any validation failures, helping you troubleshoot configuration issues before running the main Dolma Mixer process.

## Keyboard Interrupt

The script handles keyboard interrupts (Ctrl+C) gracefully.

## Exit Status

The script will exit with a non-zero status if any validation step fails.