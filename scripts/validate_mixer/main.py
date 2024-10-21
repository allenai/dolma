import sys
import signal
import sys
import signal
import argparse
from validator import load_and_validate_config, validate_s3_paths_and_permissions, validate_stream_filters, validate_documents_and_attributes
from utils import keyboard_interrupt_handler, set_verbose
from env_handler import load_env_variables

def main(config_path, num_samples, verbose):
    # Register the keyboard interrupt handler
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    # Set verbose mode
    set_verbose(args.verbose)

    load_env_variables()
    config = load_and_validate_config(config_path)
    if config is None:
        print("Configuration loading or validation FAILED")

    if not validate_s3_paths_and_permissions(config):
        print("S3 path validation FAILED")
        # return

    if not validate_stream_filters(config):
        print("Filter validation FAILED.\n")
        return

    if not validate_documents_and_attributes(config, num_samples):
        print("Document and attribute validation FAILED")
        return

    print("Validation FINISHED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate mixer configuration")
    parser.add_argument("config_path", help="Path to the configuration file")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of file samples to validate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    main(args.config_path, args.num_samples, args.verbose)
