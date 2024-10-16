import sys
import signal
import sys
import signal
from validator import load_and_validate_config, validate_s3_paths_and_permissions, validate_stream_filters, validate_documents_and_attributes
from utils import keyboard_interrupt_handler

def main(config_path, num_samples):
    # Register the keyboard interrupt handler
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    config = load_and_validate_config(config_path)
    if config is None:
        print("Configuration loading or validation failed")

    if not validate_s3_paths_and_permissions(config):
        print("S3 path validation failed")
        return

    if not validate_stream_filters(config):
        print("Filter validation failed.\n")
        return

    if not validate_documents_and_attributes(config, num_samples):
        print("Document and attribute validation failed")
        return

    print("Validation complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m validate_mixer <path_to_config_file> [number_of_file_samples, default=1]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(config_path, num_samples)