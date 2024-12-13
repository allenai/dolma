#! /bin/bash

#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -c|--collection-id <collection_id> -d|--destination <destination> [-n|--num-processes <num_processes>] [-k|--num-chunks <num_chunks>]"
    echo "  -c, --collection-id   : The ID of the Internet Archive collection (required)"
    echo "  -d, --destination     : Location where to save each file from the collection (required)"
    echo "  -n, --num-processes   : Number of parallel downloads to use (default: 1)"
    echo "  -k, --num-chunks      : Number of chunks to split the collection into (default: 1)"
    exit 1
}

# Initialize variables
collection_id=""
destination=""
num_processes=1
num_chunks=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--collection-id)
            collection_id="$2"
            shift 2
            ;;
        -d|--destination)
            destination="$2"
            shift 2
            ;;
        -n|--num-processes)
            num_processes="$2"
            shift 2
            ;;
        -k|--num-chunks)
            num_chunks="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$collection_id" ]; then
    echo "Error: Collection ID is required"
    usage
fi

if [ -z "$destination" ]; then
    echo "Error: Destination is required"
    usage
fi

# Ensure num_processes is an integer greater than or equal to 1
if ! [[ "$num_processes" =~ ^[0-9]+$ ]] || [ "$num_processes" -lt 1 ]; then
    echo "Error: num_processes must be an integer greater than or equal to 1"
    usage
fi

# Ensure num_chunks is an integer greater than or equal to 1
if ! [[ "$num_chunks" =~ ^[0-9]+$ ]] || [ "$num_chunks" -lt 1 ]; then
    echo "Error: num_chunks must be an integer greater than or equal to 1"
    usage
fi


# Check if aria2c is available
if ! command -v aria2c &> /dev/null; then
    echo "Error: aria2c is not installed or not in the system PATH"
    exit 1
fi

# check if jq is available
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed or not in the system PATH"
    exit 1
fi

# Create a temporary file to store the download urls
temp_file=$(mktemp)

# Write items to the temporary file
curl -s "https://archive.org/metadata/$collection_id" | jq -r '{collection_id: .metadata.identifier, name: .files[].name} | select(.name | endswith(".7z")) | "https://archive.org/download/\(.collection_id)/\(.name)"' > "$temp_file"

# make destination directory if it doesn't exist
mkdir -p "$destination"

# Print the number of files to be downloaded
num_files=$(wc -l < "$temp_file")
echo "Downloading $num_files files"

if [ "$num_files" -eq 0 ]; then
    echo "No files to download"
    exit 1
fi

# Download each file in parallel
aria2c \
    --continue \
    --split ${num_chunks} \
    --max-connection-per-server ${num_chunks} \
    -k 1M \
    -j ${num_processes} \
    -i "$temp_file" \
    -d "$destination" \
    --show-console-readout=true \
    --summary-interval=5 \
    --console-log-level=notice

# Remove the temporary file
rm "$temp_file"
