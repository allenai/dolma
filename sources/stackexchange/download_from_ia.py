import argparse
import boto3
import concurrent.futures
import os
import requests

from robust_downloader import download
from tqdm import tqdm
from urllib.parse import urlparse


# Fetch list of files from the Internet Archive API
def fetch_files_from_collection(collection_id):
    url = f"https://archive.org/metadata/{collection_id}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error fetching collection metadata: {response.status_code}")

    data = response.json()
    if 'files' not in data:
        raise Exception("No files found in the collection.")

    return [file['name'] for file in data['files']]



def download_files_in_parallel(files, collection_id, output_dir, max_parallel):
    def download_and_upload(source_url: str, output_dir: str = output_dir):
        s3_client = boto3.client('s3') if output_dir.startswith('s3://') else None
        source_file = source_url.split("/")[-1]
        local_path = os.path.join('/tmp', source_file)

        download(source_url, local_path)

        if s3_client:
            bucket, key = parse_s3_path(os.path.join(output_dir, source_url))
            s3_client.upload_file(local_path, bucket, key)
            os.remove(local_path)
        else:
            os.rename(local_path, os.path.join(output_dir, source_url))

        return source_url

    def parse_s3_path(s3_path):
        parsed = urlparse(s3_path)
        return parsed.netloc, parsed.path.lstrip('/')

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(download_and_upload, file) for file in files]

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Downloading files"):
            pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download files from the Internet Archive collection.")
    parser.add_argument(
        "-c", "--collection-id-or-url",
        type=str,
        required=True,
        help="The identifier of the collection to download from or the URL of the file to download.",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="The directory to download the files to; can be local or s3.",
        default=None
    )
    parser.add_argument(
        "-p", "--parallel-downloads",
        type=int,
        help="The number of parallel downloads to use.",
        default=1
    )
    opts = parser.parse_args()

    if opts.collection_id_or_url.startswith("https://archive.org/download/"):
        opts.collection_id_or_url = opts.collection_id_or_url.split("/")[-1]

    if opts.output_dir is None:
        opts.output_dir = f"s3://ai2-llm/pretraining-data/sources/{opts.collection_id}/raw"

    return opts



def main():
    opts = parse_arguments()
    files = fetch_files_from_collection(opts.collection_id_or_url)
    print(f"Found {len(files)} files to download.")
    download_files_in_parallel(files, opts.collection_id_or_url, opts.output_dir, opts.parallel_downloads)


if __name__ == "__main__":
    main()
