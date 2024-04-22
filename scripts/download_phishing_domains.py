"""
Download the blocklist for phishing domain from https://github.com/mitchellkrogza/Phishing.Database
and upload it to an S3 bucket.

Author: Luca Soldaini (@soldni)
Email:  luca@soldaini.net
"""

import os
from pathlib import Path
import shutil
from datetime import datetime
import boto3
from cached_path import cached_path
import tempfile

# source location
phishing_domains = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ALL-phishing-domains.tar.gz"    # noqa: E501
phishing_links = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ALL-phishing-links.tar.gz"      # noqa: E501

# S3 details
s3_bucket: str = "dolma-artifacts"
s3_folder: str = "blocklist_phishing_db"


def upload_directory_to_s3(directory_path: str, bucket: str, folder: str) -> None:
    s3 = boto3.client("s3")
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path: str = os.path.join(root, file)
            s3_path: str = os.path.join(folder, os.path.relpath(file_path, directory_path))
            s3.upload_file(file_path, bucket, s3_path)


def main():
    # Generate today's date in the format YYYYMMDD
    today = datetime.now().strftime("%Y%m%d")
    new_folder_name = f"{s3_folder}-{today}"

    # Temporary directory to store downloaded and extracted files
    with tempfile.TemporaryDirectory() as tmp_dir:
        (cache_dir := (tmp_path := Path(tmp_dir)) / 'tmp').mkdir(exist_ok=True)
        (local_dir := tmp_path / new_folder_name).mkdir(exist_ok=True)

        domains_loc = cached_path(phishing_domains, cache_dir=cache_dir)
        shutil.move(domains_loc, local_dir / "domains.tar.gz")

        links_loc = cached_path(phishing_links, cache_dir=cache_dir)
        shutil.move(links_loc, local_dir / "links.tar.gz")

        s3_destination = os.path.join(s3_folder, new_folder_name)
        upload_directory_to_s3(directory_path=str(local_dir), bucket=s3_bucket, folder=s3_destination)

    print(f"Uploaded to bucket '{s3_bucket}' w prefix '{s3_destination}' successfully.")


if __name__ == "__main__":
    main()
