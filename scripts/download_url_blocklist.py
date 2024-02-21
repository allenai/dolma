"""
Download the blocklists from the University of Toulouse (UT1) FTP server and upload them to
an S3 bucket.
More information about the blocklists can be found at: https://dsi.ut-capitole.fr/blacklists/

Author: Luca Soldaini (@soldni)
Email:  luca@soldaini.net
"""

import os
import tarfile
from datetime import datetime
import boto3
from ftplib import FTP
import tempfile

# FTP server details
ftp_url = "ftp.ut-capitole.fr"
ftp_path = "/pub/reseau/cache/squidguard_contrib/blacklists.tar.gz"

# S3 details
s3_bucket: str = "dolma-artifacts"
s3_folder: str = "blocklist_utp"  # This is optional, you can adjust as needed


def download_and_extract_ftp_file(ftp_url: str, ftp_path: str, extract_to: str) -> str:
    with FTP(ftp_url) as ftp:
        ftp.login()  # Use anonymous login
        local_file_path: str = os.path.join(extract_to, os.path.basename(ftp_path))
        with open(local_file_path, 'wb') as local_file:
            ftp.retrbinary(f'RETR {ftp_path}', local_file.write)
        extract_tar_gz(file_path=local_file_path, extract_to=extract_to)

    return local_file_path


def extract_tar_gz(file_path: str, extract_to: str) -> None:
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)


def upload_directory_to_s3(directory_path: str, bucket: str, folder: str) -> None:
    s3 = boto3.client('s3')
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
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Step 1 & 2: Download and extract
        local_file_path = download_and_extract_ftp_file(
            ftp_url=ftp_url, ftp_path=ftp_path, extract_to=tmpdirname
        )
        extracted_folder_path = local_file_path.rstrip(".tar.gz")
        s3_destination = os.path.join(s3_folder, new_folder_name)

        # Step 4: Upload to S3
        upload_directory_to_s3(
            directory_path=extracted_folder_path,
            bucket=s3_bucket,
            folder=s3_destination
        )

    print(f"Loc '{ftp_url}{ftp_path}' uploaded to bucket '{s3_bucket}/{s3_destination}' successfully.")


if __name__ == "__main__":
    main()
