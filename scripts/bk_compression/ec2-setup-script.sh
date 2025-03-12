#!/bin/bash
# Setup script for EC2 instance to process token stats

# Update and install basic packages
sudo yum update -y
sudo yum install -y python3 python3-pip git

# Create a directory for the project
mkdir -p ~/token-stats-project
cd ~/token-stats-project

# Clone your fork of the Dolma repository and checkout the bk_compression branch
git clone https://github.com/your-username/dolma.git
cd dolma
git checkout bk_compression

# Install dependencies
pip install transformers tqdm zstandard boto3

# Install the Dolma package in development mode so changes take effect immediately
pip install -e .

# Set up AWS credentials (if not using IAM roles)
# aws configure

# Your processor files should already be in the scripts/bk_compression directory
# through your git repository. Let's make sure they have the right paths.
cd scripts/bk_compression

# Update the config file with the correct S3 paths if needed
# You can use sed to replace placeholders if you want to automate this:
# sed -i 's|s3://your-bucket/your-dataset/|s3://actual-bucket/actual-path/|g' token_stats_config.yaml

echo "Setup complete! Now:"
echo "1. Verify S3 paths in scripts/bk_compression/token_stats_config.yaml"
echo "2. Run: dolma process scripts/bk_compression/token_stats_config.yaml"
