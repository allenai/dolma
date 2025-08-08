#!/usr/bin/env python3
import os
import csv
import io
import gzip
import boto3
import argparse
from urllib.parse import urlparse
from smart_open import open as sopen

s3_client = boto3.client('s3')
paginator = s3_client.get_paginator('list_objects_v2')

def parse_s3_uri(uri):
    """Split an S3 URI into (bucket, prefix)."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    return bucket, prefix

def list_s3_subdirs(bucket, prefix):
    """
    Gets immediate subdirectories in S3 path.
        - Used to get topic names (e.g. "adult_content") or vigintiles (e.g. "vigintile_00007").
    """
    subdirs = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip('/')+'/', Delimiter='/'):
        for cp in page.get('CommonPrefixes', []):
            # e.g. cp['Prefix'] == 'preprocessed/.../topic/vigintile/'
            name = cp['Prefix'].rstrip('/').split('/')[-1]
            subdirs.append(name)
    return subdirs

def list_s3_files(bucket, prefix, suffix='.csv.gz'):
    """List all object keys under prefix that are gzipped csvs."""
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip('/')+'/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith(suffix):
                keys.append(key)
    return keys

def build_mappings(bucket, topic_prefix, pretraining_data_path):
    """
    Scan every vigintile under a topic and return a dict:
      mappings[id] = (full_src, loc)
    """
    mappings = {}
    topic = topic_prefix.rstrip('/').split('/')[-1]  # e.g. "adult_content"
    vigintiles = list_s3_subdirs(bucket, topic_prefix)

    for vigintile in vigintiles:
        print(f"[build_mappings] processing vigintile: {vigintile}")
        vig_prefix = f"{topic_prefix.rstrip('/')}/{vigintile}/"

        for key in list_s3_files(bucket, vig_prefix):
            # key = "preprocessed/.../adult_content/vigintile_0007/file-000.csv.gz"

            with sopen(f"s3://{bucket}/{key}", 'rt') as fin:
                reader = csv.reader(fin)
                for start, end, doc_id, src, loc in reader:
                    # TODO: check if doc_id has been seen before?   

                    # build the full original S3 URI for each chunk, with vigintile
                    src_name = os.path.basename(src.strip())   # remove rest of relative path, e.g. keep "shard_00000156.jsonl.zst"

                    # need to reconstruct by hardcoding pretraining data path
                    full_src = f"{pretraining_data_path.rstrip('/')}/{topic}/{vigintile}/{src_name}"
                    
                    mappings[doc_id] = (full_src, loc)

    return mappings

def process_topic(pre_bucket, pre_prefix, post_bucket, post_prefix, topic, pretraining_data_path):
    """
    For a single topic:
      - Builds a mapping of doc_id → (original_src, original_loc) from pre-smush files.
      - Streams each post-smush CSV from S3 (auto-detecting gzip vs. plain text),
        replaces these (src, loc) with the originals
      - Writes a gzipped CSV locally (doesn't replace s3) with original (src, loc)
    """

    # original mappings containing vigintiles, from going through pre-smush files
    mappings = build_mappings(pre_bucket, f"{pre_prefix.rstrip('/')}/{topic}", pretraining_data_path)

    # go through csvs in current dolma path (post-smush csv.gz files that don't contain vigintiles)
    for key in list_s3_files(post_bucket, f"{post_prefix.rstrip('/')}/{topic}"):
        in_uri   = f"s3://{post_bucket}/{key}"
        local_fp = os.path.join("output", key)
        os.makedirs(os.path.dirname(local_fp), exist_ok=True)

        # read from S3, write gzipped locally
        with sopen(in_uri, 'rt', encoding='utf-8') as fin, \
             gzip.open(local_fp, 'wt', encoding='utf-8', newline='') as fout:

            reader = csv.reader(fin)
            writer = csv.writer(fout)

            for start, end, doc_id, src, loc in reader:
                og_src, og_loc = mappings[doc_id]
                assert og_loc == loc, f"loc mismatch for {doc_id}"
                assert os.path.basename(og_src) == os.path.basename(src), (
                    f"src basename mismatch for {doc_id}"
                )
                writer.writerow([start, end, doc_id, og_src, og_loc])


def main():
    parser = argparse.ArgumentParser(
        description="Fix src/loc on post-smush CSVs and mirror locally under ./output/")
    parser.add_argument(
        "--topic",
        help="Name of the single topic to process (e.g. adult_content).",
        required=True
    )
    args = parser.parse_args()
    
    pre_smush_path  = "s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer"
    post_smush_path = "s3://ai2-llm/preprocessed/dolma2-0625/v0.1/allenai/dolma2-tokenizer/all-dressed-snazzy2"
    original_json_data_path = "s3://ai2-llm/pretraining-data/sources/cc_all_dressed/all_dressed_v3/weborganizer_ft/dclm_plus2_vigintiles/data"

    pre_bucket,  pre_prefix  = parse_s3_uri(pre_smush_path)
    post_bucket, post_prefix = parse_s3_uri(post_smush_path)

    os.makedirs("output", exist_ok=True)

    # process one specified topic
    topic = args.topic
    print(f"Running on topic: {topic}")
    process_topic(pre_bucket, pre_prefix, post_bucket, post_prefix, topic, original_json_data_path)

if __name__ == "__main__":
    main()
