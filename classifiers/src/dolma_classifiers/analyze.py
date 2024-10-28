import argparse
import json
import os
import random
from urllib.parse import urlparse

import boto3
import numpy as np
from scipy.stats import percentileofscore
from smart_open.compression import (
    _handle_zstd,
    register_compressor,
)

from classifiers.src.dolma_classifiers.train import Classifier, DataConfig, ClassifierDataset

PERCENTILE_RANGES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9, 100]
MINIMUM_FINEWEB_EDU_SCORE = 3.5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a positive/negative classifier on a set of data sources"
    )
    parser.add_argument("-m", "--model-path", type=str, required=True, help="Classifier model name")
    parser.add_argument("--test-source", type=str, required=True, help="Test data source to score (no labels)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-source-instance-limit", type=int, default=100000, help="Number of instances to load from the test source")
    parser.add_argument("--fineweb-edu-path", type=str, default="s3://ai2-llm/pretraining-data/sources/fineweb-edu-dedup/v0/documents/train-00000-of-00234.jsonl.gz", help="Path to fineweb-edu data")
    parser.add_argument("--fineweb-edu-instance-limit", type=int, default=100000, help="Number of instances to load from the fineweb-edu data")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers to use")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--local-save-path", type=str, default="/tmp/qc_model", help="Local path to save output")
    parser.add_argument("--upload-to-s3", action="store_true", help="Upload model to S3")
    opts = parser.parse_args()

    return opts


def get_percentile_samples(rows, percentiles, n=15):
    # Create a dictionary to store examples for each percentile
    percentile_samples = {}

    # Loop through each percentile range and sample 10 examples
    for i in range(len(percentiles) - 1):
        lower_bound = percentiles[i]
        upper_bound = percentiles[i + 1]

        # Get the samples in this percentile range
        samples_in_percentile = [item for item in rows if lower_bound <= item["classifier_score"] <= upper_bound]

        # Randomly sample 10 examples, or take all if less than 10 are available
        sampled_examples = random.sample(samples_in_percentile, min(n, len(samples_in_percentile)))

        # Store the sampled examples
        percentile_label = f"{PERCENTILE_RANGES[i]}% - {PERCENTILE_RANGES[i + 1]}% Percentile"
        percentile_samples[percentile_label] = sampled_examples

    return percentile_samples


def analyze(test_rows, fineweb_edu_rows):
    # find percentiles of the test set scores
    classifier_test_scores = [item["classifier_score"] for item in test_rows]
    percentiles = np.percentile(classifier_test_scores, PERCENTILE_RANGES)

    # get the samples for each percentile to allow for manual inspection
    percentiles_samples = get_percentile_samples(test_rows, percentiles)

    # calculate the average percentile of the fineweb-edu scores
    fineweb_edu_scores = [item["score"] for item in fineweb_edu_rows]
    fineweb_edu_percentiles = [percentileofscore(classifier_test_scores, score) for score in fineweb_edu_scores]

    average_percentile = np.mean(fineweb_edu_percentiles)

    print(f"Average percentile of fineweb-edu scores: {average_percentile}")

    # examples where classifier gives > 3.5, and fine-web-edu gives < 2.5
    low_fineweb_edu_high_classifier = []

    for test_row in test_rows:
        if test_row["classifier_score"] > 3.5 and test_row["fineweb_edu_classifier_score"] < 2.5:
            low_fineweb_edu_high_classifier.append(test_row)
    low_fineweb_edu_high_classifier = random.sample(low_fineweb_edu_high_classifier, min(30, len(low_fineweb_edu_high_classifier)))

    return {
        "percentiles_samples": percentiles_samples,
        "average_percentile": average_percentile,
        "low_fineweb_edu_high_classifier": low_fineweb_edu_high_classifier
    }


def filter_fineweb_edu(row):
    return float(row["metadata"]["score"]) > MINIMUM_FINEWEB_EDU_SCORE


def main(args: argparse.Namespace):
    random.seed(args.seed)

    classifier = Classifier(load_model=args.model_path)

    # score the fineweb-edu data
    fineweb_edu_config = DataConfig(path=args.fineweb_edu_path, label=-1, sample=args.fineweb_edu_instance_limit,
                                    filter=filter_fineweb_edu)
    fineweb_edu_dataset = ClassifierDataset([fineweb_edu_config], workers=args.num_workers)
    fineweb_edu_scores = classifier.score(fineweb_edu_dataset)

    # score the test set
    test_config = DataConfig(path=args.test_source, label=-1, sample=args.test_source_instance_limit)
    test_dataset = ClassifierDataset([test_config], workers=args.num_workers)
    test_scores = classifier.score(test_dataset)

    # use fineweb_edu classifier to score the test set
    classifier = Classifier(load_model="HuggingFaceTB/fineweb-edu-classifier")
    test_scores_by_fineweb_edu_classifier = classifier.score(test_dataset)

    # merge the scores
    for test_row, fineweb_edu_classifier_row in zip(test_scores, test_scores_by_fineweb_edu_classifier):
        test_row["fineweb_edu_classifier_score"] = fineweb_edu_classifier_row["score"]
        test_row["classifier_score"] = test_row["score"]
        del test_row["score"]

    analysis = analyze(test_scores, fineweb_edu_scores)

    with open(os.path.join(args.local_save_path, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=4)

    with open(os.path.join(args.local_save_path, "fineweb_edu_scores.jsonl"), "w") as f:
        fineweb_edu_scores = sorted(fineweb_edu_scores, key=lambda x: x["score"])
        for row in fineweb_edu_scores:
            f.write(json.dumps(row) + "\n")

    with open(os.path.join(args.local_save_path, "test_scores.jsonl"), "w") as f:
        test_scores = sorted(test_scores, key=lambda x: x["score"])
        for row in test_scores:
            f.write(json.dumps(row) + "\n")

    if args.upload_to_s3:
        s3 = boto3.client("s3")

        parsed = urlparse(args.model_path)
        if parsed.scheme != "s3":
            raise ValueError("Model path must be an S3 path to upload to S3")

        for file in ["analysis.json", "fineweb_edu_scores.jsonl", "test_scores.jsonl"]:
            print(f"Uploading {file} to S3 bucket {parsed.netloc} at path {parsed.path}")
            s3.upload_file(os.path.join(args.local_save_path, file), parsed.netloc, os.path.join(parsed.path.lstrip("/"), "analysis", file))


if __name__ == "__main__":
    args = parse_args()

    # add additional extension for smart_open
    register_compressor(".zstd", _handle_zstd)

    main(args)
