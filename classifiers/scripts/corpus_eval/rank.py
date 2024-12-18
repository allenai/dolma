import concurrent.futures
import itertools
import json

import pandas as pd
import smart_open
from smart_open import register_compressor
from smart_open.compression import _handle_zstd
from tqdm import tqdm
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
from cache_utils import cache
from rank_constants import mixtures, sources_sizes, paths
import wandb

register_compressor(".zstd", _handle_zstd)
api = wandb.Api()

metrics = [
    "eval/downstream/mmlu_humanities_var_len_norm",
    "eval/downstream/mmlu_other_var_len_norm",
    "eval/downstream/mmlu_stem_var_len_norm",
    "eval/downstream/mmlu_social_sciences_var_len_norm",
    "eval/downstream/hellaswag_len_norm"
]


def compute_proportions():
    # Calculate the total size for each mixture
    mixture_sizes = {mixture: sum(sources_sizes[source] for source in sources) for mixture, sources in mixtures.items()}

    # Calculate the proportion of each data source within each mixture
    mixture_proportions = {
        mixture: {source: sources_sizes[source] / mixture_sizes[mixture] for source in sources}
        for mixture, sources in mixtures.items()
    }

    return mixture_proportions


# Function to list all files in a given S3 prefix
def list_s3_files(bucket, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(f"s3://{bucket}/{obj['Key']}")
    return files


def get_scores():
    @cache()
    def process_path(name, path):
        bucket = path.split('/')[2]
        prefix = '/'.join(path.split('/')[3:])
        files = list_s3_files(bucket, prefix)

        values = []
        for file in files:
            with smart_open.open(file, 'rb') as fin:
                for line in itertools.islice(fin, None):
                    attributes_dict = json.loads(line)["attributes"]
                    assert len(attributes_dict) == 1
                    attributes = list(attributes_dict.values())[0][0]
                    value = attributes[2]
                    values.append(value)

        if len(values) == 0:
            print(f"No values found for {name} at {path}")
        return name, values

    @cache(fn_name="process_path", skip_load=True)
    def process_path_no_cache(name, path):
        return process_path(name, path)

    values = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_path_no_cache if name == "1" else process_path, name, path): name for name, path in paths.items()}
        # futures = {executor.submit(process_path, name, path): name for name, path in paths.items()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            name, result = future.result()
            values[name] = result

    return values


def compute_weighted_average_scores(scores, mixture_proportions):
    weighted_average_scores = {}

    for mixture, proportions in mixture_proportions.items():
        total_weighted_score = 0
        total_weight = 0

        for source, proportion in proportions.items():
            if source in scores:
                if len(scores[source]) == 0:
                    continue
                source_score = sum(scores[source]) / len(scores[source])
                if source_score < 0:
                    continue
                total_weighted_score += source_score * proportion
                total_weight += proportion

        if total_weight > 0:
            weighted_average_scores[mixture] = total_weighted_score / total_weight
        else:
            weighted_average_scores[mixture] = 0

    return weighted_average_scores


# @cache()
def get_run_performance(name):
    mapping = {
        "dolma17": "baseline"
    }

    name = mapping.get(name, name)

    run_name = f"{name}-1B-5xC"
    runs = api.runs("ai2-llm/olmo-ladder-benb", {"display_name": run_name}, order="-created_at")

    if len(runs) == 0:
        run_name = f"{name}-1B-5xC-2"
        runs = api.runs("ai2-llm/olmo-ladder-benb", {"display_name": run_name}, order="-created_at")
    if len(runs) == 0:
        print(f"No runs found for {run_name}")
        return None

    for run in runs:
        run_data = {"name": name}
        for metric in metrics:
            run_data[metric] = run.summary.get(metric)
        return run_data


def get_performance():
    # Create a list to store the data
    data = []

    # Iterate over each path and load the corresponding wandb run
    for name in mixtures.keys():
        run_data = get_run_performance(name)
        if run_data is not None:
            data.append(run_data)

    # Create a DataFrame from the collected data
    performance_df = pd.DataFrame(data)

    # add an average column
    performance_df['average'] = performance_df[metrics].mean(axis=1)

    return performance_df


def draw_scatter_plot(average_scores, performance_df):
    average_scores = pd.DataFrame({
        "name": list(average_scores.keys()),
        "average_score": list(average_scores.values())
    })

    # Merge the average scores with the performance DataFrame
    scatter_df = average_scores.merge(performance_df, on="name")

    scatter_df = scatter_df[scatter_df["average_score"] > 0]
    scatter_df = scatter_df[scatter_df["name"] != "redpajama"]

    # Print the table with name, performance, and score, sorted by score
    print(scatter_df[["name", "average", "average_score", "eval/downstream/hellaswag_len_norm"
                      ]].sort_values("average_score"))
    # and save to a file
    scatter_df[["name", "average", "average_score", "eval/downstream/hellaswag_len_norm"
                ]].sort_values("average_score").to_csv("scatter_data.csv", index=False)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="average_score", y="eval/downstream/hellaswag_len_norm", data=scatter_df, label="Average MMLU")
    sns.regplot(x="average_score", y="eval/downstream/hellaswag_len_norm", data=scatter_df, scatter=False, color='blue')
    # sns.scatterplot(x="average_score", y="eval/downstream/mmlu_other_var_len_norm", data=scatter_df, label="Other")
    plt.xlabel("Average Classifier Score")
    plt.ylabel("MMLU Accuracy")
    plt.legend()
    plt.savefig("scatter_plot.png")
    plt.show()


def draw_histograms(scores):
    # Create a DataFrame from the scores dictionary
    scores_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))

    # Plot histograms for each source
    scores_df.plot(kind='hist', bins=20, alpha=0.5, subplots=True, layout=(len(scores_df.columns) // 2, 2), figsize=(15, 25), sharex=True, density=True)
    plt.tight_layout()
    plt.savefig("histograms.png")
    plt.show()


if __name__ == "__main__":
    mixture_proportions = compute_proportions()
    scores = get_scores()
    average_scores = {source: (sum(values) / len(values)) if len(values) else None for source, values in scores.items()}

    weighted_average_scores = compute_weighted_average_scores(scores, mixture_proportions)
    print(sorted([(mixture, score) for mixture, score in weighted_average_scores.items() if score is not None], key=lambda x: x[1]))

    performance_df = get_performance()

    draw_scatter_plot(weighted_average_scores, performance_df)
    draw_histograms(scores)
