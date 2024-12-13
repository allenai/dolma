from pprint import pprint

import boto3
import jinja2 as jinja
import re

from classifiers.scripts.corpus_eval.named_data_mixes import DATA_SOURCES, EXTRA_DATA_SOURCES


BUCKET_NAME = "ai2-llm"
MODEL_NAME = "regression-mmlu-20epochs"

mapping = {
    "stackexchange": "pretraining-data/sources/redpajama/v1_decon_fix/documents/train/stackexchange",
    "arxiv": "pretraining-data/sources/redpajama/v1_decon_fix/documents/train/arxiv",
    "algebraic_stack": "pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/algebraic-stack/train",
    "openwebmath": "pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/algebraic-stack/train",
    "cc_eli5_oh_top20p": "pretraining-data/sources/olmo-mix/eli5_oh_oh_top20p",
    "falcon_eli5_oh_top20p": "pretraining-data/sources/falcon-refinedweb/eli5_oh_oh_top20p/documents/",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "pretraining-data/sources/dclm/v1_pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p/documents/1t/",
    "DCLM-baseline": "'pretraining-data/sources/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_*_of_10/local-shard_*_of_10/shard_*_processed.jsonl.zstd'"
}


def path_exists(path: str) -> bool:
    response = boto3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=path)
    return "Contents" in response

output_paths = {}

if __name__ == "__main__":
    boto3 = boto3.client("s3")

    with open("template.sh") as f:
        template = jinja.Template(f.read())

    sources = {**DATA_SOURCES, **EXTRA_DATA_SOURCES}

    for source, source_paths in sources.items():
        if source in ["dclm_mmlu_top10"]:
            continue
        if source != "dclm_fw_top3":
            continue
        print("**")
        paths = [path.replace("preprocessed/", "pretraining-data/sources/") for path in source_paths]

        documents_paths_to_try = []
        if source in mapping:
            documents_paths_to_try.append(mapping[source])

        prefix = "/".join(paths[0].split("/")[:-1]).replace("/gpt-neox-olmo-dolma-v1_5", "")
        documents_paths_to_try.append(prefix)
        # try adding `documents` as second to last element
        documents_paths_to_try.append("/".join(prefix.split("/")[:-1] + ["documents", prefix.split("/")[-1]]))
        for documents_path in documents_paths_to_try:
            if "*" in documents_path or path_exists(documents_path):
                prefix = documents_path
                break
        else:
            print(f"Path {prefix} does *not* exist in s3 for source {source}.")

        found = False
        for suffix in ["documents", ""]:
            if "*" in prefix:
                files_wildcard = prefix
                found = True
                break
            if found:
                break
            files_in_path = boto3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=(prefix + "/" + suffix) if suffix else prefix)
            for obj in files_in_path["Contents"] if "Contents" in files_in_path else []:
                filename = obj["Key"]
                if "attributes" in filename:
                    continue
                if filename.endswith(".gz") or filename.endswith(".gzip") or filename.endswith(".zstd") or filename.endswith(".zst"):
                    # replace any number with a wildcard using regex
                    file_prefix, file_suffix = "/".join(filename.split("/")[:-1]), filename.split("/")[-1]
                    files_wildcard = file_prefix + "/" + re.sub(r"\d+", "*", file_suffix)

                    found = True
                    break
        if not found:
            print(f"No compressed files found in {prefix}.")
            continue

        files_wildcard = files_wildcard.replace("cc_en_head", "cc_en_*")
        files_wildcard = files_wildcard.replace("cc_en_tail", "cc_en_*")
        files_wildcard = files_wildcard.replace("cc_en_middle", "cc_en_*")

        print(f"Path {prefix} exists in s3 for source {source}. Using {files_wildcard} as wildcard.")

        output_paths[source] = f"s3://ai2-benb/corpus_scores/{MODEL_NAME}/{prefix.replace('pretraining-data/sources/', '')}"

        with open(f"beaker/{source}.sh", "w") as f:
            f.write(template.render(
                corpus_name=source,
                documents_path=f"s3://{BUCKET_NAME}/{files_wildcard}",
                max_rows=100000,
                num_nodes=1,
                num_gpus=2,
                model_name=MODEL_NAME,
                output_path=output_paths[source]
            ))

    for source in sources:
        print(f"bash classifiers/scripts/corpus_eval/beaker/{source}.sh")

    pprint(output_paths, indent=4)