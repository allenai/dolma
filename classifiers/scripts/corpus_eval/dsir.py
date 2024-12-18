import gzip
import json
import os
import argparse
import shutil
from pathlib import Path
from random import Random
from typing import List, Tuple
from urllib.parse import urlparse

import boto3
import fsspec
from concurrent.futures import ThreadPoolExecutor, as_completed
import zstandard as zstd


import numpy as np
from data_selection import HashedNgramDSIR
from datasets import load_dataset

PATHS = {
    # 'DCLM-baseline': "s3://ai2-llm/pretraining-data/sources/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_*_of_10/local-shard_*_of_10/shard_*_processed.jsonl.zstd",
    # 'algebraic_stack': 's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/algebraic-stack/train/algebraic-stack-train-*.json.gz',
    # 'all_red_pajama': 's3://ai2-llm/pretraining-data/sources/redpajama/v1/documents/split=test/dataset=arxiv/arxiv_*cd-*ee*-*e*-aa*b-*f*c*f.jsonl.gz',
    # 'arxiv': 's3://ai2-llm/pretraining-data/sources/redpajama/v1_decon_fix/documents/train/arxiv/arxiv-*.json.gz',
    'c4': 's3://ai2-llm/pretraining-data/sources/c4/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix/documents/c*-*.json.gz',
    # 'c4_debug': 's3://ai2-llm/pretraining-data/sources/c4/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix/documents/c*-*.json.gz',
    # 'cc_eli5_oh_top10p': 's3://ai2-llm/pretraining-data/sources/olmo-mix/eli5_oh_top10p/head/documents/cc-*.json.gz',
    # 'cc_eli5_oh_top20p': 's3://ai2-llm/pretraining-data/sources/olmo-mix/eli5_oh_oh_top20p/head/documents/cc_head-*.json.gz',
    # 'cc_news': 's3://ai2-llm/pretraining-data/sources/cc-news/v3/documents/cc_en_*/cc_news-*.json.gz',
    # 'cc_og_eli5_oh_top10p': 's3://ai2-llm/pretraining-data/sources/olmo-mix/og_eli5_oh_top10p/head/documents/cc_head-*.json.gz',
    # 'cc_tulu_qc_top10': 's3://ai2-llm/pretraining-data/sources/olmo-mix/tulu_qc_top10/head/documents/cc_head-*.json.gz',
    # 'dclm_ft7percentile_fw2': 's3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile_fw2/documents/0000/dclm-*.json.zst',
    # 'dclm_ft7percentile_fw3': 's3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile_fw3/documents/0000/dclm-*.json.zst',
    'dclm_fw_top10': 's3://ai2-llm/pretraining-data/sources/dclm/v0_fw_top10p/documents/1t/dclm-*.json.gzip',
    'dclm_fw_top3': 's3://ai2-llm/pretraining-data/sources/dclm/v0_fw_top3p/documents/1t/dclm-*.json.gzip',
    # 'dclm_mmlu_top3': 's3://ai2-llm/pretraining-data/sources/dclm/regression_synthetic_mmlu_3p/documents/full/dclm-*.json.gzip',
    'falcon': 's3://ai2-llm/pretraining-data/sources/falcon-refinedweb/v2-frac_005_100-qc_cc_multi_bin-paloma-rep-pii/documents/falcon-*.json.gz',
    # 'falcon_eli5_oh_top10p': 's3://ai2-llm/pretraining-data/sources/falcon-refinedweb/eli5_oh_top10p/documents/refinedweb-*.json.gz',
    # 'falcon_eli5_oh_top20p': 's3://ai2-llm/pretraining-data/sources/falcon-refinedweb/eli5_oh_oh_top20p/documents/refinedweb-*.json.gz',
    # 'falcon_og_eli5_oh_top10p': 's3://ai2-llm/pretraining-data/sources/falcon-refinedweb/og_eli5_oh_top10p/documents/refinedweb-*.json.gz',
    # 'falcon_tulu_qc_top10': 's3://ai2-llm/pretraining-data/sources/falcon-refinedweb/tulu_qc_top10/documents/refinedweb-*.json.gz',
    'fineweb_edu_dedup': 's3://ai2-llm/pretraining-data/sources/fineweb-edu-dedup/v0/documents/train-*-of-*.jsonl.gz',
    # 'gutenberg_books': 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_6-decontaminated/documents/books/books-*.json.gz',
    # 'megawika': 's3://ai2-llm/pretraining-data/sources/megawika/v1/documents/megawika-*.json.gz',
    # 'openwebmath': 's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/algebraic-stack/train/algebraic-stack-train-*.json.gz',
    # 'pes20_stem_papers': 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_6-decontaminated/documents/pes2o/pes*o-*.json.gz',
    # 'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p': 's3://ai2-llm/pretraining-data/sources/dclm/pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p/documents/1t/dclm-*.json.gzip',
    # 'pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p': 's3://ai2-llm/pretraining-data/sources/dclm/v1_pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p/documents/1t/dclm-*.json.zst',
    'prox_fineweb_pro': 's3://ai2-llm/pretraining-data/sources/prox_fineweb_pro/v0/documents/dump_*_*_*_*.jsonl.gz',
    'reddit': 's3://ai2-llm/pretraining-data/sources/reddit/v5-dedupe-pii-nsfw-toxic-fuzzydd-length/documents/reddit-*.json.gz',
    'regression_synthetic_20epochs_bs640_lf1_lre35_top10p': 's3://ai2-llm/pretraining-data/sources/dclm/regression_synthetic_20epochs_bs640_lf1_lre35_top10p/documents/1t/dclm-*.json.gzip',
    'regression_synthetic_20epochs_bs640_lf1_lre35_top20p': 's3://ai2-llm/pretraining-data/sources/dclm/regression_synthetic_20epochs_bs640_lf1_lre35_top20p/documents/1t/dclm-*.json.gzip',
    'stackexchange': 's3://ai2-llm/pretraining-data/sources/redpajama/v1_decon_fix/documents/train/stackexchange/stackexchange-*.json.gz',
    'starcoder': 's3://ai2-llm/pretraining-data/sources/starcoder/v0_decontaminated_doc_only/documents/starcoder-*.json.gz',
    # 'tulu': 's3://ai2-llm/pretraining-data/sources/tulu_flan/v2-decontaminated-60M-shots_all-upweight_1-dialog_false-sep_newline/documents/train/tulu_flan-*.json.gz',
    # 'web_instruct': 's3://ai2-llm/pretraining-data/sources/WebInstructSub/v0_decontaminated/documents/WebInstructSub-*.json.gz',
    # 'web_rest': 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_7-dd_ngram_dp_030-qc_cc_en_bin_001/documents/cc_en_*/cc_en_*-*.json.gz',
    'wikipedia_wikibooks': 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_6-decontaminated/documents/wiki/wiki-*.json.gz'
}

DOWNLOAD_PATH = 'sources'

class HashedNgramDSIRForEvaluating(HashedNgramDSIR):
    def importance_scores(self) -> List[Tuple[str, float]]:
        """Return all importance scores along with the corresponding text."""
        sharded_raw_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)

        # load log importance weights
        log_importance_weights_ls = [
            np.load(str(Path(self.log_importance_weights_dir) / f'{shard_params["overall_idx"]}.npy'), mmap_mode='r')
            for shard_params in sharded_raw_datasets
        ]
        concat_log_importance_weights = np.concatenate(log_importance_weights_ls, axis=0)

        # Load the text data
        text_data = []
        for shard_params in sharded_raw_datasets:
            dataset = self.raw_load_dataset_fn(shard_params['path'])
            for ex in dataset:
                text_data.append(self.raw_parse_example_fn(ex))

        # # Ensure the lengths match
        # assert len(text_data) == len(concat_log_importance_weights), "Mismatch between text data and importance scores"
        #
        # return list(zip(text_data, concat_log_importance_weights))

        return concat_log_importance_weights


def download_file(fs, path, source_name):
    # skip if uncompressed file exists
    uncompressed_path = get_uncompressed_path(os.path.join(DOWNLOAD_PATH, source_name, os.path.basename(path)))
    if os.path.exists(uncompressed_path):
        print(f"Skipping {path} as {uncompressed_path} exists")
        return uncompressed_path

    print(f"Downloading {path}")
    with fs.open(path, 'rb') as f:
        data = f.read()
    data_size = len(data)

    os.makedirs(os.path.join(DOWNLOAD_PATH, source_name), exist_ok=True)
    filename = os.path.basename(path)
    save_path = os.path.join(DOWNLOAD_PATH, source_name, filename)
    print(f"Saving to {save_path}")
    try:
        with open(save_path, 'wb') as local_file:
            local_file.write(data)
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return 0
    print(f"Downloaded {save_path} of size {data_size:,} bytes")

    # uncompress and delete compressed file
    uncompressed_path = uncompress_file(save_path)

    return uncompressed_path

def uncompress_file(file_path):
    if file_path.endswith('.gz') or file_path.endswith('.gzip'):
        uncompressed_path = get_uncompressed_path(file_path)
        try:
            with gzip.open(file_path, 'rb') as f_in:
                with open(uncompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except OSError:
            # If the file is not a valid gzip file, treat it as plain text
            shutil.copy(file_path, uncompressed_path)

        os.remove(file_path)
        return uncompressed_path
    elif file_path.endswith('.zst') or file_path.endswith('.zstd'):
        uncompressed_path = get_uncompressed_path(file_path)
        with open(file_path, 'rb') as f_in, open(uncompressed_path, 'wb') as f_out:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(f_in, f_out)
        os.remove(file_path)
        return uncompressed_path
    return file_path

def get_uncompressed_path(file_path):
    if file_path.endswith('.gz') or file_path.endswith('.gzip') or file_path.endswith('.zst') or file_path.endswith('.zstd'):
        return ".".join(file_path.split('.')[:-1])
    return file_path

def download_data(source_name, stop_after_mb, max_workers=5) -> List[str]:
    source_prefix = PATHS[source_name]
    fs = fsspec.get_filesystem_class((scheme := urlparse(source_prefix).scheme))()
    source_paths = [(f"{scheme}://{p}" if scheme else p) for p in fs.glob(source_prefix)]
    Random(0).shuffle(source_paths)

    sampled_paths = source_paths[:100]

    # Get file sizes with boto3
    s3 = boto3.client('s3')
    path_sizes = {}
    for path in sampled_paths:
        bucket = urlparse(path).netloc
        key = urlparse(path).path.lstrip('/')
        path_sizes[path] = s3.head_object(Bucket=bucket, Key=key)['ContentLength']

    stop_after_bytes = stop_after_mb * 1024 * 1024

    # Greedily select files until reaching max MB
    selected_paths = []
    total_size = 0
    for path, size in sorted(path_sizes.items(), key=lambda item: item[1]):
        if selected_paths and total_size + size > stop_after_bytes:
            break
        selected_paths.append(path)
        total_size += size

    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {}
        for path in selected_paths:
            future_to_path[executor.submit(download_file, fs, path, source_name)] = path

        for future in as_completed(future_to_path):
            try:
                downloaded_path = future.result()
                if downloaded_path:
                    downloaded_paths.append(downloaded_path)
            except Exception as exc:
                print(f"Error downloading {future_to_path[future]}: {exc}")

    return downloaded_paths

def score_documents(downloaded_paths, downstream_task):
    def target_load_dataset_fn(dataset):
        return load_dataset(dataset, streaming=True, split='train').take(10000)

    def target_parse_example_fn(ex):
        if downstream_task.lower() == "rowan/hellaswag":
            return ex['ctx'] + ' ' + ex['endings'][int(ex['label'])]

    def raw_load_dataset_fn(path, limit=None):
        cnt = 0
        with open(path, 'r') as f:
            for line in f:
                cnt += 1
                if len(line) > 0:
                    ex = json.loads(line)
                    ex["text"] = " ".join(ex["text"].split()[:600])
                    yield ex
                if limit is not None and cnt >= limit:
                    break

    def raw_parse_example_fn(ex):
        return ex["text"]

    max_rows_per_file = 50_000 // len(downloaded_paths)
    dsir = HashedNgramDSIRForEvaluating(
        raw_datasets=downloaded_paths,
        target_datasets=[downstream_task],
        cache_dir='.cache',
        raw_parse_example_fn=raw_parse_example_fn,
        raw_load_dataset_fn=lambda x: raw_load_dataset_fn(x, limit=max_rows_per_file),
        target_parse_example_fn=target_parse_example_fn,
        target_load_dataset_fn=target_load_dataset_fn,
        separate_targets=True,
        num_proc=6
    )
    dsir.fit_importance_estimator(num_tokens_to_fit=10_000_0)
    dsir.compute_importance_weights()
    scores = dsir.importance_scores()

    # find percentile 10
    if len(scores):
        return np.percentile(scores, 10)
    else:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source_path', type=str)
    parser.add_argument('--downstream_task', type=str, required=True)

    args = parser.parse_args()

    sources = [args.data_source_path] if args.data_source_path else PATHS.keys()
    scores = {}

    for source in sources:
        try:
            downloaded_paths = download_data(source, stop_after_mb=1000)
            score = score_documents(downloaded_paths, args.downstream_task)
            scores[source] = score
            print(json.dumps(scores, indent=2))
        except Exception as e:
            print(f"Error processing {source}: {e}")


if __name__ == "__main__":
    main()


# {
#   "c4": -1692.6298693330218,
#   "dclm_fw_top10": -1830.948619456947,
#   "dclm_fw_top3": -1878.0653081524283,
#   "falcon": -1766.3382691551592,
#   "fineweb_edu_dedup": -1795.1396116156536,
#   "prox_fineweb_pro": -1954.3925110772102,
#   "reddit": -1196.8498922172275,
#   "regression_synthetic_20epochs_bs640_lf1_lre35_top10p": -1951.8982932255471,
#   "regression_synthetic_20epochs_bs640_lf1_lre35_top20p": -1944.086880251118,
#   "stackexchange": -5832.6137423989485,
#   "starcoder": -2000.3068996123443,
#   "wikipedia_wikibooks": -1542.1675702719951
# }