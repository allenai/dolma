#!/usr/bin/env bash

set -ex

SCRIPT_PATH=$(realpath "$0")

bloom_filter_file=/tmp/oe-eval-data-dedupe_ngrams_8_1-train_dev_test.bin
remote_bloom_filter_file=s3://ai2-llm/bloom-filters/oe-eval-data-dedupe_ngrams_8_1-20241018-train_dev_test.bin

aws s3 cp $remote_bloom_filter_file $bloom_filter_file
size=331605257

dolma dedupe \
    --documents \
        "${HOME}/ai2-llm/pretraining-data/sources/dclm/v0_mmlu_web_minhash_dedup/documents/*.json.zst" \
    --dedupe.name dedupe_ngrams_8_1_all_train \
    --dedupe.paragraphs.attribute_name dedupe_ngrams_8_1_all_train \
    --dedupe.paragraphs.by_ngram.ngram_length 8 \
    --dedupe.paragraphs.by_ngram.skip_short_paragraphs \
    --dedupe.paragraphs.by_ngram.stride 1 \
    --dedupe.paragraphs.by_ngram.overlap_threshold 0 \
    --dedupe.skip_empty \
    --bloom_filter.file $bloom_filter_file \
    --bloom_filter.read_only \
    --bloom_filter.estimated_doc_count $size \
    --bloom_filter.desired_false_positive_rate 0.001 \
    --processes "$(expr $(nproc) - 4)"


dolma -c "$(dirname ${SCRIPT_PATH})/remove_all_train.yaml" mix --processes $(expr $(nproc) - 4)
