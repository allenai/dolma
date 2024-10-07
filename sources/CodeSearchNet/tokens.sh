#! /usr/bin/env bash

set -ex


dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/code_search_net/v0/documents/train/*/*.jsonl.gz' \
    --destination "${HOME}/ai2-llm/preprocessed/code_search_net/v0/train/allenai/dolma2-tokenizer" \
    --tokenizer.name_or_path 'allenai/dolma2-tokenizer' \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes 16 \
    --max_size 100_000_000 \
    --dtype 'uint32' \
    --sample_ring_prop
