#! /usr/bin/env bash

set -ex


dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/tulu_flan/v1-FULLDECON-60M-shots_all-upweight_1-dialog_false-sep_rulebased/documents/*.json.gz' \
    --destination "${HOME}/ai2-llm/preprocessed/tulu_flan/v1-FULLDECON-60M-shots_all-upweight_1-dialog_false-sep_rulebased/allenai/dolma2-tokenizer" \
    --tokenizer.name_or_path 'allenai/dolma2-tokenizer' \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --ring_size 8 \
    --processes 92 \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32'
