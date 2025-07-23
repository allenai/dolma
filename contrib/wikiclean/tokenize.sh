#!/bin/bash

path="/mnt/raid0/wikiclean-rewritten"
output_path="/mnt/raid0/wikiclean-rewritten-tokenized"
tokenizer_name="allenai/dolma2-tokenizer"

echo "Tokenizing $path with $num_processes processes"

uv run dolma tokens \
    --documents "$path" \
    --destination "$output_path" \
    --tokenizer.name_or_path "$tokenizer_name" \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --ring_size 2 \
    --processes 128 \
    --max_size 4_000_000_000 \
    --sample_ring_prop \
    --dtype 'uint32'
