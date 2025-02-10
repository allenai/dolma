#! /usr/bin/env bash

set -ex


dolma tokens \
    --documents '/data/math-ai_AutoMathText/v0/documents/arxiv/*/*.jsonl.gz' \
    --destination "${HOME}/ai2-llm/preprocessed/math-ai_AutoMathText/v0/arxiv/allenai/dolma2-tokenizer" \
    --tokenizer.name_or_path 'allenai/dolma2-tokenizer' \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes 16 \
    --max_size 100_000_000 \
    --dtype 'uint32' \
    --sample_ring_prop

dolma tokens \
    --documents '/data/math-ai_AutoMathText/v0/documents/code/*/*.jsonl.gz' \
    --destination "${HOME}/ai2-llm/preprocessed/math-ai_AutoMathText/v0/code/allenai/dolma2-tokenizer" \
    --tokenizer.name_or_path 'allenai/dolma2-tokenizer' \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes 16 \
    --max_size 100_000_000 \
    --dtype 'uint32' \
    --sample_ring_prop


dolma tokens \
    --documents '/data/math-ai_AutoMathText/v0/documents/web/*.jsonl.gz' \
    --destination "${HOME}/ai2-llm/preprocessed/math-ai_AutoMathText/v0/web/allenai/dolma2-tokenizer" \
    --tokenizer.name_or_path 'allenai/dolma2-tokenizer' \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --no-tokenizer.segment_before_tokenization \
    --tokenizer.encode_special_tokens \
    --processes 16 \
    --max_size 100_000_000 \
    --dtype 'uint32' \
    --sample_ring_prop
