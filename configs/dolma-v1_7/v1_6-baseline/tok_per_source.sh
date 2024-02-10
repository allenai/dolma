#!/bin/env bash

sources=(
    "books,1"
    "c4,5"
    "cc_en_head,10"
    "cc_en_middle,10"
    "cc_en_tail,10"
    "pes2o,3"
    "reddit,2"
    "stack,5"
    "wiki,1"
)

set -x

for i in "${!sources[@]}"; do
    # split source and number of processes
    source=$(echo "${sources[$i]}" | cut -d',' -f1)
    processes=$(echo "${sources[$i]}" | cut -d',' -f2)

    dolma tokens \
        --destination "s3://ai2-llm/preprocessed/olmo-mix/v1_6-300G-decon/gpt-neox-olmo-dolma-v1_6_persource/${source}" \
        --documents "${HOME}/ai2-llm/pretraining-data/sources/olmo-mix/v1_6-300G-decon/documents/${source}" \
        --tokenizer.name_or_path "allenai/gpt-neox-olmo-dolma-v1_5" \
        --tokenizer.eos_token_id 50279 \
        --tokenizer.pad_token_id 1 \
        --processes  ${processes} \
        --seed 3920 \
        --max_size "21_474_836_480"
done
