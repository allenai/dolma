#! /usr/bin/env bash

datasets=(
    'c4_en'
    'dolma_books'
    'dolma_common-crawl'
    'dolma_pes2o'
    'dolma_reddit'
    'dolma_stack'
    'dolma_wiki'
    'ice'
    'm2d2_s2orc'
    'pile'
    'wikitext_103'
)

splits=(
    'test'
    'val'
)

for dataset in "${datasets[@]}"; do
    for split in "${splits[@]}"; do
        dolma tokens \
            --documents "s3://ai2-llm/eval-data/perplexity/v3_small/${dataset}/${split}" \
            --destination "s3://ai2-llm/eval-data/perplexity/v3_small_gptneox20b/${dataset}/${split}" \
            --tokenizer 'allenai/eleuther-ai-gpt-neox-20b-pii-special'
    done
done
