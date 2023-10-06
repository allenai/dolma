#! /usr/bin/env bash

datasets=(
    '4chan_meta_sep'
    'c4_100_domains'
    'c4_en'
    'dolma_100_subreddits'
    'dolma-v1_5'
    'falcon-refinedweb'
    'gab'
    'ice_fixed'
    'm2d2_s2orc_unsplit'
    'm2d2_wikipedia_unsplit'
    'manosphere_meta_sep'
    'mc4'
    'pile'
    'ptb'
    'redpajama'
    'twitterAAE_HELM_fixed'
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
