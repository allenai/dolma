#!/bin/bash
pool=${POOL:-"2shards-dedup"}
attributes=${ATTRIBUTES:-"dclm_chunked"}
output_name=${OUTPUT_NAME:-"length_bins"}

python scripts/quality_bin_lengths.py \
    --attributes "s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/${pool}/attributes/${attributes}/global-shard_03_of_10/*/*.jsonl.zstd" \
    --max-files-for-percentiles 1000 --num-processes-for-percentiles 100 \
    -o stats/${output_name}.jsonl \
    --percentiles 50 60 70 80 90