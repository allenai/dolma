#!/bin/bash
pool=${POOL:-"2shards-dedup"}
attributes=${ATTRIBUTES:-"dclm"}
attribute_name=${ATTRIBUTE_NAME:-"dclm__dclm_oh_eli5_log__score"}
output_name=${OUTPUT_NAME:-"dclm_baseline"}

f=${F:-0.4}
t=${T:-80}
b=${B:-4}
r=${R:-0.4}

mix=${output_name}_f${f}-t${t}-b${b}-r${r}

python scripts/make_quality_bin_config.py \
    --attributes "s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/${pool}/attributes/${attributes}/global-shard_03_of_10/*/*.jsonl.zstd" \
    --max-files-for-percentiles 1000 --num-processes-for-percentiles 100 -o stats/${mix}.jsonl \
    --config configs/aw_mix_${mix}.yaml \
    -f ${f} \
    -t ${t} \
    -b ${b} \
    -r ${r} \
    --documents "s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/${pool}/documents/global-shard_03_of_10/*/*.jsonl.zstd" \
    --attribute-name ${attribute_name} \
    --output "s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/${pool}/${mix}"


dolma -c configs/aw_mix_${mix}.yaml mix

token_args=(
    --destination s3://ai2-llm/preprocessed/dclm/ablations/${pool}-${mix}
    --processes 190
    --documents 's3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/'${pool}'/'${mix}'/*/*.jsonl.zst'
    --seed 3920
    --max_size 268_435_456 # 0.25GB (* 4 bytes)?
    --dtype uint32
    --tokenizer.name_or_path allenai/dolma2-tokenizer
    --tokenizer.eos_token_id 100257
    --tokenizer.pad_token_id 100277
    --no-tokenizer.segment_before_tokenization
    --tokenizer.encode_special_tokens 
)

dolma tokens ${token_args[@]} |& tee logs/tokens-${pool}-${mix}.out
rm -r /tmp/olmo*  # clean up local cache
