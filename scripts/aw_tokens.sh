pool=${POOL:-2shards-dedup}
mix=${MIX:-dclm_baseline_f0.4-t60-b1-r1.0}

args=(
    --destination s3://ai2-llm/preprocessed/dclm/ablations/${pool}-${mix}
    --processes 190
    --documents 's3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/'${pool}'/'${mix}'/*/*.jsonl.zst'
    --seed 3920
    --max_size 268_435_456 # 2GB (* 4 bytes)?
    --dtype uint32
    --tokenizer.name_or_path allenai/dolma2-tokenizer
    --tokenizer.eos_token_id 100257
    --tokenizer.pad_token_id 100277
    --no-tokenizer.segment_before_tokenization
    --tokenizer.encode_special_tokens 
)

dolma tokens ${args[@]} |& tee logs/tokens-${pool}-${mix}.out
rm -r /tmp/olmo*  # clean up local cache
