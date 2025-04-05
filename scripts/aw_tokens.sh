pool=${POOL:-5shards}
mix=${MIX:-subsample}

args=(
    --destination s3://ai2-llm/preprocessed/dclm/ablations/${pool}-${mix}
    --processes 190
    --documents 's3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/'${pool}'/'${mix}'/*.jsonl.zst'
    --seed 3920
    --max_size 2_147_483_648 # 2GB
    --dtype uint32
    --tokenizer.name_or_path allenai/dolma2-tokenizer
    --tokenizer.eos_token_id 100257
    --tokenizer.pad_token_id 100277
    --no-tokenizer.segment_before_tokenization
    --tokenizer.encode_special_tokens 
)

dolma tokens ${args[@]} |& tee logs/tokens-${pool}-${mix}.out
