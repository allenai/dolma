pool=${POOL:-5shards}
mix=${MIX:-subsample}

subset_paths="$(s5cmd ls --show-fullpath s3://ai2-llm/pretraining-data/sources/dclm/refinedweb/dolma_reformat/pools/${pool}/${mix}/)"
echo "Found subsets:"
echo "$subset_paths"

for subset_path in $subset_paths; do
    subset_name=$(basename $subset_path)
    args=(
	    --destination s3://ai2-llm/preprocessed/dclm/ablations/${pool}-${mix}/${subset_name}
        --processes 190
        --documents $subset_path'*.jsonl.zst'
        --seed 3920
        --max_size 2_147_483_648 # 2GB
        --dtype uint32
        --tokenizer.name_or_path allenai/dolma2-tokenizer
        --tokenizer.eos_token_id 100257
        --tokenizer.pad_token_id 100277
        --no-tokenizer.segment_before_tokenization
        --tokenizer.encode_special_tokens 
    )
    
    dolma tokens ${args[@]} |& tee logs/tokens-${pool}-${mix}-${subset_name}.out
done
