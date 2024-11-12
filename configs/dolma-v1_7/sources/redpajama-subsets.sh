
dolma tokens \
    --documents '${oc.env:HOME}/ai2-llm/pretraining-data/sources/redpajama/v1_decon_fix/documents/train/arxiv/*gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/redpajama_v1_decon_fix/arxiv/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 32 \
    --files_per_process 4 \
    --ring_size 4



dolma tokens \
    --documents '${oc.env:HOME}/ai2-llm/pretraining-data/sources/redpajama/v1_decon_fix/documents/train/stackexchange/*gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/redpajama_v1_decon_fix/stackexchange/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 32 \
    --files_per_process 4 \
    --ring_size 4
