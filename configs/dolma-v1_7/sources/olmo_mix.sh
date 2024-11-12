set -ex


export DOLMA_SUBSET=${1}

# check if the subset is provided
if [ -z "${DOLMA_SUBSET}" ]; then
    echo "Please provide the subset to process"
    exit 1
fi

# reseerve 4 cores for the system
export NUMBER_OF_CORES=$(($(nproc) - 4))

dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/${oc.env:DOLMA_SUBSET}/documents/cc_en_head/*.gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/olmo-mix/${oc.env:DOLMA_SUBSET}/cc_en_head/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '20_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"

dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/${oc.env:DOLMA_SUBSET}/documents/cc_en_middle/*.gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/olmo-mix/${oc.env:DOLMA_SUBSET}/cc_en_middle/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '20_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"


dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/${oc.env:DOLMA_SUBSET}/documents/cc_en_tail/*.gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/olmo-mix/${oc.env:DOLMA_SUBSET}/cc_en_tail/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '20_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"
