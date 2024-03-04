set -ex

NUMBER_OF_CORES=188

dolma tokens \
    --documents '${oc.env:HOME}/ai2-llm/pretraining-data/sources/c4/v1_dd_ngram_doc_le030/documents/*.gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/c4/v1_dd_ngram_doc_le030/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '4_294_967_296' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"

dolma tokens \
    --documents '${oc.env:HOME}/ai2-llm/pretraining-data/sources/c4/v1_dd_ngram_docpara_le030/documents/*.gz' \
    --destination '${oc.env:HOME}/ai2-llm/preprocessed/c4/v1_dd_ngram_docpara_le030/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '4_294_967_296' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"

# dolma tokens \
#     --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_6-decontaminated/documents/cc_en_tail/*.gz' \
#     --destination 's3://ai2-llm/preprocessed/olmo-mix/v1_6-decontaminated/cc_en_tail/gpt-neox-olmo-dolma-v1_5' \
#     --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
#     --max_size '20_147_483_648' \
#     --seed 0 \
#     --tokenizer.eos_token_id 50279 \
#     --tokenizer.pad_token_id 1 \
#     --processes "${NUMBER_OF_CORES}"
