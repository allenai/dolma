#!/usr/bin/env bash

# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"


set -ex

NUMBER_OF_CORES=188

aws s3 cp s3://ai2-llm/bloom-filters/paloma_documents_20240219.bin "${HOME}/paloma_documents.bin"
aws s3 cp s3://ai2-llm/bloom-filters/paloma_paragraphs_20240219.bin "${HOME}/paloma_paragraphs.bin"

dolma dedupe \
    --documents \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/algebraic-stack/train/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/algebraic-stack/validation/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/algebraic-stack/test/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/arxiv/train/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/arxiv/validation/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/arxiv/test/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/open-web-math/train/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/open-web-math/validation/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/open-web-math/test/*.gz' \
    --dedupe.name 'paloma_documents' \
    --dedupe.documents.attribute_name 'paloma_documents' \
    --dedupe.documents.key '$.text'  \
    --dedupe.skip_empty \
    --bloom_filter.file "${HOME}/paloma_documents.bin" \
    --bloom_filter.read_only \
    --bloom_filter.size_in_bytes 33557012 \
    --processes "${NUMBER_OF_CORES}"

dolma dedupe \
    --documents \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/algebraic-stack/train/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/algebraic-stack/validation/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/algebraic-stack/test/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/arxiv/train/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/arxiv/validation/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/arxiv/test/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/open-web-math/train/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/open-web-math/validation/*.gz' \
        's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0/documents/open-web-math/test/*.gz' \
    --dedupe.name 'paloma_paragraphs' \
    --dedupe.paragraphs.attribute_name 'paloma_paragraphs' \
    --dedupe.skip_empty \
    --bloom_filter.file "${HOME}/paloma_paragraphs.bin" \
    --bloom_filter.read_only \
    --bloom_filter.size_in_bytes 2099156 \
    --processes "${NUMBER_OF_CORES}"

dolma -c ${SCRIPT_DIR}/proof-pile-2.yaml mix

dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/algebraic-stack/train/*.gz' \
    --destination 's3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"

dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/arxiv/train/*.gz' \
    --destination 's3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/arxiv/train/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"

dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/proof-pile-2/v0_decontaminated/documents/open-web-math/train/*.gz' \
    --destination 's3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/open-web-math/train/gpt-neox-olmo-dolma-v1_5' \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes "${NUMBER_OF_CORES}"
