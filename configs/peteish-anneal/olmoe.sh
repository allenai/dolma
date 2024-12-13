collections=(
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/dclm/*/*.json.zst"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/flan/*.json.gz"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/codesearchnet-owmfilter/*/*.jsonl.gz"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/dolmino_math_synth/basic_math/*TRAIN.jsonl"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/dolmino_math_synth/gsm8k-synth/resample_v1_6x/*.jsonl"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/dolmino_math_synth/gsm_mind/*/*.jsonl"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/gsm8k/*/train/*.jsonl.zst"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/mathcoder2-synthmath/ajibawa-2023/*.jsonl"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/mathcoder2-synthmath/m-a-p_Matrix/*/*.jsonl"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/metamath-owmfilter/*.jsonl.gz"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/tinyGSM-MIND/*/*.jsonl.gz"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/math/tulu_math/*/*.jsonl"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/pes2o/*.json.gz"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/stackexchange/*.json.gz"
    "${HOME}/ai2-llm/pretraining-data/sources/dolmino-mix-1124/documents/wiki/*.json.gz"
)

for path in "${collections[@]}"; do
    name=$(echo "${path}" | sed -E 's|.*/documents/([^*]+).*|\1|' | sed 's:^/::; s:/$::')
    destination="${HOME}/ai2-llm/preprocessed/dolmino-mix-1124/allenai/gpt-neox-olmo-dolma-v1_5/${name}"

    echo "Tokenizing $path to $destination"
    echo "Number of files: $(ls -1 $path 2>/dev/null | wc -l)"

    if [[ "$name" == *"dclm"* ]]; then
        processes=$(($(nproc) - 4))
    else
        processes=20
    fi

    set -ex
    dolma tokens \
        --documents "${path}" \
        --destination $destination \
        --no-tokenizer.segment_before_tokenization \
        --tokenizer.name_or_path "allenai/gpt-neox-olmo-dolma-v1_5" \
        --tokenizer.eos_token_id 50279 \
        --tokenizer.pad_token_id 1 \
	--tokenizer.encode_special_tokens \
        --processes ${processes} \
        --seed 3920 \
        --max_size 1073741824 \
        --sample_ring_prop \
        --dtype uint32
    set +ex
done
