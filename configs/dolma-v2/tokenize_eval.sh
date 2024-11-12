tokenizer="allenai/dolma2-tokenizer"
path="s3://ai2-llm/eval-data/perplexity"
suffix="dolma2-tokenizer"


v2_eval=(
    "4chan"
    "c4_100_domains"
    "c4_en"
    "gab"
    "ice"
    "m2d2_s2orc"
    "m2d2_wiki"
    "manosphere"
    "mc4_en"
    "pile"
    "ptb"
    "twitterAEE"
    "wikitext_103"
)

v3_eval=(
    "c4_en"
    "dolma_books"
    "dolma_common-crawl"
    "dolma_pes2o"
    "dolma_reddit"
    "dolma_stack"
    "dolma_wiki"
    "ice"
    "m2d2_s2orc"
    "pile"
    "wikitext_103"
)

set -ex

for dataset in "${v2_eval[@]}"; do
    for split in "val" "test"; do
        dolma tokens \
            --tokenizer.name_or_path $tokenizer \
            --tokenizer.eos_token_id 100257 \
            --tokenizer.pad_token_id 100277 \
            --tokenizer.encode_special_tokens \
            --dtype uint32 \
            --destination "${path}/v2_small_${suffix}/${dataset}/${split}" \
            --documents "${path}/v2_small/${dataset}/${split}/*.gz" &
    done
done

for dataset in "${v3_eval[@]}"; do
    for split in "val" "test"; do
        dolma tokens \
            --tokenizer.name_or_path $tokenizer \
            --tokenizer.eos_token_id 100257 \
            --tokenizer.pad_token_id 100277 \
            --tokenizer.encode_special_tokens \
            --dtype uint32 \
            --destination "${path}/v3_small_${suffix}/${dataset}/${split}" \
            --documents "${path}/v3_small/${dataset}/${split}/*.gz" &
    done
done
