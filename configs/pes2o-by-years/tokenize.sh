BASE_S3="s3://ai2-llm/pretraining-data/sources/s2/v3-by-year/documents"

for year in {1970..2023}; do
    for split in train valid; do
        echo "Tokenizing $year $split"
        dolma tokens \
            --documents "$BASE_S3/split=$split/year=$year/*.gz" \
            --destination "s3://ai2-llm/preprocessed/pes2o/v2-by-year/$split/$year/gpt-neox-20b" \
            --max_size 34359738368 \
            --tokenizer_name_or_path 'allenai/eleuther-ai-gpt-neox-20b-pii-special'
    done
    exit
done
