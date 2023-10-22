BASE_S3="s3://ai2-llm/pretraining-data/sources/s2/v3-by-year/documents"

# for year in {1970..2023}; do
for year in {2016..2023}; do
    for split in train valid; do
        # increase number of processes after 2015
        if [[ $year -gt 2015 ]]; then
            processes=4
        else
            processes=1
        fi

        echo "Tokenizing $year $split"
        dolma tokens \
            --documents "$BASE_S3/split=$split/year=$year/*.gz" \
            --destination "s3://ai2-llm/preprocessed/pes2o/v2-by-year/$split/$year/gpt-neox-20b" \
            --max_size 4294967296 \
            --processes $processes \
            --tokenizer_name_or_path 'allenai/eleuther-ai-gpt-neox-20b-pii-special' &
    done
done
