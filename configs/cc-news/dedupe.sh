#! /usr/bin/env bash

# documents:
#   - s3://ai2-llm/pretraining-data/sources/c4/v0/documents/train/*.gz

# dedupe:
#   name: dedupe_para_ngrams_13_1
#   paragraphs:
#     attribute_name: dedupe_para_ngrams_13_1
#     by_ngram:
#       ngram_length: 13
#       stride: 1
#       overlap_threshold: 0.5
#   skip_empty: true

# bloom_filter:
#   file: ${oc.env:HOME}/c4_dedupe_para_ngrams_13_1.bin
#   read_only: false
#   # estimated doc count is obtained by counting number of words in paragraphs
#   # then dividing by 13 (ngram_length) and multiplying by 2 (for each ngram)
#   estimated_doc_count: 359_916_731_334
#   desired_false_positive_rate: 0.1

# processes: 188
# work_dir:
#   input: /tmp/c4_dedupe_para_ngrams_13_1/input
#   output: /tmp/c4_dedupe_para_ngrams_13_1/output

# run years between 2016 and 2024
for year in {2016..2024}; do
    # run months between 1 and 12
    for month in {1..12}; do
        # skip months after 7 if year is 2024
        if [ $year -eq 2024 ] && [ $month -gt 7 ]; then
            continue
        fi

        # skip months before 8 if year is 2016
        if [ $year -eq 2016 ] && [ $month -lt 8 ]; then
            continue
        fi

        # rename month to 2 digits
        month=$(printf "%02d" $month)

        documents="s3://ai2-llm/pretraining-data/sources/cc-news/v0-resiliparse/documents/${year}-${month}/*.zst"

        size=$(aws s3api list-objects --bucket ai2-llm --prefix "pretraining-data/sources/cc-news/v0-resiliparse/documents/${year}-${month}/" --output json --query "[sum(Contents[].Size)]" | jq '.[0]' -rc)

        # run deduplication
        echo "Running fuzzy dedupe for ${year}-${month} with ${size} bytes Bloom filter"

        set -ex

        dolma dedupe \
            --documents ${documents} \
            --dedupe.name dedupe_ngrams_13_1 \
            --dedupe.paragraphs.attribute_name dedupe_ngrams_13_1 \
            --dedupe.paragraphs.by_ngram.ngram_length 13 \
            --dedupe.paragraphs.by_ngram.stride 1 \
            --dedupe.paragraphs.by_ngram.overlap_threshold 0.5 \
            --dedupe.skip_empty \
            --bloom_filter.file "${HOME}/cc-news/dedupe_ngrams_13_1-${year}-${month}.bin" \
            --no-bloom_filter.read_only \
            --bloom_filter.estimated_doc_count $size \
            --bloom_filter.desired_false_positive_rate 0.01 \
            --processes "$(expr $(nproc) - 4)" \
            --work_dir.input /tmp/cc-news/dedupe_ngrams_13_1/${year}-${month}/input \
            --work_dir.output /tmp/cc-news/dedupe_ngrams_13_1/${year}-${month}/output

        set +ex
    done
done
