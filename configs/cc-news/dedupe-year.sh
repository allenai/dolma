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

    # Initialize an empty array to store document paths and a variable for total size
    documents=()
    size=0

    # Collect all month document paths into the array and accumulate size
    for month in {1..12}; do
        # Skip months after 7 if year is 2024
        if [ $year -eq 2024 ] && [ $month -gt 7 ]; then
            continue
        fi

        # Skip months before 8 if year is 2016
        if [ $year -eq 2016 ] && [ $month -lt 8 ]; then
            continue
        fi

        # Format month as 2 digits
        month=$(printf "%02d" $month)

        # Add the document path for this month to the array
        documents+=("s3://ai2-llm/pretraining-data/sources/cc-news/v0-resiliparse/documents/${year}-${month}/*.zst")

        # Get the size for this month and add it to the total size
        month_size=$(aws s3api list-objects --bucket ai2-llm --prefix "pretraining-data/sources/cc-news/v0-resiliparse/documents/${year}-${month}/" --output json --query "[sum(Contents[].Size)]" | jq '.[0]' -rc)
        size=$((size + month_size))
    done


    # run deduplication
    echo "Running fuzzy dedupe for ${year} with ${size} bytes Bloom filter"

    # Start the output
    document_linearized="documents:\n"

    # Loop through the array and append each element
    for doc in "${documents[@]}"; do
        document_linearized+="  - $doc\n"
    done

    config_yaml=$(cat <<EOF
${document_linearized}
dedupe:
  name: dedupe_by_year
  paragraphs:
    attribute_name: dedupe_ngrams_13_1
    by_ngram:
      ngram_length: 13
      stride: 1
      overlap_threshold: 0.5
      skip_short_paragraphs: true
  skip_empty: true

bloom_filter:
  file: /tmp/cc_news_${year}_dedupe_ngram.bin
  read_only: false
  estimated_doc_count: ${size}
  desired_false_positive_rate: 0.1

work_dir:
  input: /tmp/cc_news_${year}_dedupe_para_ngrams_13_1/input
  output: /tmp/cc_news_${year}_dedupe_para_ngrams_13_1/output
EOF
)


    # Create a temporary file for the YAML config
    temp_config_file=$(mktemp)

    # Write the YAML config to the temporary file
    printf "$config_yaml" > "$temp_config_file"


    set -ex
    # Run dolma with the temporary config file
    dolma -c "$temp_config_file" dedupe --processes $(expr $(nproc) - 4)
    set +ex

    # Remove the temporary file
    rm "$temp_config_file"

    done
done
