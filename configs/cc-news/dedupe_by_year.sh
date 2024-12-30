#! /usr/bin/env bash

base_dir="${HOME}/ai2-llm/pretraining-data/sources/cc-news/v1-resiliparse-year/documents"

# run years between 2016 and 2024
for year in {2016..2024}; do
    # Initialize an empty array to store document paths and a variable for total size
    documents=()
    size=0
    while IFS= read -r -d '' file; do
      documents+=("$file")
      size=$(expr $size + $(stat -c %s "$file"))
    done < <(find "${base_dir}/${year}" -type f \( -name "*.zst" -o -name "*.gz" -o -name "*.gzip" -o -name "*.json" -o -name "*.jsonl" \) -print0)

    # run deduplication
    echo "Running fuzzy dedupe for ${year} with ${size} bytes Bloom filter (files: ${#documents[@]})"

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
    attribute_name: dedupe_ngrams_20_1
    by_ngram:
      ngram_length: 20
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
  input: /tmp/cc_news_${year}_dedupe_ngrams_20_1/input
  output: /tmp/cc_news_${year}_dedupe_ngrams_20_1/output
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
    rm -rf "/tmp/cc_news_${year}*"

done
