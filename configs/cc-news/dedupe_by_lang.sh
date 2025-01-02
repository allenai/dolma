#! /usr/bin/env bash

base_dir="${HOME}/ai2-llm/pretraining-data/sources/cc-news/v2-resiliparse-year_dedup-lang/documents"

langs=($(du -sh "${base_dir}"/* 2>/dev/null | sort -hr | awk '{print $2}' | xargs -n1 basename))

for lang in "${langs[@]}"; do
    documents=()
    size=0
    while IFS= read -r -d '' file; do
      documents+=("$file")
      size=$(expr $size + $(stat -c %s "$file"))
    done < <(find "${base_dir}/${lang}" -type f \( -name "*.zst" -o -name "*.gz" -o -name "*.gzip" -o -name "*.json" -o -name "*.jsonl" \) -print0)

    # run deduplication
    echo "Running fuzzy dedupe for ${lang} with ${size} bytes Bloom filter (files: ${#documents[@]})"

    # Start the output
    document_linearized="documents:\n"

    # Loop through the array and append each element
    for doc in "${documents[@]}"; do
        document_linearized+="  - $doc\n"
    done

    config_yaml=$(cat <<EOF
${document_linearized}
dedupe:
  name: dedupe_by_lang
  paragraphs:
    attribute_name: dedupe_ngrams_13_1
    by_ngram:
      ngram_length: 13
      stride: 1
      overlap_threshold: 0.5
      skip_short_paragraphs: true
  skip_empty: true

bloom_filter:
  file: /tmp/cc_news_${lang}_dedupe_ngram.bin
  read_only: false
  estimated_doc_count: ${size}
  desired_false_positive_rate: 0.1

work_dir:
  input: /tmp/cc_news_${lang}_dedupe_ngrams_13_1/input
  output: /tmp/cc_news_${lang}_dedupe_ngrams_13_1/output
EOF
)
    # Set the number of processes to the minimum of the number of documents
    # and <number of available processors - 4> to leave some room for other processes
    processes=$(( $(expr $(nproc) - 4) < ${#documents[@]} ? $(expr $(nproc) - 4) : ${#documents[@]} ))

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
    rm -rf "/tmp/cc_news_${lang}*"

done
