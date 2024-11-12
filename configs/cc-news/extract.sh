#! /usr/bin/env bash

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

        documents="s3://ai2-russella/crawl-data/CC-NEWS/${year}/${month}/*.warc.gz"

        # run the extraction
        echo "Running extraction for ${year}-${month}"

        set -ex

        dolma warc \
            --documents ${documents} \
            --destination s3://ai2-llm/pretraining-data/sources/cc-news/v0-resiliparse/documents/${year}-${month} \
            --processes "$(expr $(nproc) - 4)" \
            --source_name cc-news_${year}-${month} \
            --linearizer resiliparse \
            --pre.taggers cc_re \
            --no-pre.skip \
            --no-store.html \
            --store.attr_spans 500 \
            --skip_duplicate_urls \
            --work_dir.input /tmp/cc-news/${year}-${month}/input \
            --work_dir.output /tmp/cc-news/${year}-${month}/output

        set +ex
    done
done
