#!/bin/bash

BUCKET="ai2-llm"
BASE_PREFIX="pretraining-data/sources/cccc/v0/documents/"

mapfile -t SNAPSHOTS < <(aws s3api list-objects-v2 --bucket ${BUCKET} --prefix ${BASE_PREFIX} --delimiter '/' --query 'CommonPrefixes[*].Prefix' --output json | jq -r '.[]')

for SNAPSHOT in "${SNAPSHOTS[@]}"; do
    ALL_FILES=$(aws s3api list-objects-v2 --bucket ${BUCKET} --prefix "${SNAPSHOT}" --query 'Contents[*].[Key,Size]' --output json)
    ALL_SIZES=$(echo $ALL_FILES | jq -r '.[] | select(.[0] | endswith(".zst")) | .[1]')

    sum=0
    count=0
    while IFS= read -r number; do
        sum=$((sum + number))
        count=$((count + 1))
    done <<< "${ALL_SIZES}"

    echo "Snapshot: $(basename ${SNAPSHOT}): $count files, total size: $sum"
done
