#!/bin/bash

script_dir=$(dirname $(realpath $0))

langs=(
    "adult_content"
    "art_and_design"
    "crime_and_law"
    "education_and_jobs"
    "electronics_and_hardware"
    "entertainment"
    "fashion_and_beauty"
    "finance_and_business"
    "food_and_dining"
    "games"
    "health"
    "history_and_geography"
    "home_and_hobbies"
    "industrial"
    "literature"
    "politics"
    "religion"
    "science_math_and_technology"
    "social_life"
    "software_development"
    "software"
    "sports_and_fitness"
    "transportation"
    "travel_and_tourism"
)


set -ex


for lang in "${langs[@]}"; do
    uv run python -m dolma.tokenizer.reshard $script_dir/../config/alex/$lang.yaml
done
