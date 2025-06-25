#!/bin/bash

script_dir=$(dirname $(realpath $0))

langs=(
    "C"
    "Cpp"
    "CSharp"
    "Go"
    "Java"
    "JavaScript"
    "Markdown"
    "PHP"
    "Python"
    "Ruby"
    "Rust"
    "Shell"
    "SQL"
    "Swift"
    "TypeScript"
)


set -ex


for lang in "${langs[@]}"; do
    uv run python -m dolma.tokenizer.reshard $script_dir/config/$lang.yaml
done
