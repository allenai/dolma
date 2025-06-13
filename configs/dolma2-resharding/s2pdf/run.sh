#!/bin/bash

script_dir=$(dirname $(realpath $0))


set -ex


for config in $script_dir/config/*.yaml; do
    uv run python -m dolma.tokenizer.reshard $config
done
