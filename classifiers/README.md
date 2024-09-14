# Dolma Classifiers

This package contains utilities to train and run classifiers for data filtering.

## Installation

From Dolma Toolkit root directory, run:

```bash

pip install -e classifiers
```

## Usage

To train a classifier, run:

```bash
python -m classifiers.train \
    --train-data-path=<path/to/train/data> \
    --test-data-path=<path/to/test/data> \
    --output-model-path=<path/to/output/model>
```

## Inference


To run this script with torchrun, use the following command:

```bash
torchrun --nproc_per_node=${NUM_GPUS} scripts/fineweb_classifier.py \
    --source-prefix s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/documents/*.zst \
    --output-prefix s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/attributes/fineweb-edu-classifier
    --batch-size 512 # 128 on A6000
```

Replace <num_gpus> with the number of GPUs you want to use.
