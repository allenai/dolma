# Dolma Classifiers


## Getting Started

From root directory, install the package:

```bash
pip install -e classifiers
```

## Examples

Run [Huggingface FineWeb classifier](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier) on S3 data:

```bash
python -m dolma_classifiers.inference \
    -s 's3://ai2-llm/pretraining-data/sources/dclm/v0/documents/40b-split/*/*zstd' \
    -m HuggingFaceFW/fineweb-edu-classifier
```


<!-- Run [NVIDIA's Deberta quality classifier](https://huggingface.co/nvidia/quality-classifier-deberta) on S3 data:
 -->
