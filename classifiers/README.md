# Classifiers



## Examples

Run [NVIDIA's Deberta quality classifier](https://huggingface.co/nvidia/quality-classifier-deberta) on S3 data:

```bash
python -m dolma_classifiers.inference \
    -s 's3://ai2-llm/pretraining-data/sources/dclm/v0/documents/40b-split/*/*zstd' \
    -m HuggingFaceFW/fineweb-edu-classifier
```
