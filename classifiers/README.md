# Classifiers



## Examples

Run [NVIDIA's Deberta quality classifier](https://huggingface.co/nvidia/quality-classifier-deberta) on S3 data:

```bash
python classifiers/inference.py \
    -s '/ai2-llm/pretraining-data/sources/dclm/v0/documents/40b-split/*/*zstd' \
    -m nvidia/quality-classifier-deberta
```
