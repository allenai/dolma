# Dolma v1.5

Files is this directory are used to generate Dolma v1.5.

## Tagging

Tagging is largely the same as v1, but we report it here for completeness.

### C4

```bash
dolma tag
    --documents "s3://ai2-llm/pretraining-data/sources/c4/v0/documents/*/*.gz"
    --taggers pii_regex_with_counts_v2 \
    --processes 188
```
