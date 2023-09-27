# Dolma v1.5

Files is this directory are used to generate Dolma v1.5.

## Tagging

Tagging is largely the same as v1, but we report it here for completeness.

### C4

```bash
dolma tag --documents 's3://ai2-llm/pretraining-data/sources/c4/v0/documents/train/*.gz' --taggers pii_regex_with_counts_v2 --processes 188
```

### Common Crawl



## Filtering


## Sampling of CC


```bash
dolma tag --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/cc_en_head//*.gz' 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/cc_en_tail/*.gz' --taggers random_number_v1 --processes 188
```
