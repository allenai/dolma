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
dolma tag --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/cc_en_head/*.gz' 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/cc_en_tail/*.gz' --taggers random_number_v1 --processes 188
```

dolma tag --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/cc_en_middle/*.gz' --taggers random_number_v1 --processes 188

## Tokenization

```bash
python -m dolma.tokenizer --sources 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/*/*' --destination $HOME/preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special --num-writers 188 --max-size 17179869184
```

```bash
dolma tokens \
    --documents 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/books/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/c4/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-sample/documents/cc_en_head/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-sample/documents/cc_en_middle/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-sample/documents/cc_en_tail/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/pes2o/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/reddit/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/stack/*' \
        's3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5/documents/wiki/*' \
    --tokenizer_name_or_path 'allenai/gpt-neox-20b-pii-special' \
    --destination $HOME/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special \
    --processes 188 \
    --ring_size 8 \
    --batch_size 10000 \
    --max_size 5368709120
```
