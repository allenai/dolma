# Dolma v1.5 Eval set

We create the eval set by sampling documents in each subset. Some subsets already have an eval set (e.g. C4), so we use that. Also, for some subsets, creation of eval set was done using a different strategy (e.g., reddit; documented below), so we use other approaches.

For each subset, we aim for roughly 1M tokens


## CommonCrawl

```bash
python scripts/hash_sample.py \
    -s 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_head/*.gz' 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_middle/*.gz' 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_tail/*.gz' \
    -p 0.0000005 \
    -d s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/common-crawl \
    -n 188

```

Output:

```plain-text
{
  "debug": false,
  "destination": "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/common-crawl",
  "dryrun": false,
  "num_workers": 188,
  "probability": 5e-07,
  "source": [
    "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_head/*.gz",
    "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_middle/*.gz",
    "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_tail/*.gz"
  ]
}
Sampling with probability 5e-07 using MD5 suffixes ['ffffff', 'fffffe', 'fffffd', 'fffffc', 'fffffb', 'fffffa', 'fffff9', 'fffff8']
Found 2,878 files to process
uniseg_words: 1.00Mu [19:23, 860u/s]
extracted: 1.91ke [19:23, 1.64e/s]]
documents: 4.60Gd [19:23, 3.95Md/s]
files: 2.88kf [19:23, 2.47f/s]59u/s]
```


## PeS2o

```bash
python scripts/hash_sample.py \
    -s  s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2orc/split=valid/*/*.gz \
        s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2ag/split=valid/*/*.gz \
    -p 0.004 \
    -d s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/pes2o \
    -n 188
```

Output:
```plain-text
{
  "debug": false,
  "destination": "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/pes2o",
  "dryrun": false,
  "num_workers": 188,
  "probability": 0.004,
  "source": [
    "s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2orc/split=valid/*/*.gz",
    "s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2ag/split=valid/*/*.gz"
  ]
}
Sampling with probability 0.004 using MD5 suffixes ['ff']
Found 600 files to process
uniseg_words: 1.21Mu [00:06, 177ku/s]
extracted: 610e [00:06, 89.4e/s]s]
documents: 161kd [00:06, 23.6kd/s]
files: 600f [00:06, 87.9f/s] 77.4ku/s]
```

## Books

```bash
python scripts/hash_sample.py \
    -s 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/books/*.gz'\
    -p 0.00035\
    -d s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/books \
    -n 188
```

Output:

```plain-text
{
  "debug": false,
  "destination": "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/books",
  "dryrun": false,
  "num_workers": 188,
  "probability": 0.00038,
  "source": [
    "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/books/*.gz"
  ]
}
Sampling with probability 0.00038 using MD5 suffixes ['fff', 'ffe']
Found 3 files to process
uniseg_words: 1.73Mu [01:12, 23.7ku/s]
extracted: 30.0e [01:12, 2.42s/e]
documents: 52.1kd [01:12, 717d/s]
files: 3.00f [01:12, 24.2s/f]20.2ku/s]
```

## Wiki

```bash
python scripts/hash_sample.py \
    -s 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/wiki/*.gz'\
    -p 0.00038\
    -d s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/wiki \
    -n 188
```

Output:

```plain-text
{
  "debug": false,
  "destination": "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/wiki",
  "dryrun": false,
  "num_workers": 188,
  "probability": 0.00038,
  "source": [
    "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/wiki/*.gz"
  ]
}
Sampling with probability 0.00038 using MD5 suffixes ['fff', 'ffe']
Found 2 files to process
uniseg_words: 1.43Mu [01:58, 12.0ku/s]
extracted: 2.94ke [01:58, 24.7e/s]]
documents: 6.11Md [01:58, 51.4kd/s]
files: 2.00f [01:58, 59.4s/f]7.85ku/s]
```
