# Dolma v1.5 Eval set

We create the eval set by sampling documents in each subset. Some subsets already have an eval set (e.g. C4), so we use that. Also, for some subsets, creation of eval set was done using a different strategy (e.g., reddit; documented below), so we use other approaches.

For each subset, we aim for roughly 1M tokens


## CommonCrawl

```bash
python scripts/hash_sample.py \
    -s 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_head/*.gz' 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_middle/*.gz' 's3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_tail/*.gz' \
    -p 0.000001 \
    -d s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/common-crawl \
    -n 188

```

Output:
> Sampling with probability 1e-06 using MD5 suffixes ['fffff']
>
> Found 2,878 files to process
>
> uniseg_words: 2.04Mu [19:22, 1.76ku/s]
>
> extracted: 3.87ke [19:22, 3.33e/s]
>
> documents: 4.60Gd [19:22, 3.96Md/s]
>
> files: 2.88kf [19:22, 2.48f/s]55u/s


## PeS2o

```bash
python scripts/hash_sample.py \
    -s  s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2orc/split=valid/*/*.gz \
        s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2ag/split=valid/*/*.gz \
    -p 0.008 \
    -d s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5-eval/documents/pes2o \
    -n 188
```

Output:
> Sampling with probability 0.008 using MD5 suffixes ['ff', 'fe']
>
> Found 600 files to process
>
> uniseg_words: 2.57Mu [00:06, 376ku/s]
>
> extracted: 1.23ke [00:06, 179e/s]
>
> documents: 161kd [00:06, 23.6kd/s]
>
> files: 600f [00:06, 87.8f/s] 135ku/s
