# C4-like processing

Please run this experiment from mainline.

## Step 0

Convert data to dolma format. For example, we reformat the [trafilatura](https://trafilatura.readthedocs.io/en/latest/) output to dolma format.


```bash
python examples/c4-like/reformat_trafilatura.py \
    src='s3://ai2-llm/experimental/dcnlp_beta_ai2/raw/cc_trafilatura_beta/crawl-data/*/segments/*/warc/*.jsonl' \
    dst='s3://ai2-llm/experimental/dcnlp_beta_ai2/v0/documents' \
    proc=16 \
    debug=False
```

## Step 1

Tag with fasttext language id and c4 rules.

```bash
time dolma -c examples/c4-like/tagger.yaml tag
```

Timing on a `c6a.4xlarge` instance (8 cores/16 threads, 32 GB memory, gp3 volume/16000 IOPS/125 MB/s throughput):

| **Processes** | **Seconds** |
|:-------------:|:-----------:|
|       1       |     N/A     |
|       8       |     655     |
|      16       |     493     |

## Step 2

Run mixer to generate the final dataset.

```bash
time dolma -c examples/c4-like/mixer.yaml mix
```

Timing on a `c6a.4xlarge` instance (8 cores/16 threads, 32 GB memory, gp3 volume/16000 IOPS/125 MB/s throughput):

| **Processes** | **Seconds** |
|:-------------:|:-----------:|
|       1       |     N/A     |
|       8       |     631     |
|      16       |     491     |

### Observations

Ok I'm done with the dolma C4 baselines. on a c6a.4xlarge instance (using 8 processes), these are the timing I get:

- Tagging with language id (fasttext) and C4 heuristics: 655s total, or 52s per jsonl per core
- Filtering data: 631s total, or 50s per jsonl per core
- Total time: 1286s, or 102s per jsonl per core. Projected CPU hours: (1286s * 8 cores / 23.92GB * 1024) / 3600 = 122 CPU Hours per TB.

A couple of caveats:

- I had to reformat the data to conform to dolma's format (create an ID field, rename "content" to "text"); I also gzip'ed files (dolma taggers can work with either regular or gzip'ed, although mixer expects gzip)
- I am not using the C4 first, lang id after heuristics you all are using bc it's not automatically supported on dolma (you'd have to do it manually)
- C4 rules implementation can still be implemented a bit. so far, the naughty words lookup step is the most expensive. Maybe a 15/20% speedup is possible.
- The mixer requires fully downloading each file before it can be filtered. I could improve the rust code to allow streaming (taggers already stream)
