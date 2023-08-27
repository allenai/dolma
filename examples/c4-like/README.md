# C4-like processing

**Step 0**: (Optional) Convert data to dolma format. For example, we reformat the [trafilatura](https://trafilatura.readthedocs.io/en/latest/) output to dolma format.


```bash
python examples/c4-like/reformat_trafilatura.py \
    src='s3://ai2-llm/experimental/dcnlp_beta_ai2/raw/cc_trafilatura_beta/crawl-data/*/segments/*/warc/*.jsonl' \
    dst='s3://ai2-llm/experimental/dcnlp_beta_ai2/v0/documents' \
    proc=16 \
    debug=False
```

**Step 1**: Tag with fasttext language id and c4 rules.

```bash
time dolma -c examples/c4-like/tagger.yaml tag
```

Timing on a `c6a.4xlarge` instance (8 cores, 16 threads, 32 GB memory, gp3 volume, 16000 IOPS, 125 MB/s throughput):

| **Processes** | **Seconds** |
|:-------------:|:-----------:|
|       1       |     ???     |
|       4       |     ???     |
|       8       |     ???     |
|      16       |     ???     |


**Step 2**: Run mixer to generate the final dataset.

```bash
time dolma -c examples/c4-like/mixer.yaml mix
```

Timing on a `c6a.4xlarge` instance (8 cores, 16 threads, 32 GB memory, gp3 volume, 16000 IOPS, 125 MB/s throughput):

| **Processes** | **Seconds** |
|:-------------:|:-----------:|
|       1       |     ???     |
|       4       |     ???     |
|       8       |     769     |
|      16       |     ???     |
