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
dolma -c examples/c4-like/tagger.yaml tag
```
