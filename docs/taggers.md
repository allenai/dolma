# Taggers

The tag command is used to run any of the built-in taggers on a set of documents. For example:

```shell
dolma tag \
    --experiment sample \
    --documents \
        's3://ai2-llm/pretraining-data/sources/common-crawl/test/v0/documents/**/*.json.gz' \
        's3://ai2-llm/pretraining-data/sources/common-crawl/test/v1/documents/*.json.gz' \
    --taggers random_number_v1 \
    --processes 2
```

This command will run the `random_number_v1` tagger on all documents in the specified S3 paths. The results will be written to the `s3://ai2-llm/pretraining-data/sources/common-crawl/test/v0/attributes/sample` and `s3://ai2-llm/pretraining-data/sources/common-crawl/test/v1/attributes/sample` paths.
