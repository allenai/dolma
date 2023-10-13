### The `dedupe` command

The dedupe command is used to deduplicate a set of documents at the attribute level using a bloom filter.

As an example let's consider the following command from the getting started guide:


```shell
dolma dedupe \
    --documents wikipedia/v0/documents/* \
    --dedupe.name dups \
    --dedupe.paragraphs.attribute_name bff_duplicate_paragraph_spans \
    --dedupe.skip_empty \
    --bloom_filter.file /tmp/deduper_bloom_filter.bin \
    --no-bloom_filter.read_only \
    --bloom_filter.estimated_doc_count '6_000_000' \
    --bloom_filter.desired_false_positive_rate '0.0001' \
    --processes 188
```

The above command will create an attribute directory called `dups` in `wikipedia/v0/attributes`. The `bff_duplicate_paragraph_spans` attribute will contain a list of duplicate paragraphs for each paragraph in the dataset.

```yaml
{
    "id": "page_1",
    "attributes": {
        "bff_duplicate_paragraph_spans": [
            [0, 500, 1] # paragraph starting at character 0 and ending at character 500 is a duplicate
        ]
    }
}
{
    "id": "page_2",
    "attributes": {
        "bff_duplicate_paragraph_spans": [] # no duplicates for page 2
    }
}
```
