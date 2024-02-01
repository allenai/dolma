# Deduplication

The `dedupe` command is used to deduplicate a set of documents at the attribute or paragraph level using a [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter).

Just like [taggers](taggers.md), `dolma dedupe` will create a set of attribute files, corresponding to the specified input document files. The attributes will identify whether the entire document is a duplicate (based on some key), or identify spans in the text that contain duplicate paragraphs.

Deduplication is done via an in-memory Bloom Filter, so there is a possibility of false positives.

Dropping any documents that are identified as duplicates, or deleting the duplicate paragraphs, can be done in a subsequent run of the mixer via `dolma mix`.

## Configuration

See sample config files [dedupe-by-url.json](examples/dedupe-by-url.json) and [dedupe-paragraphs.json](examples/dedupe-paragraphs.json).

## Parameters

The following parameters are supported either via CLI (e.g. `dolma dedupe --parameter.name value`) or via config file (e.g. `dolma -c config.json dedupe`, where `config.json` contains `{"parameter" {"name": "value"}}`):

|Parameter|Required?|Description|
|:---:|---|---|
|`documents`|Yes| One or more paths for input document files. Each accepts a single wildcard `*` character. Can be local, or an S3-compatible cloud path. |
|`work_dir.input`|No| Path to a local scratch directory where temporary input files can be placed. If not provided, Dolma will make one for you and delete it upon completion. |
|`work_dir.output`|No| Path to a local scratch directory where temporary output files can be placed. If not provided, Dolma will make one for you and delete it upon completion. |
|`dedupe.name`|No| Used to name output attribute files. One output file will be created for each input document file, where the key is obtained by substituting `documents` with `attributes/<name>`. If not provided, we will use either `dedupe.documents.attribute_name` or `dedupe.paragraphs.attribute_name`. |
|`dedupe.documents.key`| Mutually exclusive with `dedupe.paragraphs.attribute_name` | Use the json-path-specified field as the key for deduping. The value of the key must be a string. |
|`dedupe.documents.attribute_name`|Mutually exclusive with `dedupe.paragraphs.attribute_name`| Name of the attribute to set if the document is a duplicate. |
|`dedupe.paragraphs.attribute_name`|Mutually exclusive with `dedupe.documents.key` and `dedupe.documents.attribute_name` | Name of the attribute that will contain spans of duplicate paragraphs. Paragraphs are identified by splitting the `text` field by newline characters. |
|`dedupe.skip_empty`|No| If true, empty documents/paragraphs will be skipped.|
|`dedupe.min_length`|No| Minimum length of documents/paragraphs to be deduplicated. Defaults to 0.|
|`dedupe.min_words`|No| Minimum number of uniseg word units in documents/paragraphs to be deduplicated. Defaults to 0.|
|`bloom_filter.file`|Yes| Save the Bloom filter to this file after processing. If present at startup, the Bloom filter will be loaded from this file. |
|`bloom_filter.size_in_bytes`| Mutually exclusive with `bloom_filter.estimated_doc_count` and `bloom_filter.desired_false_positive_rate`| Used to set the size of the Bloom filter (in bytes). |
|`bloom_filter.read_only`|No| If true, do not write to the Bloom filter. Useful for things like deduping against a precomputed list of blocked attributes (e.g. URLs) or for decontamination against test data. |
|`bloom_filter.estimated_doc_count`| Mutually exclusive with `bloom_filter.size_in_bytes`; must be set in conjunction with `bloom_filter.desired_false_positive_rate` | Estimated number of documents to dedupe. Used to set the size of the Bloom filter. |
|`bloom_filter.desired_false_positive_rate`| Mutually exclusive with `bloom_filter.size_in_bytes`; must be set in conjunction with `bloom_filter.estimated_doc_count` | Desired false positive rate for the Bloom filter. Used to set the size of the Bloom filter. |
|`processes`|No| Number of processes to use for deduplication. One process is used by default. |
|`dryrun`|No| If true, only print the configuration and exit without running the deduper. |


If running with lots of parallelism, you might need to increase the number of open files allowed:

```shell
ulimit -n 65536
```
