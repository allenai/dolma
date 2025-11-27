# Deduplication

The `dedupe` command is used to deduplicate a set of documents at the attribute or paragraph level using a [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter).

Just like [taggers](taggers.md), `dolma dedupe` will create a set of attribute files, corresponding to the specified input document files. The attributes will identify whether the entire document is a duplicate (based on some key), or identify spans in the text that contain duplicate paragraphs.

Deduplication is done via an in-memory Bloom Filter, so there is a possibility of false positives.

Dropping any documents that are identified as duplicates, or deleting the duplicate paragraphs, can be done in a subsequent run of the mixer via `dolma mix`.

## Configuration

See sample config files [dedupe-by-url.json](examples/dedupe-by-url.json) and [dedupe-paragraphs.json](examples/dedupe-paragraphs.json).

For an overview of the Dolma document format (including timestamps and metadata fields that are useful for temporal strategies), see the [data format documentation](data-format.md).

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
|`dedupe.paragraphs.by_ngram.ngram_length`|No| If provided, segment each paragraph into [Unicode words](https://www.unicode.org/reports/tr29/) and check whether ngrams of this length are in the Bloom filter. Tagger will report the fraction of matched ngrams in each paragraph. If not provided, full paragraphs are going to be used for the bloom filter. By default, it is off. |
|`dedupe.paragraphs.by_ngram.stride`|No| If provided, it skips `stride` step when computing ngrams. By default, all possible ngrams in a paragraph are checked for duplicates. |
|`dedupe.paragraphs.by_ngram.threshold`|No| If provided, the paragraph is considered a duplicate if the fraction of matched ngrams is greater than or equal to this threshold. By default, it is 1.0, meaning that all ngrams have to match. |
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

## Temporal deduplication and document types

In many curation pipelines, Dolma is applied repeatedly over **multiple temporal snapshots** of the same underlying source (for example, monthly crawls of the web, or periodically updated copies of a code or paper corpus). In these settings, it is often desirable to:

- keep **only one copy** of each document across all snapshots, and
- choose **which copy to keep** based on time (for example, preferring the most recent crawl) or document type.

Dolma does not have a separate “temporal deduper”; instead, temporal behaviour is obtained by **reusing the same Bloom filter across runs** and by controlling the **order in which snapshots are processed**.

### Document structure and timestamps

Dolma documents follow the unified format described in [data-format.md](data-format.md). Two fields are particularly relevant for temporal strategies:

- `source`: identifies the high-level data source (for example, `common-crawl`, `github`, `s2ag`).
- `added` / `created` (optional): timestamps indicating when AI2 acquired the document and when the original document was created (where available).

Temporal policies are usually expressed in terms of:

- **which snapshots to consider** (for example, directories such as `documents/2019-08/`, `documents/2019-09/`), and
- **which document to keep** when multiple snapshots contain equivalent content.

The deduper itself is agnostic to timestamps; it only sees the stream of documents and the state of the Bloom filter. Temporal behaviour comes from how you **sequence your runs**.

### Basic temporal strategy: “keep newest” or “keep oldest”

Suppose you have a directory structure like the one recommended in [data-format.md](data-format.md):

```plain-text
dataset-name/
  documents/
    2019-08/
    2019-09/
    2019-10/
```

You can implement simple temporal policies as follows:

- **Keep the newest copy of each document**
  - Create a single Bloom filter file (for example, `bloom_filter.file = "bloom_filters/web.bin"`).
  - Process snapshots **from newest to oldest**, reusing the same Bloom filter file each time.
  - The first time a document (or paragraph) is seen, it is written and inserted into the Bloom filter. When the deduper later encounters the same content in an older snapshot, it will be treated as a duplicate and only marked in attributes.

- **Keep the oldest copy of each document**
  - Use the same single Bloom filter file.
  - Process snapshots **from oldest to newest**.
  - The first copy you see is considered canonical; later copies are marked as duplicates when encountered.

In both cases, temporal deduplication is entirely controlled by:

- the **order in which you invoke** `dolma dedupe`, and
- the fact that the **Bloom filter file persists** and accumulates keys across invocations (with `bloom_filter.read_only = false`).

### Document types and multiple filters

Real-world corpora often contain multiple **document types** (for example, web pages, academic articles, code repositories) that may or may not be deduplicated against each other. Dolma does not prescribe a specific notion of document type; instead, it is usually encoded via:

- the `source` field on each document, and/or
- fields inside `metadata` (for example, `metadata.doc_type`, `metadata.source_dataset`).

You can express different policies by choosing **how many Bloom filters to maintain**:

- **Shared filter across types**  
  - Use a **single** `bloom_filter.file` for all relevant document types.  
  - All documents that hash to the same key (for example, by URL in `metadata.url` or by paragraph content) will be considered duplicates, even if they come from different sources.

- **Per-type filters**  
  - Use a **separate** `bloom_filter.file` for each type or source (for example, one for `common-crawl`, one for `s2ag`).  
  - This prevents cross-type deduplication while still deduplicating within each type over time.

The choice depends on whether, for a given corpus, you want to treat the same content found in multiple places (for example, a paper PDF vs a preprint) as duplicates or as distinct documents.

### Example: temporal paragraph deduplication by URL

The following (simplified) configuration illustrates a temporal paragraph-level deduplication strategy where we:

- deduplicate paragraphs based on content,
- reuse a single Bloom filter across monthly web snapshots, and
- treat all snapshots as a single logical stream.

```json
{
  "documents": [
    "dataset-name/documents/2024-03/*.jsonl.gz",
    "dataset-name/documents/2024-02/*.jsonl.gz",
    "dataset-name/documents/2024-01/*.jsonl.gz"
  ],
  "dedupe": {
    "name": "paragraph_duplicates_temporal",
    "paragraphs": {
      "attribute_name": "bff_duplicate_paragraph_spans"
    },
    "skip_empty": true,
    "min_length": 0,
    "min_words": 0
  },
  "bloom_filter": {
    "file": "bloom_filters/web_paragraphs_temporal.bin",
    "read_only": false,
    "estimated_doc_count": 6000000,
    "desired_false_positive_rate": 1e-4
  },
  "processes": 16
}
```

Key points:

- The **order of paths** in `documents` reflects the temporal policy (here, newest first, so newer snapshots are treated as canonical).
- The **same Bloom filter file** is used across all snapshots, so once a paragraph has been inserted, later encounters of the same content will be marked as duplicates.
- Document type distinctions (if any) are determined by `source` or `metadata` fields and can be further exploited in downstream [mixer](mixer.md) configs (for example, by filtering or weighting documents differently).

For more complex temporal policies (for example, favouring newer web pages but keeping older scientific articles), you can combine:

- separate runs of `dolma dedupe` with different `bloom_filter.file` values, and
- filtering by `source` or `metadata` fields in subsequent `dolma mix` configurations.
