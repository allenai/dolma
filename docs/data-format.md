# Data Format

In this document, we explain the data format for the datasets processed by Dolma toolkit.


## Directory Structure

While all components of the Dolma toolkit can read from arbitrary local and S3 locations, we recommend the following directory structure for storing datasets:

```plain-text
|-- dataset-name/
    |-- documents/
        |-- 2019-09/
            |-- 0933_uk_all.jsonl.gz        (1GB)
            |-- 0933_vi_all.jsonl.gz        (1GB)
            |-- 0106_uk_all.jsonl.gz        (1GB)
            |-- 0106_vi_all.jsonl.gz        (1GB)
        |-- 2019-08/
            |-- ...
    |-- attributes/
        |-- toxicity-0/
            |-- 2019-09/
                |-- 0933_uk_all.jsonl.gz    (..MB)
                |-- 0933_vi_all.jsonl.gz    (..MB)
                |-- 0106_uk_all.jsonl.gz    (..MB)
                |-- 0106_vi_all.jsonl.gz    (..MB)
            |-- 2019-08/
                |-- ...
        |-- paragraph_duplicates/
            |-- ...
```

In the example above, all data is stored under the `documents` subdirectory. The directory structure under `documents` is left up to Dolma users. Each file in the `documents` directory is a gzipped JSONL file, where each line is a JSON object representing a document. We explain format of each file in the next section.

Data produced by taggers and deduper is stored under `attributes/attribute-name`; the original directory structure is preserved, and each attributes file contains the same documents as the corresponding file in `documents`.


### Dolma Document Format

This is the unified format we will use across all the sources to represent a single **document**. Each row in one of the `documents/*/*.jsonl.gz` file looks like:

```yaml
{
    "id": "...",             # MANDATORY: source-specific identifier
    "text": "foo",           # MANDATORY: textual content of the document
    "source": "...",         # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
    "added": "...",          # OPTIONAL: timestamp ai2 acquired this data
    "created": "..."         # OPTIONAL: timestamp when orig document was created (best-guess if not available)
    "metadata": {...}        # OPTIONAL: source-specific metadata
}
```

#### `id` field

The `id` field is very important as we will need:

- the ability to trace every single document in every version back to the original source document,
- the ability to store a `blocklist` of documents (e.g. avoid due to LLM-Eval, takedown requests, manual inspection).

It is important that document IDs are stable across dataset versions. For example, Document 12345 in `raw` is the same as Document 12345 in `v0`, `v1`, ...

The `id` only needs to be consistent/unique within a `source`. For example, `id='123'` is acceptable because `(c4, '123')` and `(github, '123')` would uniquely identify this document still. But there cannot be two rows in The Stack `v0` dataset that has `id='123'`.

#### `metadata` field

The `metadata` field will be a free-for-all field that contains any source-specific information. This could be things like code license for the Stack, or paper identifiers for Semantic Scholar (S2) data.

It is especially important to preserve source-specific identifiers when possible. For example, in S2 raw data, we have S2 IDs for each document, but we should also persist things like the DOI, arXiv ID, ACL ID, PubMed ID, etc. when they're available to us.

### Dolma Toolkit Attributes Format

Let's say we are at a good state of document, but we need to iterate on the toxicity classifier a few times. We don't want to duplicate multiple copies of the dataset just because we updated the toxicity classifier. Hence, we store **documents** separately from **attributes**, where attributes are newly derived/predicted aspects as a result of using our tools to analyze the documents.

These are flat JSONs that look like:

```yaml
{
    "source": "...",
    "id": "...",
    "attributes": {
      "toxicity": 0.7
    }
}
```

where the `source` and `id` keys uniquely identify which document carries these attributes.

The mixer create a unified `attributes` dictionary by merging all of the individual `attributes` dictionaries.

Note that it's very important that the `*.jsonl.gz` files for attributes lines up exactly (same number of rows, same sort order) with the `*.jsonl.gz` files for the associated documents. It'll save us a lot of headache in the future.

For something like Language identification, this JSON might look like:

```yaml
{
    "id": "...",
    attributes: {
        "olmo_mix_v1_taggers__ft_lang_id_en_paragraph_with_doc_score_v2__en": [
            [0, 300, 0.9],         # this means text[0:300] is tagged with score 0.9
            [300, 540, 0.3],       # this means text[300:540] is tagged with score 0.3
            ...
        ],
        ...
    }
}
```

Each attribute can have one or more scores associated with it; in the example above, each paragraph in the document is tagged with a language score.
For each paragraph, the tuple indicate the start and end index of the paragraph, and the score associated with it.

The idea that we're going with is that attributes identify spans of text within a document that might be problematic.
These signal get cached during tagging and allow for "building" of the dataset to happen as a configuration afterwards. so for example, given signal data like this, we might try different confidence thresholds on mean_word_length when creating final data mixture
how does your signals data look?
}
