# Data Format

In this document, we explain the data format for the datasets processed by Dolma.


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

### Dolma Attributes Format

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

```
{
    "source": "...",
    "id": "...",
    "attributes": {
      "lang": {"en": 0.9, "fr": 0.2, "de": 0.1 }
    }
}
```

###### `attribute` names

We need a separate versioning schemes for Attributes and Documents. To keep things simple, just increment the name of the attribute as you make updates (e.g. `toxicity-0` vs `toxicity-1`).

### tools

To make progress on dataset versions, we will employ tools. These are still TBD, but the idea is that the input & output of these tools always preserves the JSON format in each `jsonl.gz` dump, so we can re-run functions applied to earlier dataset versions onto later dataset versions without worrying about format changes.

More details can be found in this [Proposal Doc](https://docs.google.com/document/d/18T5_v3QeWPiiuSsUi09_6-ZxW_i47cBatblABb9IZ0M/edit?usp=sharing).

##### validate files on s3

Try running the tool like this:

```python
python pretrain_data/toolkit/src/ai2_llm_toolkit/api.py --source stack-dedup --version raw
python pretrain_data/toolkit/src/ai2_llm_toolkit/api.py --source s2 --version v2_hard_dedup
python pretrain_data/toolkit/src/ai2_llm_toolkit/api.py --source common-crawl --version v0
```

A good outcome would be something like:

```python
python pretrain_data/api.py --source common-crawl --version v0
2023-03-28 22:42:45,903 INFO Creating Dataset from S3: source=common-crawl version=v0
2023-03-28 22:42:45,903 INFO Found one dataset from source=common-crawl version=v0
2023-03-28 22:42:45,919 INFO Found credentials in shared credentials file: ~/.aws/credentials
2023-03-28 22:42:46,895 INFO Inspecting first file at s3://ai2-llm/pretraining-data/sources/common-crawl/v0/documents/mined_split/2021-49/0000/af_all.json.gz
2023-03-28 22:42:46,900 INFO Downloading s3://ai2-llm/pretraining-data/sources/common-crawl/v0/documents/mined_split/2021-49/0000/af_all.json.gz to a temporary file
2023-03-28 22:42:47,602 INFO Finished verifying format of file s3://ai2-llm/pretraining-data/sources/common-crawl/v0/documents/mined_split/2021-49/0000/af_all.json.gz
```

A bad outcome would be something like:

```python
Traceback (most recent call last):
  File "/Users/kylel/ai2/LLM/pretrain_data/api.py", line 177, in <module>
    dataset.verify_one_file(s3_filepath=first_s3_filepath)
  File "/Users/kylel/ai2/LLM/pretrain_data/api.py", line 156, in verify_one_file
    for example in self._read_examples_from_file(s3_filepath=s3_filepath):
  File "/Users/kylel/ai2/LLM/pretrain_data/api.py", line 142, in _read_examples_from_file
    example = Example.from_json(example_json=example_dict)
  File "/Users/kylel/ai2/LLM/pretrain_data/api.py", line 83, in from_json
    source=example_json['source'],
KeyError: 'source'
```

The issue in this case is probably the JSON schema uploaded doesn't adhere to the data contract specified above.
