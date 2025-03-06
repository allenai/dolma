# Code File Concatenation

A standalone-script used for concatenating code files within
a programming language and repo. Assumes data is pre-partitioned
by repo, and sorted by repo+pl. Each partition contain more than
one repo.

Simple at present, concatenates with special token as delimiter;
can randomize file order. Extend from here.

## Usage

Assumes partitions consist of jsonl strings, one per line. Must contain
a `"text"` field as well as a `"metadata"` field. The latter should contain
subfields that specify programming language and repo name. The field names
are parameterized, with defaults shown in the command below.

```shell
cargo run --release -- \
    --inputs 'data/input/*jsonl' \
    --output data/output \
    --randomize-order \
    --file-separator-token '<|file_sep|>'
    --repo-field-name repo_name
    --pl-field-name language
```

It will run over all files matching `data/input/*.jsonl`,
writing results to `data/output`. Ordering and partition will be preserved,
with fewer resulting documents in each output partition, per concatenation.

