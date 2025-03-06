# Fill-In-Middle (FIM)

A standalone-script used for producing reordered code text,
for use in training code completion between a given code block
prefix and suffix.

For instance, given an example block of code like this:

```plaintext
def add_two_numbers(a: int, b: int) -> int:
    sum = a + b
    return sum
```

we would expect a so-called FIM rearrangement
of this general format:

```plaintext
<|fim_prefix>def add_two_numbers(a: int, b: int) -> int:
<|fim_suffix|>
    return sum<|fim_middle|>    sum = a + b
```

Note the use of special sentinel tokens to demarcate the beginnings
of each rearranged subsection.

## Usage

Examples to rewrite as FIM documents must be partitioned into 1 or more
jsonl files, where each row contains a member `"text"`. Partitioning and
ordering within partitions will be preserved in output files.

If more than one code source file is present per `"text"` entry, delimit
with a separator token, e.g. `<|file_sep|>`. Reordering will only be applied
within a given source file.

Here's an example on how to use the FIM script:

```shell
cargo run --release -- \
    --inputs 'data/input/*jsonl' \
    --output data/output \
    --fim-rate 0.5 \
    --psm-spm-split 0.25 \
    --file-separator-token '<|file_sep|>'
    --fim-prefix-token '<|fim_prefix|>'
    --fim-middle-token '<|fim_middle|>'
    --fim-suffix-token '<|fim_suffix|>'
```

It will run over all files matching `data/input/*.jsonl`,
writing results to `data/output`. FIM reordering will be applied
to 50% of source code files detected within the provided rows after splitting
on `<|file_sep|>`. Of those, 25% will be ordered as Prefix-Suffix-Middle (psm),
and the remaining 75% will be reordered as Suffix-Prefix-Middle (spm).

