# Getting Started

To get started, please install Dolma using `pip`:

```shell
pip install dolma
```

After installing Dolma, you get access to the `dolma` command line tool. To see the available commands, use the `--help` flag.

```
$ dolma --help

usage: dolma [command] [options]

Command line interface for the DOLMa dataset
processing toolkit

positional arguments:
  {dedupe,mix,tag,list,stat,tokens}
    dedupe          Deduplicate documents or
                    paragraphs using a bloom
                    filter.
    mix             Mix documents from multiple
                    streams.
    tag             Tag documents or spans of
                    documents
                    using one or more taggers.
                    For a list of available
                    taggers, run `dolma list`.
    list            List available taggers.
    stat            Analyze the distribution
                    of attributes values in a
                    dataset.
    tokens          Tokenize documents using
                    the provided tokenizer.

options:
  -h, --help        Show this help message
                    and exit
  -c CONFIG, --config CONFIG
                    Path to configuration
                    optional file
```

The CLI supports six commands: `dedupe`, `mix`, `tag`, `list`, `stat`, and `tokens`.
Each command has its own set of options.
To see the options for a command, use the `--help` flag, e.g., `dolma tag --help`.

In this tutorial, we will show how to use the `tag`, `dedupe`, and `mix` commands to curate a wikipedia dataset.

## Example: Process Wikipedia

Run all following commands from root of this repository.

## Step 1: Run Taggers

We once

```shell
ai2_llm_filters \
    -d wikipedia/v0 \
    -n abl0 \
    -t random_number_v1 \
        cld2_en_paragraph_with_doc_score_v2 \
        ft_lang_id_en_paragraph_with_doc_score_v2 \
        char_length_with_paragraphs_v1 \
        whitespace_tokenizer_with_paragraphs_v1 \
    -p 96   # run on 96 cores
```

## Step 2: Deduplicate Against Perplexity Eval Set

Compile and install mixer/deduper:

```shell
cd pretrain_data/mixer
make build-tools    # will install rust and tools to build the mixer
make mixer          # will build the mixer; available at ./target/release/mixer
```

Download the bloom filter for decontamination:

```shell
aws s3 cp \
    s3://ai2-llm/eval-data/perplexity/blocklists/eval_subset_v2/deduper_decontamination_lucas_20230525.bin \
    /tmp/decontamination/deduper_decontamination_lucas_20230525.bin
```

Now run the deduper:

```shell
DEDUPER_BIN="pretrain_data/mixer/target/release/deduper"
$DEDUPER_BIN \
    examples/wikipedia_ablation/deduper_config.json
```

## Step 3: Run Mixer

Run mixer with `mixer_config.json`:

```shell
MIXER_BIN="pretrain_data/mixer/target/release/mixer"
$MIXER_BIN \
    examples/wikipedia_ablation/mixer_config.json
```

You can check out the mixer config to see how it works. In particular, it applies four operations:

- Include all documents with length less than 100,000 whitespace-separated words:

    ```yaml
    "include": [
        "$.attributes[?(@.abl0__whitespace_tokenizer_with_paragraphs_v1__document[0][2] < 100000)]"
    ]
    ```

- Remove any document that is shorter than 50 words:

    ```yaml
    "exclude": [
        "$.attributes[?(@.abl0__whitespace_tokenizer_with_paragraphs_v1__document[0][2] < 50)]",
        ...
    ]
    ```

- Remove any document whose total English cld2 score is below 0.5:

    ```yaml
    "exclude": [
        ...,
        "$.attributes[?(@.abl0__ft_lang_id_en_paragraph_with_doc_score_v2__doc_en[0][2] <= 0.5)]",
        ...
    ]
    ```

- Replace paragraphs whose not-English cld2 socre is below 0.9 in a document with an empty string

    ```yaml
    "span_replacement": [
        {
            "span": "$.attributes.abl0__cld2_en_paragraph_with_doc_score_v2__not_en",
            "min_score": 0.1,
            "replacement": ""
        },
        ...
    ]
    ```

- Remove all documents that contain a paragraph that has tagged as duplicates with the validation set using bff

    ```yaml
    "exclude": [
        ...,
        "$@.attributes[?(@.bff_duplicate_paragraph_spans && @.bff_duplicate_paragraph_spans[0] && @.bff_duplicate_paragraph_spans[0][2] >= 1.0)]"
    ]
    ```

Note how the configuration only runs the mixing on 27 languages.
Nevertheless, with the filters above, we went from 27GB to just over 8.4GB.
