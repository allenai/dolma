# Getting Started

To get started, please install Dolma toolkit using `pip`:

```shell
pip install dolma
```

After installing Dolma toolkit, you get access to the `dolma` command line tool. To see the available commands, use the `--help` flag.

```plain-text
$ dolma --help

usage: dolma [command] [options]

Command line interface for the DOLMa dataset processing toolkit

positional arguments:
  {dedupe,mix,tag,list,stat,tokens}
    dedupe              Deduplicate documents or paragraphs using a bloom filter.
    mix                 Mix documents from multiple streams.
    tag                 Tag documents or spans of documents using one or more taggers. For a
                        list of available taggers, run `dolma list`.
    list                List available taggers.
    stat                Analyze the distribution of attributes values in a dataset.
    tokens              Tokenize documents using the provided tokenizer.

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to configuration optional file
```

The CLI supports six commands: `dedupe`, `mix`, `tag`, `list`, `stat`, and `tokens`.
Each command has its own set of options.
To see the options for a command, use the `--help` flag, e.g., `dolma tag --help`.

In this tutorial, we will show how to use the `tag`, `dedupe`, and `mix` commands to curate a wikipedia dataset.

## Example: Process Wikipedia

Run all following commands from root of this repository.

### Step 0: Obtain Wikipedia

We use [this script](https://github.com/allenai/dolma/blob/main/scripts/make_wikipedia.py) to download and process Wikipedia:

```shell
python scripts/make_wikipedia.py \
  --output wikipedia \
  --languages simple \
  --date 20231001 \
  --lang simple \
  --num_processes 16
```

The code above will download and process Wikipedia articles in the `simple` language from the October 1, 2023 wikipedia dump.
After running it, you will have a directory called `wikipedia/v0` with Wikipedia articles in it.
Wikipedia articles are going to be grouped in compressed JSONL files in dolma

### Step 1: Run Taggers

Our first step in preparing the Wikipedia is to tag it with a few taggers. We will use the following taggers:

- `random_number_v1`: Assigns a random number to each document. This allows us to split the dataset into train, validation, and test sets.
- `cld2_en_paragraph_with_doc_score_v2`: Uses the cld2 language detector to tag each paragraph as English or not English. It also assigns a score to each document based on the fraction of paragraphs that are English.
- `ft_lang_id_en_paragraph_with_doc_score_v2`: Uses the fastText language detector to tag each paragraph as English or not English. It also assigns a score to each document based on the fraction of paragraphs that are English.
- `char_length_with_paragraphs_v1`: Counts the number of characters in each document and each paragraph.
- `whitespace_tokenizer_with_paragraphs_v1`: Counts the number of whitespace-separated tokens in each document and each paragraph.

To get a list of available taggers, run `dolma list`.

To invoke the tagger, run:

```bash
dolma tag \
    --documents "wikipedia/v0/documents/*" \
    --experiment exp \ # optional; assigning a name groups taggers in a single directory
    --taggers random_number_v1 \
              cld2_en_paragraph_with_doc_score_v2 \
              ft_lang_id_en_paragraph_with_doc_score_v2 \
              char_length_with_paragraphs_v1 \
              whitespace_tokenizer_with_paragraphs_v1 \
    --processes 16   # run on 96 cores
```

To learn more about the taggers, see the [taggers documentation](taggers.md).

### Step 2: Deduplicate Paragraphs

After tagging, we deduplicate the dataset at a paragraph level.

```shell
dolma dedupe \
    --documents "wikipedia/v0/documents/*" \
    --dedupe.paragraphs.attribute_name 'bff_duplicate_paragraph_spans' \
    --dedupe.skip_empty \
    --bloom_filter.file /tmp/deduper_bloom_filter.bin \
    --no-bloom_filter.read_only \
    --bloom_filter.estimated_doc_count '6_000_000' \
    --bloom_filter.desired_false_positive_rate '0.0001' \
    --processes 188
```

The above command will create an attribute directory called `bff_duplicate_paragraph_spans` in `wikipedia/v0/attributes`. The `bff_duplicate_paragraph_spans` attribute will contain a list of duplicate paragraphs for each paragraph in the dataset.

### Step 3: Run Mixer

After running the taggers and and marking which paragraphs are duplicates, we can run the mixer to create a dataset with a subset of the languages and documents.

For this step, we will pass a configuration file to the mix command instead of passing all the options on the command line. CLI invocation looks like this:

```shell
dolma -c wikipedia-mixer.json mix --processes 16
```

Note how the configuration in this case is a JSON file; a YAML file would also work.
Further, we override the number of processes to use to 96 using the `--processes` flag.

`wikipedia-mixer.json` looks like the following (A YAML-equivalent version is available at [`wikipedia-mixer.yaml`](examples/wikipedia-mixer.yaml)):


```yaml
{
  # mix command operates on one or more stream; each can correspond to a different data source
  # and can have its own set of filters and transformations
  "streams": [
    {
      # name of the stream; this will be used as a prefix for the output files
      "name": "getting-started",
      # the documents to mix; note how we use a glob pattern to match all documents
      "documents": [
        "wikipedia/v0/documents/*.gz",
      ]
      # this is the directory where the output will be written
      # note how the toolkit will try to create files of size ~1GB
      "output": {
        "path": "wikipedia/example0/documents",
        "max_size_in_bytes": 1000000000
      },
      "attributes": [
        "exp",                           # load the attributes from the taggers
        "bff_duplicate_paragraph_spans"  # load the attributes from the deduper
      ],
      # filers remove or include whole documents based on the value of their attributes
      "filter": {
        "include": [
           # Include all documents with length less than 100,000 whitespace-separated words
          "$.attributes[?(@.exp__whitespace_tokenizer_with_paragraphs_v1__document[0][2] < 100000)]"
        ],
        "exclude": [
          # Remove any document that is shorter than 50 words
          "$.attributes[?(@.exp__whitespace_tokenizer_with_paragraphs_v1__document[0][2] < 50)]",
          # Remove any document whose total English fasttext score is below 0.5
          "$.attributes[?(@.exp__ft_lang_id_en_paragraph_with_doc_score_v2__doc_en[0][2] <= 0.5)]",
          # Remove all documents that contain a duplicate paragraph
          "$@.attributes[?(@.bff_duplicate_paragraph_spans && @.bff_duplicate_paragraph_spans[0] && @.bff_duplicate_paragraph_spans[0][2] >= 1.0)]"
        ]
      },
      # span replacement allows you to replace spans of text with a different string
      "span_replacement": [
        {
          # remove paragraphs whose not-English cld2 socre is below 0.9 in a document
          "span": "$.attributes.exp__cld2_en_paragraph_with_doc_score_v2__not_en",
          "min_score": 0.1,
          "replacement": ""
        }
      ]
    }
  ],
  # this process option is overridden by the command line flag
  "processes": 1
}
```

The above configuration will create a directory called `wikipedia/example0/documents` with a set of files that contain the documents that pass the filters.

### Step 4: Tokenize The Dataset

Finally, we tokenize the dataset using the `tokens` command. In this example, we use EleutherAI's excellent [GPT Neo-X 20B](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

```shell
dolma tokens \
    --documents "wikipedia/example0/documents/*.gz" \
    --tokenizer_name_or_path "EleutherAI/gpt-neox-20b" \
    --destination wikipedia/example0/tokens \
    --processes 16
```

Tokenized documents will be written to `wikipedia/example0/tokens`.
