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
First, install the required dependencies:
```bash
pip install git+https://github.com/santhoshtr/wikiextractor.git requests smart_open tqdm
```
Next, use [this script](https://github.com/allenai/dolma/blob/main/scripts/make_wikipedia.py) to download and process Wikipedia:

```shell
python scripts/make_wikipedia.py \
  --output wikipedia \
  --date 20231001 \
  --lang simple \
  --processes 16
```

This script will download and process Wikipedia articles in the `simple` language from the October 1, 2023 Wikipedia dump. After running it, you will find the articles stored in a directory named `wikipedia/v0`. The articles will be grouped into compressed JSONL files suitable for dolma.

Note: Update the `--date 20231001` argument by selecting a specific dump date from the Wikimedia dump website. Make sure to use the date format `YYYYMMDD`.

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

After running the taggers and marking which paragraphs are duplicates, we can run the mixer to create a dataset with a subset of the languages and documents.

For this step, we will pass a configuration file to the `mix` command instead of passing all the options on the command line. The CLI invocation looks like this:

```shell
dolma -c examples/wikipedia-mixer.json mix --processes 16
```

In this case, the configuration is provided via a JSON file, though a YAML file would also work. Additionally, we override the number of processes to 16 using the `--processes` flag.

You can find the configuration file [`wikipedia-mixer.json`](examples/wikipedia-mixer.json) in the examples repository, along with its YAML-equivalent version at [`wikipedia-mixer.yaml`](examples/wikipedia-mixer.yaml).

The configuration will create a directory named `wikipedia/example0/documents` with a set of files containing the documents that pass the filters.

### Step 4: Tokenize The Dataset

Finally, we tokenize the dataset using the `tokens` command. In this example, we use EleutherAI's excellent [GPT Neo-X 20B](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.

```shell
dolma tokens \
    --documents "wikipedia/example0/documents/*.gz" \
    --tokenizer.name_or_path "EleutherAI/gpt-neox-20b" \
    --destination wikipedia/example0/tokens \
    --processes 16
```

Tokenized documents will be written to `wikipedia/example0/tokens`.
