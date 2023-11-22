# Taggers

The tag command is used to run any of the built-in taggers on a set of documents. For example:

```shell
dolma tag \
    --experiment sample \
    --documents \
        's3://ai2-llm/pretraining-data/sources/common-crawl/test/v0/documents/**/*.json.gz' \
        's3://ai2-llm/pretraining-data/sources/common-crawl/test/v1/documents/*.json.gz' \
    --taggers random_number_v1 \
    --processes 2
```

This command will run the `random_number_v1` tagger on all documents in the specified S3 paths. The results will be written to the `s3://ai2-llm/pretraining-data/sources/common-crawl/test/v0/attributes/sample` and `s3://ai2-llm/pretraining-data/sources/common-crawl/test/v1/attributes/sample` paths.

## Parameters

The following parameters are supported either via CLI (e.g. `dolma tag --parameter.name value`) or via config file (e.g. `dolma -c config.json tag`, where `config.json` contains `{"parameter" {"name": "value"}}`):

|Parameter|Required?|Description|
|:---:|---|---|
|`documents`|Yes| One or more paths for input document files. Paths can contain arbitrary wildcards. Can be local, or an S3-compatible cloud path. |
|`destination`|No| One or more paths for output attribute files. Each accepts a single wildcard `*` character. Can be local, or an S3-compatible cloud path. If not provided, the destination will be derived from the document path. |
|`experiment`|No| Used to name output attribute files. One output file will be created for each input document file, where the key is obtained by substituting `documents` with `attributes/<experiment>`. If not provided, we will use `attributes/<tagger_name>`. |
|`taggers`|Yes| One or more taggers to run. |
|`tagger_modules`|No| List of one or more Python modules to load taggers from. See section [*"Using Custom Taggers"*](#using-custom-taggers) for more details. |
|`processes`|No| Number of processes to use for tagging. One process is used by default. |
|`ignore_existing`|No| If true, ignore existing outputs and re-run the taggers. |
|`dryrun`|No| If true, only print the configuration and exit without running the taggers. |
|`debug`|No| If true, run in debug mode (i.e., disable parallelism). Useful when developing new taggers. |
|`profile.enable`|No| If true, enable profiling. Useful when benchmarking taggers during development. |
|`profile.output`|No| Path to save the profiling output; if not provided, the output will be printed to stdout. |


## Built-in Taggers

A list of built-in taggers can be obtained by running `dolma list` command. At the time of writing, the following taggers are available:

| Tagger Name | Description |
|:-----------:| ----------- |
| `c4_v1`     | Implements taggers used to generate the [C4](https://arxiv.org/abs/1910.10683) dataset.|
| `c4_v2`     | Faster implementation of the C4 taggers. |
| `char_length_v1` | Computes the length of the document in characters. |
| `char_length_with_paragraphs_v1` | Computes the length of the document and each paragraph in characters. |
| `cld2_en_doc_v2` | Uses [cld2](https://github.com/CLD2Owners/cld2) to detect the language of the document. |
| `cld2_en_paragraph_v2` | Uses [cld2](https://github.com/CLD2Owners/cld2) to detect the language of each paragraph. |
| `cld2_en_paragraph_with_doc_score_v2` | Uses [cld2](https://github.com/CLD2Owners/cld2) to detect the language of each paragraph and assigns a score to the document based on the fraction of paragraphs that are English. |
| `cld3_en_doc_v2` | Uses [cld3](https://github.com/google/cld3) to detect the language of the document. |
| `cld3_en_paragraph_v2` | Uses [cld3](https://github.com/google/cld3) to detect the language of each paragraph. |
| `cld3_en_paragraph_with_doc_score_v2` | Uses [cld3](https://github.com/google/cld3) to detect the language of each paragraph and assigns a score to the document based on the fraction of paragraphs that are English. |
| `code_copyright_comments_v1` | For code documents, tags spans that contain a copyright statement |
| `code_redpajama_taggers_v1` | Applies [RedPajama code processing rules](https://github.com/togethercomputer/RedPajama-Data/tree/main/data_prep/github) to tag spans of documents.
| `code_secrets_v1` | Tags spans that contain secrets (e.g., passwords, API keys, etc.) using the [yelp/detect-secrets](https://github.com/Yelp/detect-secrets) library |
| `ft_lang_id_en_doc_v2` | Uses [fastText](https://fasttext.cc/) to detect the language of the document. |
| `ft_lang_id_en_paragraph_v2` | Uses [fastText](https://fasttext.cc/) to detect the language of each paragraph. |
| `ft_lang_id_en_paragraph_with_doc_score_v2` | Uses [fastText](https://fasttext.cc/) to detect the language of each paragraph and assigns a score to the document based on the fraction of paragraphs that are English. |
| `gopher_v1` | Tags spans of documents matching [Deepmind's Gopher](https://arxiv.org/abs/2112.11446) removal rules. |
| `jigsaw_hatespeech_document_v2` | Tags documents as containing hate speech or not using a FastText classifier trained on the [Jigsaw](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) hate speech dataset. |
| `jigsaw_hatespeech_sentence_v2` | Tags spans of documents as containing hate speech or not using a FastText classifier trained on the [Jigsaw](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) hate speech dataset. |
| `jigsaw_nsfw_document_v1` | Tags documents as containing NSFW content or not using a FastText classifier trained on the [Jigsaw](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) NSFW dataset. |
| `jigsaw_nsfw_sentence_v2` | Tags spans of documents as containing NSFW content or not using a FastText classifier trained on the [Jigsaw](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) NSFW dataset. |
| `olmo_pretokenizer_v1` | Count the number of tokens in each document using pre-tokenizer used by [OLMo v1](allenai.org/olmo), which is a the same as [GPT Neo-X 20B](https://huggingface.co/EleutherAI/gpt-neox-20b). |
| `olmo_pretokenizer_with_paragraphs_v1` | Count the number of tokens in each document and each paragraph using pre-tokenizer used by [OLMo v1](allenai.org/olmo), which is a the same as [GPT Neo-X 20B](https://huggingface.co/EleutherAI/gpt-neox-20b). |
| `pii_presidio_v1` | Tags spans of documents that contain personally identifiable information (PII) using the [Presidio Analyzer](https://microsoft.github.io/presidio/analyzer/) library. |
| `pii_regex_v1` | Tags spans of documents that contain personally identifiable information (PII) using a set of regular expressions. |
| `pii_regex_v2` | Faster implementation of `pii_regex_v1`. |
| `pii_regex_with_counts_v2` | Tags spans of documents that contain personally identifiable information (PII) using a set of regular expressions. It also counts the number of matches for each regular expression. |
| `pii_regex_with_counts_fast_v2` | Faster implementation of `pii_regex_with_counts_v2`. |
| `random_number_v1` | Assigns a random number to each document. This allows us to split the dataset into train, validation, and test sets. |
| `uniseg_length_paragraphs_v1` | Count the number of [unicode "words" (grapheme clusers)](https://www.unicode.org/reports/tr29/) in each paragraph. |
| `uniseg_length_paragraphs_with_doc_length_v1` | Count the number of [unicode "words" (grapheme clusers)](https://www.unicode.org/reports/tr29/) in each paragraph and the document. |
| `whitespace_tokenizer_v1` | Count the number of whitespace-separated tokens in each document. |
| `whitespace_tokenizer_with_paragraphs_v1` | Count the number of whitespace-separated tokens in each document and each paragraph. |

## Adding a New Tagger

All taggers inherit from the `BaseTagger` class defined in [`core/taggers.py`](https://github.com/allenai/dolma/blob/main/python/dolma/core/taggers.py). To add a new tagger, you need to create a new class that inherits from `BaseTagger` and implements the `predict` method. For example, the following code implements a tagger that assigns a random number to each document:

```python
import random

from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger

@add_tagger("new_random_number")
class RandomNumberTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        # first, we generate a random number
        score = random.random()

        # we assign the random score to a span that
        # covers the entire document
        span = Span(
            start=0,
            end=len(doc.text),
            type="random",
            score=score
        )

        # we return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=[span])
```

Name for each tagger is specified using the `add_tagger` decorator. The name must be unique.

## Using Custom Taggers

Taggers can be added either as part of the Dolma package, or they can be imported at runtime by providing the `tagger_modules` parameter.

For example, let's assume `new_random_number` is saved in a file called `my_taggers.py` in python module `my_module`. Then, we can run the tagger using one of the following commands:

- `dolma tag --taggers new_random_number --tagger_modules path/to/my_module/my_taggers.py ...`
- `PYTHONPATH="path/to/my_module" dolma tag --taggers new_random_number --tagger_modules my_taggers`
