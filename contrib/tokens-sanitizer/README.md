# Tokens Sanitizer

This script is designed to sanitize a documents by replacing special tokens with properly spaced versions.

Why would you want to do that?
Usually, when tokenizing data for pretraining, we split any special token that occur naturally in the text before adding special tokens to denote end of documents.
This is useful for cases when you want a model to be able to answer questions such as "what is <|endoftext|>?";
if you don't escape special tokens properly, the model would conflate different uses of `<|endoftext|>`, and likely would never learn its meta-semantics (since it is vastly more likely to be used to indicate the end of a document).

However, in some cases, data pipeline might inject special tokens that should be treated as such:
For example, when concatenating files in a code repository, you might want to use `<|repo_name|>author/repo<|file_sep|>` to prefix all your code documents.

In that case, we use the following strategy:

1. We first run this sanitization script to process all original documents; if used as described below, this script would take care of modifying special tokens in a way that is equivalent to them being escaped.
2. Then, we perform whatever text substitution we want to inject special tokens.
3. Finally, we tokenize, but, this time, we set `split_special_tokens=False`, since sanitization script already took care of encoding.

**⚠️ IMPORTANT ⚠️***

You must use a **tokenizer that has been modified** to support sanitization. In the rest of this Readme, we use [`allenai/dolma2-tokenizer-U10F0F0`](https://huggingface.co/allenai/dolma2-tokenizer-U10F0F0).


## How does the sanitization work?

When running the sanitization script, we inject a Unicode code point from [Supplementary Private Use Area-B (SPUA-B)](https://en.wikipedia.org/wiki/Private_Use_Areas).
The Unicode consortium does not assign symbols to these code points; rather, they are left open for private entities to use in their own software.
While the Basic Private Use Area (U+E000 to U+F8FF) is commonly used (for example, Apple uses `U+F8FF` for the Apple logo ), the supplementary area B is exceedingly rare.
I picked `U+10F0F0` for sanitization.

In order for sanitization to work, we must use a tokenizer that is designed to split of this token as part of its pre-tokenization strategy.
In HuggingFace Tokenizers library, we achieve that by adding the follow pre-tokenization rule:

```json
  ...
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {
          "String": "􏃰"
        },
        "behavior": "Removed",
        "invert": false
      },
      ...
```

## Usage

Here's an example on how to use this sanitization script.

```shell
cargo run --release -- \
    --inputs 'data/input/*jsonl' \
    --output data/output \
    --substitutions '<|endoftext|>=<|􏃰endoftext|>' \
    --substitutions '<|fim_prefix|>=<|􏃰fim_prefix|>' \
    --substitutions '<|fim_middle|>=<|􏃰fim_middle|>' \
    --substitutions '<|fim_suffix|>=<|􏃰fim_suffix|>' \
    --substitutions '<|im_start|>=<|􏃰im_start|>' \
    --substitutions '<|im_end|>=<|􏃰im_end|>' \
    --substitutions '<|endofprompt|>=<|􏃰endofprompt|>' \
    --substitutions '<|pad|>=<|􏃰pad|>' \
    --substitutions '<|repo_name|>=<|􏃰repo_name|>' \
    --substitutions '<|file_sep|>=<|􏃰file_sep|>'
```

If you are tokenizing using Dolma toolkit, you should tokenize using the following config

```yaml
destination: ...
documents:
  - ...

processes: ...
seed: ...
max_size: ...
dtype: uint32

tokenizer:
  name_or_path: allenai/dolma2-tokenizer-U10F0F0
  bos_token_id: null
  eos_token_id: 100257
  pad_token_id: 100277
  segment_before_tokenization: false
  encode_special_tokens: false
```

Note the `encode_special_tokens` and `name_or_path` keys.

If you are using Hugging Face `transformers`, make sure to set `split_special_tokens` to `False`:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer-U10F0F0")

text = ...

tok(text, split_special_tokens=False)
```
