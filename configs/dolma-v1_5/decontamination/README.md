# Decontamination Runbook

## Step 1: Create decontamination bloom filter

> Okay I think every thing is ready for decon testing now. The finalized ppl suite v3 is in `s3://ai2-llm/eval-data/perplexity/v3/`. And here is my proposed plan for decon testing if you agree and it's not too much compute. The following is the sequence of things to try. At each step if the document removal rate is >0.1% or so we back off to the next step and hope the remove rate is lower:
>
> - **Option 1** Decon against PPL Suite v3 (`s3://ai2-llm/eval-data/perplexity/v3/`) + PPL Suite v2 (`s3://ai2-llm/eval-data/perplexity/v2/`) for full backwards compatibility.
> - **Option 2** Decon against PPL Suite v3 (`s3://ai2-llm/eval-data/perplexity/v3/`) + PPL Suite v2-small (`s3://ai2-llm/eval-data/perplexity/v2_small/`) for at least full backwards for the in-loop metrics the model team was using.
> - **Option 3** Decon against PPL Suite v3 (`s3://ai2-llm/eval-data/perplexity/v3/`) + a subset of PPL Suite v2-small requested by Dirk and Iz (`s3://ai2-llm/eval-data/perplexity/v2_small/c4_en/`, `s3://ai2-llm/eval-data/perplexity/v2_small/pile/`, `s3://ai2-llm/eval-data/perplexity/v2_small/m2d2_s2orc/`, `s3://ai2-llm/eval-data/perplexity/v2_small/ice/`)
>
> Let me know if you disagree with any of this or if there's any thing I can do to help run the decon trials!


### Step 1.1: copy data locally

We copy data locally since the directory structure of the eval data in S3 is slightly different from the one we need.
In particular, we need all documents to be under `documents/` directory.

```bash
aws s3 sync s3://ai2-llm/eval-data/perplexity/v2 $HOME/perplexity/v2/documents
aws s3 sync s3://ai2-llm/eval-data/perplexity/v2_small $HOME/perplexity/v2_small/documents
aws s3 sync s3://ai2-llm/eval-data/perplexity/v3 $HOME/perplexity/v3/documents

aws s3 sync s3://ai2-llm/eval-data/perplexity/v2_small/c4_en $HOME/perplexity/v2_small_subset/documents/c4_en
aws s3 sync s3://ai2-llm/eval-data/perplexity/v2_small/pile $HOME/perplexity/v2_small_subset/documents/pile
aws s3 sync s3://ai2-llm/eval-data/perplexity/v2_small/m2d2_s2orc $HOME/perplexity/v2_small_subset/documents/m2d2_s2orc
aws s3 sync s3://ai2-llm/eval-data/perplexity/v2_small/ice $HOME/perplexity/v2_small_subset/documents/ice
```

### Step 1.1b: change type of IDs in v3 subset (TEMPORARY FIX)

v3 accidentally contains ids that are integers instead of strings. Until that's fixed, run:

```bash
python config/dolma-v1_5/decontamination/fix_ids_type.py
```

### Step 1.2: tag out paragraphs by uniseg length

For dolma, we want to decontaminate against paragraphs that are at least 13 uniseg words long,
so we need to compute their length first.

```bash
dolma tag --documents "${HOME}/perplexity/v2/documents/*/*/*.gz" --taggers uniseg_length_paragraphs_with_empty_v1 not_alphanum_paragraph_v1 --processes 188
dolma tag --documents "${HOME}/perplexity/v2_small/documents/*/*/*.gz" --taggers uniseg_length_paragraphs_with_empty_v1 not_alphanum_paragraph_v1 --processes 188
dolma tag --documents "${HOME}/perplexity/v3/documents/*/*/*.gz" --taggers uniseg_length_paragraphs_with_empty_v1 not_alphanum_paragraph_v1 --processes 188
dolma tag --documents "${HOME}/perplexity/v2_small_subset/documents/*/*/*.gz" --taggers uniseg_length_paragraphs_with_empty_v1 not_alphanum_paragraph_v1 --processes 188
```

### Step 1.3: filter out paragraphs that are too short

After tagging, we can filter out to make option 1/2/3.

```bash

dolma -c configs/dolma-v1_5/decontamination/step1_3-make-eval-set/option1.yaml mix
dolma -c configs/dolma-v1_5/decontamination/step1_3-make-eval-set/option2.yaml mix
dolma -c configs/dolma-v1_5/decontamination/step1_3-make-eval-set/option3.yaml mix

```

### Step 1.4: create bloom filter

First, we cat the contents of each dataset to get number of documents:

```bash
zcat $HOME/perplexity/option1/documents/* | jq '.text' -cr | wc -l
>>> 3681169
zcat $HOME/perplexity/option2/documents/* | jq '.text' -cr | wc -l
>>> 2336120
zcat $HOME/perplexity/option3/documents/* | jq '.text' -cr | wc -l
>>> 2020471
```

We use this numbers in the config files at `bloom_filter.estimated_doc_count`. For all three options, we set a `bloom_filter.desired_false_positive_rate` of 0.00001.

```bash
dolma -c configs/dolma-v1_5/decontamination/step1_4-create-bloom-filter/option1.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step1_4-create-bloom-filter/option2.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step1_4-create-bloom-filter/option3.yaml dedupe
```

## Step 2: Run decontamination

Tag content for Dolma V1.5 for decontamination:


```bash
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/cc.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/c4.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/stack.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/reddit.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/peS2o.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/books.yaml dedupe
dolma -c configs/dolma-v1_5/decontamination/step2-run-decontamination/wiki.yaml dedupe
```
