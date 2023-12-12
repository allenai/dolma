# Deduplication

Dolma pioneers an approach for the removal of pretraining data contaminated with respect to perplexity evaluations. The following procedure is used to remove contamination with respect to [Paloma](paloma.allen.ai), a benchmark of perplexity evaluations, which we apply to Dolma. Here we describe how to apply this same decontamination to your own data.

## Setup
Format your data in the standardized format for Dolma tooling detailed [here](data-format.md)

Then set up the Dolma tooling
```
pip install dolma
```

## Marking contamination
Here we mark parts of the pretraining data that are contaminated with respect to the Paloma evaluation suite.

First we make a `dedupe` configuration following the example [here](examples/mark_paloma_contam.yaml). You'll need to change the values with a comment with this symbol (❗) next to them, including the setting the paths to your data. This will make use of the [precomputed bloom filter](../bloom_filters/paloma_decon.bin) that is already fit to the evaluation data in Paloma.

Then from the root directory run the following command to add attributes files to your data that record contamination. Note that this may require explicitly setting the `TMPDIR` environment variable to a directory with at least enough space for your whole pretraining corpus.

```
dolma -c docs/examples/mark_paloma_contam.yaml dedupe
```

## Removing contaminated data
Next we will remove any document that has any contamination

We need to make a `mix` configuration following the example [here](examples/remove_paloma_contam.yaml). Again, you'll need to change the values with a comment with this symbol (❗) next to them, including the setting the paths to your data.

Then run the following command from the root directory
```
dolma -c docs/examples/remove_paloma_contam.yaml mix
```

Now the decontaminated data will be located at the output path you provided in the configuration.