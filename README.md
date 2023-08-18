<img alt="Dolma's official logo. It's dolma written in yellow, round lowercase letters over a blue background." src="https://github.com/allenai/dolma/blob/main/res/logo.png?raw=true" width="100%">


Dolma is an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
It was created as a training corpus for [OLMo](https://allenai.org/olmo), AI2 language model.

Dolma is available for download on the HuggingFace 🤗 Hub: [`huggingface.co/datasets/allenai/dolma`](https://huggingface.co/datasets/allenai/dolma). To access Dolma, users must agree to the terms of the terms of [AI2 ImpACT License for Medium Risk Artifacts](https://allenai.org/licenses/impact-mr).
You can also read more about Dolma in [our announcement](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64), as well as by consulting its [data sheet](https://drive.google.com/file/d/12gOf5I5RytsD159nSP7iim_5zN31FCXq/view?usp=drive_link).

This repository contains tools for generating and inspecting Dolma. To get started, install the Dolma Python library from [PyPI](https://pypi.org/project/dolma/).

```shell
pip install dolma
```

## Usage

The dolma CLI can be access using the `dolma` command. To see the available commands, use the `--help` flag.

```shell
dolma --help
```

At the moment, the CLI supports three commands: `tag`, `dedupe`, and `mix`.

For all commands, configurations can be specified from command line, or by passing a YAML or JSON file using the `-c` flag. For example:

```shell
dolma -c config.yaml dedupe --dedupe.name "test"
```

### The `tag` command

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

### The `dedupe` command

The dedupe command is used to deduplicate a set of documents at the attribute level using a bloom filter.
For example configurations, see directory `tests/config`. For example:

```shell
dolma dedupe -c tests/config/dedupe-paragraphs.json
```

### The `mix` command

The mix command is used to mix documents from multiple sources, optionally filtering by attributes and/or performing string replacement. For example configurations, see directory `tests/config`. For example:

```shell
dolma mix -c tests/config/mixer.json
```


## Development

Create a conda environment with Python >= 3.8. In this case, we use Python 3.10 and use Anaconda to create the environment.

```shell
conda create -n dolma python=3.10
```

After creating the environment, activate it and install necessary tools using the included makefile.

```shell
conda activate dolma
make setup
```

and restart your shell. Finally, to begin development, install the repository in editable mode using maturin.

```shell
make develop
```

To run tests, use the following command.

```shell
make test
```

You can choose to run just the Python or Rust tests by calling `make test-python` or `make test-rust` respectively.


## Citation

If you use this repository, please cite it as:

```bibtex
@software{dolma,
    author = {{Soldaini, Luca and Lo, Kyle and Kinney, Rodney and Naik, Aakanksha and Ravichander, Abhilasha and Bhagia, Akshita and Groeneveld, Dirk and Schwenk, Dustin and Magnusson, Ian and Chandu, Khyathi}},
    license = {{Apache-2.0}},
    title = {{Dolma}},
    url = {https://github.com/allenai/dolma}
}
```
