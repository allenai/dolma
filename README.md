# dolma

*Data to feed OLMo's Appetite*


<img alt="DOLMa logo. It's a watercolor of grape leaves with the word DOLMa in the top left." src="https://github.com/allenai/dolma/blob/main/res/logo.png?raw=true" width="256">

Data and tools for generating and inspecting OLMo pre-training data.

To get started, install dolma using [pip](https://pypi.org/project/dolma/).

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

### `dolma tag`

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

### `dolma dedupe`

The dedupe command is used to deduplicate a set of documents at the attribute level using a bloom filter.
For example configurations, see directory `tests/config`. For example:

```shell
dolma dedupe -c tests/config/dedupe-paragraphs.json
```

### `dolma mix`

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
    title = {{DOLMa}},
    url = {https://github.com/allenai/dolma}
}
```
