# Contributing to the Dolma Toolkit

We welcome contributions to the Dolma Toolkit. Please read this document to learn how to contribute.

## Development

Create a conda environment with Python >= 3.8. In this case, we use Python 3.11 and use Anaconda to create the environment.

```shell
conda create -n dolma python=3.11
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

You can skip S3 related tests by exporting `DOLMA_TESTS_SKIP_AWS=True`

```shell
DOLMA_TESTS_SKIP_AWS=True make test
```

## Contributing

Before committing, use the following command

```shell
make style
```
