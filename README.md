# dolma

*Data to feed OLMo's Appetite*


<img alt="DOLMa logo. It's a watercolor of grape leaves with the word DOLMa in the top left." src="https://github.com/allenai/dolma/blob/main/res/logo.png?raw=true" width="256"></img>

Data and tools for generating and inspecting OLMo pre-training data.


## Setup

Create a conda environment with Python >= 3.8. In this case, we use Python 3.10 and use Anaconda to create the environment.

```shell
conda create -n dolma python=3.10
```

After creating the environment, activate it and install necessary tools using the included makefile.

```shell
conda activate dolma
make setup
```

Finally, to begin development, install the repository in editable mode using maturin.

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
