<img alt="Dolma's official logo. It's dolma written in yellow, round lowercase letters over a blue background." src="https://github.com/allenai/dolma/blob/main/res/logo.png?raw=true" width="100%">


Dolma is two things:

1. **Dolma Dataset**: an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
2. **Dolma Toolkit**: a high-performance toolkit for curating datasets for language modeling.

## Dolma Dataset

Dolma is an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
It was created as a training corpus for [OLMo](https://allenai.org/olmo), AI2 language model.

Dolma is available for download on the HuggingFace 🤗 Hub: [`huggingface.co/datasets/allenai/dolma`](https://huggingface.co/datasets/allenai/dolma). To access Dolma, users must agree to the terms of the terms of [AI2 ImpACT License for Medium Risk Artifacts](https://allenai.org/licenses/impact-mr).
You can also read more about Dolma in [our announcement](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64), as well as by consulting its [data sheet](https://drive.google.com/file/d/12gOf5I5RytsD159nSP7iim_5zN31FCXq/view?usp=drive_link).


## Dolma Toolkit

The Dolma Toolkit is a series of high-performance tools for curating datasets for language modeling. It is designed to be highly portable between different compute environments, including local machines, clusters, and cloud computing environments.

The toolkit opera

This repository contains tools for generating and inspecting Dolma. To get started, install the Dolma Python library from [PyPI](https://pypi.org/project/dolma/).

```shell
pip install dolma
```

## Usage

To learn more about how to use Dolma, please visit the [documentation](/docs).

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
