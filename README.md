<img alt="Dolma's official logo. It's dolma written in yellow, round lowercase letters over a blue background." src="https://raw.githubusercontent.com/allenai/dolma/main/docs/assets/AI2_Blog_1400x685_2x.webp" width="100%">

Dolma is two things:

1. **Dolma Dataset**: an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
2. **Dolma Toolkit**: a high-performance toolkit for curating datasets for language modeling -- this repo contains the source code for the Dolma Toolkit.

## Dolma Dataset

Dolma is an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
It was created as a training corpus for [OLMo](https://allenai.org/olmo), a language model from the [Allen Institute for AI](https://allenai.org) (AI2).

Dolma is available for download on the HuggingFace 🤗 Hub: [`huggingface.co/datasets/allenai/dolma`](https://huggingface.co/datasets/allenai/dolma). To access Dolma, users must agree to the terms of [AI2 ImpACT License for Medium Risk Artifacts](https://allenai.org/licenses/impact-mr) on the [HuggingFace 🤗 Hub](https://huggingface.co/datasets/allenai/dolma). Once agreed you can follow the instructions [here](https://huggingface.co/datasets/allenai/dolma#download) to download the dataset.

You can also read more about Dolma in [our announcement](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64), as well as by consulting its [data sheet](docs/assets/dolma-datasheet-v0.1.pdf).


## Dolma Toolkit

This repository houses the Dolma Toolkit, which enables curation of large datasets for (pre)-training ML models. Its key features are:

1. **High Performance** ⚡: Can process billions of documents concurrently thanks to built-in parallelism.
2. **Portabilty** 🧳: Works on a single machine, a cluster, or cloud environment.
3. **Built-In Taggers** 🏷: Includes ready-to-use taggers commonly used to curate datasets such as [Gopher](https://arxiv.org/abs/2112.11446), [C4](https://arxiv.org/abs/1910.10683), and [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/).
4. **Fast Deduplication** 🗑: Speedy document deduplication using a Rust Bloom filter.
5. **Extensibility** 🧩 & **Cloud Support** ☁: Supports custom taggers and AWS S3-compatible locations.

To install, simply type `pip install dolma` in your terminal.

To learn more about how to use the Dolma Toolkit, please visit the [documentation](/docs).

## Citation

If you use the Dolma dataset or toolkit, please cite the following items:

<!-- {% raw %} -->
```bibtex
@article{dolma,
  title = {{Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research}},
  author = {Luca Soldaini and Rodney Kinney and Akshita Bhagia and Dustin Schwenk and David Atkinson and Russell Authur and Ben Bogin and Khyathi Chandu and Jennifer Dumas and Yanai Elazar and Valentin Hofmann and Ananya Harsh Jha and Sachin Kumar and Li Lucy and Xinxi Lyu and Ian Magnusson and Jacob Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Abhilasha Ravichander and Kyle Richardson and Zejiang Shen and Emma Strubell and Nishant Subramani and Oyvind Tafjord and Evan Pete Walsh and Hannaneh Hajishirzi and Noah A. Smith and Luke Zettlemoyer and Iz Beltagy and Dirk Groeneveld and Jesse Dodge and Kyle Lo},
  year = {2023},
  journal={arXiv preprint},
}
```
<!-- {% endraw %} -->

