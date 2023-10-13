<img alt="Dolma's official logo. It's dolma written in yellow, round lowercase letters over a blue background." src="https://raw.githubusercontent.com/allenai/dolma/main/docs/assets/AI2_Blog_1400x685_2x.webp" width="100%">

Dolma is two things:

1. **Dolma Dataset**: an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
2. **Dolma Toolkit**: a high-performance toolkit for curating datasets for language modeling.

## Dolma Dataset

Dolma is an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
It was created as a training corpus for [OLMo](https://allenai.org/olmo), a language model from the [Allen Institute for AI](https://allenai.org) (AI2).

Dolma is available for download on the HuggingFace 🤗 Hub: [`huggingface.co/datasets/allenai/dolma`](https://huggingface.co/datasets/allenai/dolma). To access Dolma, users must agree to the terms of the terms of [AI2 ImpACT License for Medium Risk Artifacts](https://allenai.org/licenses/impact-mr).

You can also read more about Dolma in [our announcement](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64), as well as by consulting its [data sheet](docs/assets/dolma-datasheet-v0.1.pdf).


## Dolma Toolkit

Dolma is a toolkit to curate large datasets for (pre)-training ML models. Its key features are:

1. **High Performance** ⚡: Can process billions of documents concurrently thanks to built-in parallelism.
2. **Portabilty** 🧳: Works on a single machine, a cluster, or cloud environment.
3. **Built-In Taggers** 🏷: Includes ready-to-use taggers commonly used to curate datasets such as [Gopher](https://arxiv.org/abs/2112.11446), [C4](https://arxiv.org/abs/1910.10683), and [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/).
4. **Fast Deduplication** 🗑: Speedy document deduplication using a Rust Bloom filter.
5. **Extensibility** 🧩 & **Cloud Support** ☁: Supports custom taggers and AWS S3-compatible locations.

To install, simply type `pip install dolma` in your terminal.

To learn more about how to use the Dolma Toolkit, please visit the [documentation](/docs).

## Citation

If you use the Dolma dataset or toolkit, please cite the following items:

```bibtex
@techreport{DolmaDataset,
    author = {Soldaini, Luca and Kinney, Rodney and Bhagia, Akshita and Schwenk, Dustin and Atkinson, David and Authur, Russell and Chandu, Khyathi and Dumas, Jennifer and Lucy, Li and Lyu, Xinxi and Magnusson, Ian and Naik, Aakanksha and Nam , Crystal and  Peters, Matthew E.  and Ravichander, Abhilasha and Shen, Zejiang and Strubell, Emma and Subramani, Nishant and Tafjord, Oyvind and Walsh, Evan Pete and Hajishirzi, Hannaneh and Smith, Noah A. and Zettlemoyer, Luke and Beltagy, Iz and Groeneveld, Dirk and Dodge, Jesse and Lo, Kyle},
    title = {{Dolma: An Open Corpus of 3 Trillion Tokens for Language Model Pretraining Research}},
    institution = {{Allen Institute for AI}},
    year = {2023},
    note = {Released under ImpACT License as Medium Risk artifact, \url{https://github.com/allenai/dolma}}
}
```

```bibtex
@software{DolmaToolkit,
    author = {{Soldaini, Luca and Lo, Kyle and Kinney, Rodney and Naik, Aakanksha and Ravichander, Abhilasha and Bhagia, Akshita and Groeneveld, Dirk and Schwenk, Dustin and Magnusson, Ian and Chandu, Khyathi}},
    title = {{The Dolma Toolkit}},
    year = {2023},
    note = {{Apache 2.0 License, Version \texttt{0.9.0}, \url{https://github.com/allenai/dolma}}}
}
```
