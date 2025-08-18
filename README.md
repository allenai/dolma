<img alt="Dolma's official logo. It's dolma written in yellow, round lowercase letters over a blue background." src="https://raw.githubusercontent.com/allenai/dolma/main/docs/assets/AI2_Blog_1400x685_2x.webp" width="100%">

Dolma is two things:

1. **Dolma Dataset**: an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
2. **Dolma Toolkit**: a high-performance toolkit for curating datasets for language modeling -- this repo contains the source code for the Dolma Toolkit.

## Dolma Dataset

Dolma is an open dataset of 3 trillion tokens from a diverse mix of web content, academic publications, code, books, and encyclopedic materials.
It was created as a training corpus for [OLMo](https://allenai.org/olmo), a language model from the [Allen Institute for AI](https://allenai.org) (AI2).

Dolma is available for download on the HuggingFace 🤗 Hub: [`huggingface.co/datasets/allenai/dolma`](https://huggingface.co/datasets/allenai/dolma). Dolma is licensed under **[ODC-BY](https://opendatacommons.org/licenses/by/1-0/)**; see our [blog post](https://blog.allenai.org/making-a-switch-dolma-moves-to-odc-by-8f0e73852f44) for explanation.

You can also read more about Dolma in [our announcement](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64), as well as by consulting its [data sheet](docs/assets/dolma-v0_1-20230819.pdf).


## Dolma Toolkit

This repository houses the Dolma Toolkit, which enables curation of large datasets for (pre)-training ML models. Its key features are:

1. **High Performance** ⚡: Can process billions of documents concurrently thanks to built-in parallelism.
2. **Portability** 🧳: Works on a single machine, a cluster, or cloud environment.
3. **Built-In Taggers** 🏷: Includes ready-to-use taggers commonly used to curate datasets such as [Gopher](https://arxiv.org/abs/2112.11446), [C4](https://arxiv.org/abs/1910.10683), and [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/).
4. **Fast Deduplication** 🗑: Speedy document deduplication using a Rust Bloom filter.
5. **Extensibility** 🧩 & **Cloud Support** ☁: Supports custom taggers and AWS S3-compatible locations.

To install, simply type `pip install dolma` in your terminal.

To learn more about how to use the Dolma Toolkit, please visit the [documentation](/docs).

## Development

The Dolma Toolkit is a mixed Python/Rust project that uses [maturin](https://github.com/PyO3/maturin) for building. Here's how to set up a development environment:

### Using uv (Recommended)

If you're using [uv](https://docs.astral.sh/uv/), the easiest way to work with the codebase is:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/allenai/dolma.git
   cd dolma
   ```

2. **Run dolma commands during development:**
   ```bash
   # For regular use (caches builds)
   uv run dolma --version
   
   # After making Rust changes (rebuilds if needed)
   uv run --reinstall-package dolma dolma --version
   ```

3. **Create a development alias (optional):**
   ```bash
   alias dolma-dev="uv run --reinstall-package dolma dolma"
   # Then use: dolma-dev --version
   ```

### Using pip/conda

If you're not using uv:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/allenai/dolma.git
   cd dolma
   ```

2. **Install maturin and build dependencies:**
   ```bash
   pip install maturin[patchelf]  # Linux
   pip install maturin           # macOS/Windows
   ```

3. **Build and install in development mode:**
   ```bash
   # Initial build
   maturin develop --release
   
   # After making Rust changes, rebuild
   maturin develop --release
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -e .
   ```

### Version Management

This project uses a unified versioning system where:
- The version is defined in `Cargo.toml` (Rust)
- Python dynamically reads the version from the Rust extension
- Both `dolma --version` and `import dolma; dolma.__version__` show the same version

To update the version, simply change it in `Cargo.toml` and rebuild.

## Citation

If you use the Dolma dataset or toolkit, please cite the following items:

<!-- {% raw %} -->
```bibtex
@article{dolma,
  title = {{Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research}},
  author={Luca Soldaini and Rodney Kinney and Akshita Bhagia and Dustin Schwenk and David Atkinson and Russell Authur and Ben Bogin and Khyathi Chandu and Jennifer Dumas and Yanai Elazar and Valentin Hofmann and Ananya Harsh Jha and Sachin Kumar and Li Lucy and Xinxi Lyu and Nathan Lambert and Ian Magnusson and Jacob Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Abhilasha Ravichander and Kyle Richardson and Zejiang Shen and Emma Strubell and Nishant Subramani and Oyvind Tafjord and Pete Walsh and Luke Zettlemoyer and Noah A. Smith and Hannaneh Hajishirzi and Iz Beltagy and Dirk Groeneveld and Jesse Dodge and Kyle Lo},
  year={2024},
  journal={arXiv preprint},
  url={https://arxiv.org/abs/2402.00159}
}
```
<!-- {% endraw %} -->
