<img alt="Dolma's official logo. It's dolma written in yellow, round lowercase letters over a blue background." src="https://raw.githubusercontent.com/allenai/dolma/main/docs/assets/AI2_Blog_1400x685_2x.webp" width="100%">

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

Dolma 包含两个核心部分：

1. **Dolma 数据集（Dolma Dataset）**：一个规模达 3 万亿（3 Trillion）词元（Token）的开源数据集，广泛汇集了网页内容、学术论文、代码、书籍以及百科资料等多种来源的数据。
2. **Dolma 工具包（Dolma Toolkit）**：一个用于构建与清洗语言模型训练数据集的高性能工具包——本仓库包含 Dolma 工具包的源代码。

## Dolma 数据集

Dolma 是一个规模达 3 万亿词元的开源数据集，涵盖网页内容、学术论文、代码、书籍以及百科资料等多样化语料。
它最初作为[艾伦人工智能研究所（Allen Institute for AI, AI2）](https://allenai.org)推出的开源大语言模型 [OLMo](https://allenai.org/olmo) 的预训练语料库而构建。

Dolma 可在 HuggingFace 🤗 Hub 上直接下载：[`huggingface.co/datasets/allenai/dolma`](https://huggingface.co/datasets/allenai/dolma)。Dolma 采用 **[ODC-BY](https://opendatacommons.org/licenses/by/1-0/)** 许可证发布；有关许可证变更的详细说明请参阅我们的[技术博客](https://blog.allenai.org/making-a-switch-dolma-moves-to-odc-by-8f0e73852f44)。

此外，您还可以通过阅读我们的[官方发布公告](https://blog.allenai.org/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64)或查阅其[数据说明书（Data Sheet）](docs/assets/dolma-v0_1-20230819.pdf)进一步了解 Dolma。


## Dolma 工具包

本仓库托管了 Dolma 工具包，旨在支持大规模机器学习模型（预）训练数据集的高效清洗与构建。其核心特性包括：

1. **高性能 (High Performance)** ⚡：得益于内置的并发与并行机制，能够同时高效处理数十亿篇文档。
2. **高可移植性 (Portability)** 🧳：支持在单机、计算集群或云端环境中无缝运行。
3. **内置标注器 (Built-In Taggers)** 🏷：开箱即用地内置了在构建 [Gopher](https://arxiv.org/abs/2112.11446)、[C4](https://arxiv.org/abs/1910.10683) 和 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) 等经典数据集时常用的特征标注器（Taggers）。
4. **极速去重 (Fast Deduplication)** 🗑：基于 Rust 实现的布隆过滤器（Bloom filter），提供极速的文档级与段落级去重能力。
5. **高可扩展性 (Extensibility)** 🧩 与**云存储支持 (Cloud Support)** ☁：支持编写自定义标注器，并原生支持读写 AWS S3 及兼容的对象存储路径。

安装非常简单，只需在终端中运行 `pip install dolma` 即可。

如需进一步了解如何使用 Dolma 工具包，请访问[官方文档](/docs)。

## 引用 (Citation)

如果您在研究或项目中使用了 Dolma 数据集或工具包，请引用以下文献：

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

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
