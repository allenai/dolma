# Tokenization Tool

The Dolma toolkit comes with a command line tool for tokenizing documents. The tool can be accessed using the `dolma tokens` command.

The tokenizer is optimized processing large datasets that are split over multiple files. If you are using the dolma toolkit to curate a dataset, this is achieved via the `dolma mix` command.

The Dolma tokenizer tool can use any [HuggingFace-compatible tokenizer](https://huggingface.co/docs/tokenizers/index).

The library employs the following strategy to provide a light shuffling of the data:

- First, paths to input files are shuffled.
- Each parallel process in the tokenizer opens N files in parallel for tokenization.
- The process reads a chunk k of documents, equally divided between the N files (i.e., k/N documents per file).
- The process shuffles the documents in the chunk.
- The process writes the output.

The tokenization library outputs to files: a `.npy` file containing the concatenated tokenized documents, and a `.csv.gz` file containing the metadata for each tokenized document. The metadata file contains the following columns:

- `start`: the start index of the document in the `.npy` file.
- `end`: the end index of the document in the `.npy` file.
- `path`: the path to the original document.
- `id`: the id of the original document.

## Parameters

The following parameters are supported either via CLI (e.g. `dolma tokens --parameter.name value`) or via config file (e.g. `dolma -c config.json tokens`, where `config.json` contains `{"parameter" {"name": "value"}}`):

|Parameter|Required?|Description|
|:---:|---|---|
|`documents`|Yes| One or more paths for input document files. Paths can contain arbitrary wildcards. Can be local, or an S3-compatible cloud path. |
|`destination`|Yes| One or more paths for output files. Should match number of `documents` paths. Can be local, or an S3-compatible cloud path. |
|`tokenizer.name_or_path`|Yes| Name or path of the tokenizer to use. Must be a HuggingFace-compatible tokenizer. |
| `tokenzier.bos_token_id`| Yes if `tokenizer.eos_token_id` is missing | The id of the beginning-of-sequence token. |
| `tokenizer.eos_token_id`| Yes if `tokenizer.bos_token_id` is missing | The id of the end-of-sequence token. |
| `tokenizer.pad_token_id`| No | The id of the padding token. |
| `tokenizer.segment_before_tokenization`| No | Whether to segment documents by paragraph before tokenization. This is useful for tokenizers like Llama that are very slow on long documents. Might not be needed once [this bugfix is merged](https://github.com/huggingface/tokenizers/pull/1413). Defaults to False.|
|`processes`|No| Number of processes to use for tokenization. By default 1 process is used. |
|`files_per_process`|No| Maximum number of files per tokenization process. By default, only one file is processed. This controls the number of output files generated. |
|`batch_size`|No| Number of k sequences to tokenize and shuffle before writing to disk. By default, k=10000. |
|`ring_size`|No| Number of N files to open in parallel for tokenization. By default, N=8. |
|`max_size`|No| Maximum size of a file in bytes. By default, 1GB. |
|`dtype`|No| Data type for the memmap file; must be a valid numpy dtype. By default, `uint16`. |
|`work_dir.input`|No| Path to a local scratch directory where temporary input files can be placed. If not provided, Dolma will make one for you and delete it upon completion. |
|`work_dir.output`|No| Path to a local scratch directory where temporary output files can be placed. If not provided, Dolma will make one for you and delete it upon completion. |
|`dryrun`|No| If true, only print the configuration and exit without running the tokenizer. |
|`seed`|No| Seed for random number generation. |
