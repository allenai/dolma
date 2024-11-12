"""
Script for preparing the Tulu data for fine-tuning an OLMo model.

python scripts/tokenize_sft_dataset.py \
    --tokenizer.name_or_path allenai/dolma2-tokenizer \
    --tokenizer.bos_token_id 100257 \
    --tokenizer.eos_token_id 100257 \
    --tokenizer.pad_token_id 100277 \
    --dataset.path allenai/tulu-v3.9-tmp

"""

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

from dolma.tokenizer.tokenizer import Tokenizer
from dolma.cli.tokenizer import TokenizerConfig
from dolma.cli import field, BaseCli


@dataclass
class DatasetConfig:
    path: str | None = field(default=None, help="Path or name of the dataset. Required.")
    name: str | None = field(default=None, help="Defining the name of the dataset configuration.")
    split: str | None = field(default='train', help="Name of the split to load.")


@dataclass
class TokenizationConfig:
    tokenizer: TokenizerConfig = field(default=TokenizerConfig(), help="Configuration for the tokenizer.")
    dataset : DatasetConfig = field(default=DatasetConfig(), help="Configuration for the dataset.")
    processes: int = field(default=1, help="Number of parallel processes to use.")
    output_dir: str = field(help="Output directory to save the tokenized data.")
    max_seq_len: int = field(default=4096, help="Maximum sequence length.")
    max_label_len: int | None = field(default=None, help="Maximum label length.")
    dtype: None | str = field(default=None, help="Data type for the tokenized data.")
    max_tokens_per_file: int = field(default=2 ** 32, help="Maximum number of tokens per file.")


def run_tokenizer(opts: TokenizationConfig) -> None:
    assert opts.tokenizer is not None, "Tokenizer configuration is missing."
    assert opts.tokenizer.name_or_path is not None, "Tokenizer name or path must be provided."
    assert getattr(opts, "output_dir", None) is not None, "Output directory is missing."

    opts.max_label_len = opts.max_label_len or opts.max_seq_len

    tokenizer_config = {}
    if opts.tokenizer.bos_token_id is not None:
        tokenizer_config["bos_token_id"] = opts.tokenizer.bos_token_id
    if opts.tokenizer.eos_token_id is not None:
        tokenizer_config["eos_token_id"] = opts.tokenizer.eos_token_id
    if opts.tokenizer.pad_token_id is not None:
        tokenizer_config["pad_token_id"] = opts.tokenizer.pad_token_id

    if Path(opts.tokenizer.name_or_path).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer.name_or_path, **tokenizer_config)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer.name_or_path, **tokenizer_config)

    expected_bits = int(np.ceil(np.log2(tokenizer.vocab_size) / 16)) * 16
    expected_dtype = f"uint{expected_bits}"

    if opts.dtype is not None and opts.dtype != expected_dtype:
        raise ValueError(f"Invalid data type, expected: {expected_dtype}, got: {opts.dtype}")
    elif opts.dtype is None:
        np_dtype = getattr(np, expected_dtype)
    else:
        np_dtype = getattr(np, opts.dtype)

    assert opts.dataset is not None, "Dataset configuration is missing."
    assert opts.dataset.path is not None, "Dataset path is missing."

    dataset_config = {}
    if opts.dataset.name is not None:
        dataset_config["name"] = opts.dataset.name
    if opts.dataset.split is not None:
        dataset_config["split"] = opts.dataset.split

    dataset = ds.load_dataset(opts.dataset.path, **dataset_config)

    # # sample 10k
    # dataset = dataset.shuffle(seed=42).select(range(10000))

    print("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.max_seq_len),
        batched=False,
        remove_columns=dataset.column_names,  # type: ignore
        num_proc=opts.processes,    # type: ignore
        desc="Tokenizing dataset",  # type: ignore
    )

    print("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(
        partial(filter_long_sequences, max_label_len=opts.max_label_len, max_seq_len=opts.max_seq_len),
        batched=False,
        num_proc=opts.processes,
        desc="Filtering sequences that are too long",
    )
    print(f"Filtered out {n - len(dataset):,d} examples")

    print(f"Saving results to '{opts.output_dir}'...")
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    total_tokens = len(dataset) * opts.max_seq_len
    batch_size = int(np.floor((opts.max_tokens_per_file / total_tokens) * len(dataset)))
    print(f"Saving {len(dataset):,d} examples to {output_dir} in batches of {batch_size:,d} examples")

    dataset.map(
        partial(save_memmap, output_dir=output_dir, batch_size=batch_size, dtype=np_dtype),
        batched=True,
        batch_size=batch_size,
        num_proc=opts.processes,
        desc="Saving memmaps",
        remove_columns=dataset.column_names,  # type: ignore
        with_indices=True,
    )


def save_memmap(
    data: dict[str, list],
    idx: list[int],
    output_dir: Path,
    batch_size: int,
    dtype: np.dtype
) -> dict[str, list]:
    output_dir.mkdir(exist_ok=True, parents=True)

    pos = idx[0] // batch_size
    size = sum(len(input_ids) for input_ids in data["input_ids"])
    input_ids_mm = np.memmap(output_dir / f"input_ids_{pos:06d}.npy", dtype=dtype, mode="w+", shape=(size,))
    label_mask_mm = np.memmap(output_dir / f"label_mask_{pos:06d}.npy", dtype=np.bool_, mode="w+", shape=(size,))

    offset = 0
    for input_ids, label_mask in zip(data["input_ids"], data["label_mask"]):
        n = len(input_ids)
        input_ids_mm[offset : offset + n] = input_ids
        label_mask_mm[offset : offset + n] = label_mask
        offset += n

    input_ids_mm.flush()
    label_mask_mm.flush()

    return {}


def filter_long_sequences(example: dict, max_label_len: int = 2 ** 30, max_seq_len: int = 2 ** 30) -> bool:
    return (
        example["n_labels"] > 0
        and example["n_labels"] <= max_label_len
        and example["n_total"] <= max_seq_len
    )


def preprocess(example: dict, tokenizer: Tokenizer, max_seq_len: int) -> dict:
    eos_token = tokenizer.base_tokenizer.id_to_token(tokenizer.eos_token_id)

    input_ids = [tokenizer.bos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    n_total = len(input_ids)

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len
    elif len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        label_mask = label_mask[:max_seq_len]

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels, "n_total": n_total}


class SftTokenizerCli(BaseCli):
    CONFIG = TokenizationConfig
    DESCRIPTION = "Tokenize the Tulu V2 SFT dataset."

    @classmethod
    def run(cls, parsed_config: TokenizationConfig):
        run_tokenizer(parsed_config)


if __name__ == "__main__":
    parser = SftTokenizerCli.make_parser(ArgumentParser())
    SftTokenizerCli.run_from_args(parser.parse_args())
