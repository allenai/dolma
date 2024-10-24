"""
Author: Luca Soldaini (@soldni)
Email:  lucas@allenai.org

To run this script with torchrun, use the following command:
torchrun --nproc_per_node=<num_gpus> scripts/fineweb_classifier.py \
    --source-prefix s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/documents/dclm-1969.json.zst \
    --output-prefix s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/attributes/fineweb-edu-classifier
    --batch-size 512 # 128 on A6000

Replace <num_gpus> with the number of GPUs you want to use.
"""

import argparse
from functools import partial
from hashlib import md5
from collections import abc
import multiprocessing as mp
import os
import logging
import re
import time
from itertools import chain
from math import ceil
from itertools import zip_longest
from queue import Queue as QueueType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
)
from urllib.parse import urlparse

import msgspec
import fsspec
import smart_open
from smart_open.compression import _handle_zstd
import tqdm

import torch  # pyright: ignore
from torch.utils.data import IterableDataset, DataLoader, get_worker_info  # pyright: ignore
from torch.nn.utils.rnn import pad_sequence

from transformers import BatchEncoding, PreTrainedTokenizer
import wandb
import jq

from .models import Registry
from .utils import setup, cleanup, get_local_gpu_rank, sanitize_model_name
from .loggers import get_logger, WandbLogger, ProgressLogger


class Document(NamedTuple):
    id: str
    text: str


class Batch(NamedTuple):
    encoding: BatchEncoding | dict[str, torch.Tensor]
    ids: list[str]
    length: list[int]

    def __len__(self):
        return len(self.ids)


class DocumentsIterableDataset(IterableDataset[Batch]):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int | None,
        text_selector: str = '.text',
        id_selector: str = ".id",
    ):
        self.queue: QueueType[Document | None] = mp.Queue()  # pyright: ignore
        self.path = path
        self.text_selector = text_selector
        self.id_selector = id_selector
        self.tokenizer = tokenizer
        self.logger = get_logger(self.__class__.__name__)
        self.max_length = max_length or int(tokenizer.model_max_length)

    @property
    def worker_info(self):
        worker_rank = 0
        world_size = 1
        if (worker_info := get_worker_info()):
            worker_rank = worker_info.id
            world_size = worker_info.num_workers
        return worker_rank, world_size

    def __iter__(self) -> Generator[Batch, None, None]:
        decoder = msgspec.json.Decoder()
        text_selector = jq.compile(self.text_selector)
        id_selector = jq.compile(self.id_selector)

        self.logger.info(f"Reading {self.path}")
        count = 0
        worker_rank, world_size = self.worker_info

        with smart_open.open(self.path, "rt") as source_file:
            for i, line in enumerate(source_file):
                if i % world_size != worker_rank:
                    # skip lines that are not assigned to this worker
                    continue

                doc = decoder.decode(line)
                text = str(text_selector.input(doc).first())
                id_ = str(id_selector.input(doc).first())
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                yield Batch(encoding=encoding, ids=[id_], length=[len(text)])
                count += 1

        self.logger.info(f"Read {count:,} documents from {self.path}")



def collate_batch(batch: list[Batch], pad_token_id: int) -> Batch:
    max_lengths = [len(b.encoding['input_ids'][0]) for b in batch]  # pyright: ignore
    padded_encodings = {
        key: pad_sequence(
            # assuming first dimension is batch size
            [b.encoding[key][-1,:] for b in batch],   # pyright: ignore
            batch_first=True,
            padding_value=pad_token_id,
        )
        for key in batch[0].encoding.keys()
    }
    return Batch(
        encoding=padded_encodings,
        ids=[id_ for elem in batch for id_ in elem.ids],
        length=[length for elem in batch for length in elem.length],
    )



def process_documents(
    source_paths: list[str],
    destination_paths: list[str],
    batch_size: int,
    model_name: str,
    model_dtype: str,
    log_every: int,
    max_length: int | None = None,
    text_selector: str = ".text",
    id_selector: str = ".id",
    num_workers: int = 1,
    suffix: str | None = None
):
    """Processes a batch of files using distributed processing."""
    console_logger = get_logger("process_documents")


    classifier = Registry.get(
        model_name=model_name,
        device=f'cuda:{get_local_gpu_rank()}',
        dtype='float16'
    )

    # get filesystem for first source path (we assume is the same for all source paths); we will use this
    # to check if destination path exists (file already processed)
    fs = fsspec.get_filesystem_class(urlparse(source_paths[0]).scheme)()

    wandb_logger = WandbLogger()
    progress_logger = ProgressLogger(log_every=log_every, wandb_logger=wandb_logger)

    # this encoder will be used to write the attributes to the destination file
    encoder = msgspec.json.Encoder()

    for source_path, destination_path in zip(source_paths, destination_paths):
        if fs.exists(destination_path):
            console_logger.info(f"Skipping {source_path} because {destination_path} already exists")
            continue

        with torch.no_grad(), smart_open.open(destination_path, "wt") as destination_file:
            source_dataset = DocumentsIterableDataset(
                path=source_path,
                tokenizer=classifier.tokenizer,
                max_length=max_length,
                text_selector=text_selector,
                id_selector=id_selector,
            )

            data_loader = DataLoader(
                source_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=partial(collate_batch, pad_token_id=getattr(classifier.tokenizer, "pad_token_id", 0)),
            )

            for batch in data_loader:
                inputs = {k: v.to(classifier.device) for k, v in batch.encoding.items()}
                scores = classifier.score(**inputs)

                attributes = [
                    {"id": doc_id, "attributes": {pred.label: [0, doc_length, pred.score] for pred in doc_preds}}
                    for doc_preds, doc_id, doc_length in zip(scores, batch.ids, batch.length)
                ]
                progress_logger.increment(docs=len(batch))
                destination_file.write(encoder.encode_lines(attributes).decode("utf-8"))
        progress_logger.increment(files=1)
    cleanup()


def longest_common_sequence(paths: List[str]) -> str:
    # Split each string by "/"
    split_strings = [s.split("/") for s in paths]

    # Zip the split lists together and find the longest common sequence
    common_sequence = []
    for fragments in zip_longest(*split_strings, fillvalue=None):
        # Check if all fragments in this position are the same
        if len(set(fragments)) == 1:
            common_sequence.append(fragments[0])
        else:
            break

    # Join the longest common sequence back with "/"
    return "/".join(common_sequence)


def main(args: argparse.Namespace) -> None:
    # disable multiprocessing for tokenizer
    console_logger = get_logger("main")

    # initialize distributed processing
    rank, world_size = setup()

    # initialize wandb logging (if enabled)
    WandbLogger()

    # check for available GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs available, but the script is designed to use multiple GPUs.")

    # if necessary, unglob source prefix
    fs = fsspec.get_filesystem_class((scheme := urlparse(args.source_prefix).scheme))()
    source_paths = [(f"{scheme}://{p}" if scheme else p) for p in fs.glob(args.source_prefix)]

    assert len(source_paths) > 0, f"No files found in {args.source_prefix}"

    print(f"Tagging {len(source_paths)} files from {args.source_prefix} to {args.output_prefix}")

    if all("/documents/" in p for p in source_paths):
        source_prefix = longest_common_sequence([p.split("/documents/", 1)[0] for p in source_paths])
        source_prefix = f"{source_prefix}/documents/"
    else:
        source_prefix = longest_common_sequence(source_paths)

    destination_paths = [
        f'{args.output_prefix.rstrip("/")}/{p.replace(source_prefix, "").lstrip("/")}' for p in source_paths
    ]

    # Filter out existing files unless --override is set
    if not args.override:
        existing_destinations = set(f"{scheme}://{p}" for p in fs.glob(f'{args.output_prefix.rstrip("/")}/**'))
        source_paths, destination_paths = map(
            lambda t: list(t),
            zip(*[(p, d) for p, d in zip(source_paths, destination_paths) if d not in existing_destinations]),
        )

    # Distribute files across processes
    files_per_process = len(source_paths) / world_size
    start_idx = int(rank * files_per_process)
    end_idx = int((rank + 1) * files_per_process) if rank < world_size - 1 else len(source_paths)
    partition_source_paths = source_paths[start_idx:end_idx]
    partition_destination_paths = destination_paths[start_idx:end_idx]

    console_logger.info(f"Partitioned into {world_size} workers of with avg {files_per_process:.2f} files.")
    console_logger.info(f"Processing GPU {rank}/{world_size}: {len(partition_source_paths)} files")

    process_documents(
        model_name=args.model_name,
        model_dtype=args.model_dtype,
        log_every=args.log_every,
        source_paths=partition_source_paths,
        destination_paths=partition_destination_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        text_selector=args.text_key,
        id_selector=args.id_key,
        suffix=args.attribute_suffix,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify text from JSONL files on S3 using a Hugging Face model."
    )
    parser.add_argument(
        "-s",
        "--source-prefix",
        type=str,
        required=True,
        help="S3 glob pattern for input files (e.g., s3://path/to/docs/*/*.jsonl.gz)",
    )
    parser.add_argument("--output-prefix", type=str, default=None, help="S3 prefix to save the results")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument("-m", "--model-name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument(
        "--max-length", type=int, default=None, help="Maximum sequence length for tokenization (default: None)"
    )
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Gantry task name")
    parser.add_argument("--override", action="store_true", help="Override existing files")
    parser.add_argument("--text-key", type=str, default=".text", help="JQ key to extract text from documents")
    parser.add_argument("--id-key", type=str, default=".id", help="JQ key to extract id from documents")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for processing")
    parser.add_argument("--log-every", type=int, default=10000, help="Log every n documents")
    parser.add_argument("--model-dtype", type=str, default="float16", help="Data type for model")
    parser.add_argument("--attribute-suffix", type=str, default=None, help="Optional suffix for attribute keys")
    opts = parser.parse_args()

    if opts.output_prefix is None:
        if "/documents/" not in opts.source_prefix:
            raise ValueError("Output prefix is required unless source prefix contains 'documents'")
        base, _ = opts.source_prefix.split("/documents/", 1)
        opts.output_prefix = f"{base}/attributes/{sanitize_model_name(opts.model_name)}"

    if opts.use_wandb:
        WandbLogger.use_wandb = True
        WandbLogger.project = opts.wandb_project or WandbLogger.project
        WandbLogger.entity = opts.wandb_entity or WandbLogger.entity
        # use name provided by user, or name of run in wandb, or sanitize model name
        WandbLogger.name = opts.wandb_name or WandbLogger.name or sanitize_model_name(opts.model_name, opts.__dict__)

    return opts


if __name__ == "__main__":
    args = parse_args()
    main(args)
