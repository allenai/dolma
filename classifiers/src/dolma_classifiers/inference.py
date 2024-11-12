import argparse
import multiprocessing as mp
import time
from collections import defaultdict
from functools import partial
from itertools import zip_longest
from multiprocessing import Event, Process
from queue import Queue as QueueType
from typing import Any, Generator, NamedTuple
from urllib.parse import urlparse
import traceback
from queue import Empty
import os 
import fsspec
import jq
import msgspec
import smart_open
import torch
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (  # pyright: ignore
    DataLoader,
    IterableDataset,
    get_worker_info,
)
from transformers import BatchEncoding, PreTrainedTokenizer

from .loggers import ProgressLogger, WandbLogger, get_logger
from .models import Prediction, Registry
from .utils import cleanup, get_local_gpu_rank, sanitize_model_name, setup
import math

class Batch(NamedTuple):
    encoding: BatchEncoding | dict[str, torch.Tensor]
    ids: list[str]
    lengths: list[int]
    sources: list[str]

    def __len__(self):
        return len(self.ids)


class OutputPath(NamedTuple):
    source: str
    count: int


class DocumentsIterableDataset(IterableDataset[Batch]):
    def __init__(
        self,
        input_paths_queue: QueueType[str],
        output_paths_queue: QueueType[OutputPath],
        tokenizer: PreTrainedTokenizer,
        max_length: int | None,
        text_selector: str = '.text',
        id_selector: str = ".id",
    ):
        self.input_paths_queue = input_paths_queue
        self.output_paths_queue = output_paths_queue

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

        text_selectors = [jq.compile(selector) for selector in self.text_selector.strip().split('\\n')]
        id_selector = jq.compile(self.id_selector)

        def format_text(text):

            return '\n'.join([str(selector.input(text).first()) for selector in text_selectors])

        try:
            while self.input_paths_queue.qsize() > 0:
                path = self.input_paths_queue.get()
                self.logger.info(f"Reading {path}")
                count = 0
                with smart_open.open(path, "rt") as source_file:
                    for line in source_file:
                        try:
                            doc = decoder.decode(line)
                            text = format_text(doc)
                            id_ = str(id_selector.input(doc).first())
                            encoding = self.tokenizer(
                                text,
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.max_length,
                            )
                            yield Batch(encoding=encoding, ids=[id_], lengths=[len(text)], sources=[path])
                            count += 1
                        except Exception as e:
                            self.logger.info(f"ERROR READING LINE {line}\n\n{e}\n{traceback.format_exc()}")

                self.logger.info(f"Read {count:,} documents from {path}")
                self.output_paths_queue.put(OutputPath(source=path, count=count))

        except Exception as e:
            self.logger.info(f"❌ Something went wrong processing {path}: {e}\n{traceback.format_exc()}")

            self.logger.error(f"❌ Something went wrong processing {path}: {e}\n{traceback.format_exc()}")
            
    


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
        lengths=[length for elem in batch for length in elem.lengths],
        sources=[source for elem in batch for source in elem.sources],
    )



class AttributeRow(NamedTuple):
    sources: list[str]
    attributes: list[dict[str, Any]]


def writer_worker(
    error_event: Event,
    scores_queue: QueueType[AttributeRow | None],
    output_paths_queue: QueueType[OutputPath],
    source_destination_mapping: dict[str, str],
    error_queue: mp.Queue,
    log_every: int = 10,
):

    progress_logger = ProgressLogger(log_every=log_every, wandb_logger=WandbLogger())
    console_logger = get_logger("writer_worker")

    files_writers = {}
    try:
        encoder = msgspec.json.Encoder()
        counts = defaultdict(int)
        total_count = 0

        while True:
            if scores_queue.qsize() == 0:
                time.sleep(0.1)
                continue

            element = scores_queue.get()
            if element is None:
                break

            group_by_source = defaultdict(list)
            for source, attribute in zip(element.sources, element.attributes):
                group_by_source[source].append(attribute)
                if source not in files_writers:
                    destination_path = source_destination_mapping[source]
                    files_writers[source] = smart_open.open(destination_path, "wt", encoding="utf-8")
                    console_logger.info(f"Opened {destination_path} for writing")

            for source, attributes in group_by_source.items():
                files_writers[source].write(
                    encoder.encode_lines(attributes).decode("utf-8")
                )
                progress_logger.increment(docs=len(attributes))
                counts[source] += len(attributes)
                total_count += len(attributes)

            if total_count > log_every:
                # we at most close one file per log_every documents
                try:
                    # get the paths from the output queue (these have been fully processed)
                    path = output_paths_queue.get_nowait()
                except Empty:
                    path = None
                    console_logger.info(f"No paths to close.")


                if path is not None and path.count == counts[path.source]:
                    # I've finished processing this source; close the file
                    f = files_writers.pop(path.source)
                    f.close()
                    console_logger.info(f"Closed {source_destination_mapping[path.source]}")
                    progress_logger.increment(files=1)
                elif path is not None and counts[path.source] > path.count:
                    console_logger.info(
                        f"More documents ({counts[path.source]}) than expected ({path.count}) " +
                        f"for source {path.source}. This should not happen!")
                    raise RuntimeError(
                        f"More documents ({counts[path.source]}) than expected ({path.count}) " +
                        f"for source {path.source}. This should not happen!"
                    )
                elif path is not None:
                    console_logger.info(
                        f"Tried to close {source_destination_mapping[path.source]}, " +
                        f"but only seen {counts[path.source]}/{path.count} documents"
                    )
                    # more documents still to be written for this source; put it back
                    output_paths_queue.put(path)
                total_count = 0
    except Exception as e:
        console_logger.error(f"Writer process encountered an error: {e}")
        console_logger.info(f"Writer process encountered an error: {e} {traceback.format_exc}")

        error_event.set()
        error_traceback = traceback.format_exc()
        error_queue.put(error_traceback)
    finally:
        for f in files_writers.values():
            f.close()


def process_documents(
    source_paths: list[str],
    destination_paths: list[str],
    batch_size: int,
    model_name: str,
    model_dtype: str,
    model_compile: bool,
    log_every: int,
    max_length: int | None = None,
    text_selector: str | None = None,
    id_selector: str = ".id",
    num_workers: int = 10,
    prefetch_factor: int = 2,
    suffix: str | None = None,
    rank: int = 0 
):
    """Processes a batch of files using distributed processing."""
    console_logger = get_logger("process_documents")

    console_logger.info(f"Rank is : {rank}")
    classifier = Registry.get(
        model_name=model_name,
        device=f'cuda:{rank}',
        dtype='float16',
        compile=model_compile,
    )

    if len(source_paths) <= 0 :
        return

    if not text_selector:
        text_selector = classifier.input_template
    # get filesystem for first source path (we assume is the same for all source paths); we will use this
    # to check if destination path exists (file already processed)
    fs = fsspec.get_filesystem_class(urlparse(source_paths[0]).scheme)()

    # this encoder will be used to write the attributes to the destination file
    encoder = msgspec.json.Encoder()


    source_destination_mapping = {
        source_path: destination_path
        for source_path, destination_path in zip(source_paths, destination_paths)
        if not fs.exists(destination_path)
    }
    with torch.no_grad(), mp.Manager() as manager:
        input_paths_queue: QueueType[str] = manager.Queue()
        output_paths_queue: QueueType[OutputPath] = manager.Queue()
        scores_queue: QueueType[AttributeRow | None] = manager.Queue()
        error_queue: mp.Queue = manager.Queue()
        
        for source_path in source_destination_mapping:
            input_paths_queue.put(source_path)

        writer_process_error = Event()
        writer_process = Process(
            target=writer_worker,
            kwargs=dict(
                scores_queue=scores_queue,
                output_paths_queue=output_paths_queue,
                source_destination_mapping=source_destination_mapping,
                log_every=log_every,
                error_event=writer_process_error,
                error_queue=error_queue,        
            ),
        )
        writer_process.start()
        try:
            source_dataset = DocumentsIterableDataset(
                # path=source_path,
                input_paths_queue=input_paths_queue,
                output_paths_queue=output_paths_queue,
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
                prefetch_factor=prefetch_factor,
                collate_fn=partial(collate_batch, pad_token_id=getattr(classifier.tokenizer, "pad_token_id", 0)),
            )

            counts = defaultdict(int)
            tracebacks = []
            for batch in data_loader:
                for s in batch.sources:
                    counts[s] += 1

                if writer_process_error.is_set():
                    try:
                        error_traceback = error_queue.get_nowait()
                        tracebacks.append(error_traceback)
                        console_logger.info(f"Writer process error traceback:\n{error_traceback}")
                    except Empty:
                        pass
                    raise RuntimeError("Writer process encountered an error")

                inputs = {k: v.to(classifier.device) for k, v in batch.encoding.items()}
                scores = classifier.score(**inputs)
                attributes = [
                    {"id": doc_id, "attributes": {pred.label: [[pred.score]] for pred in doc_preds}}
                    for doc_preds, doc_id, doc_length in zip(scores, batch.ids, batch.lengths)
                ]
                scores_queue.put_nowait(AttributeRow(sources=batch.sources, attributes=attributes))

            scores_queue.put(None)
        except Exception as e:
            console_logger.info(f"Something went wrong in writer loop: {e} {tracebacks}")
        finally:
            writer_process.join()
            if writer_process_error.is_set():
                raise RuntimeError("Writer process encountered an error")
    cleanup()


def longest_common_sequence(paths: list[str]) -> str:
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

    console_logger.info(f"source paths found: {len(source_paths)}")
    if all("/documents/" in p for p in source_paths):
        source_prefix = longest_common_sequence([p.split("/documents/", 1)[0] for p in source_paths])
        source_prefix = f"{source_prefix}/documents/"
    else:
        source_prefix = longest_common_sequence(source_paths)

    destination_paths = [
        f'{args.output_prefix.rstrip("/")}/{p.replace(source_prefix, "").lstrip("/")}' for p in source_paths
    ]
    console_logger.info(f"destiantion paths: {len(destination_paths)}")

    console_logger.info(f"Processing up to {len(source_paths)} files from {args.source_prefix} to {args.output_prefix}")

    # Filter out existing files unless --override is set
    if not args.override:

        # possible existing destinations might contain more files than destination_paths because it glob
        # at the attribute name level, while destination_paths might only be about a subset of documents.
        possible_existing_destinations = set(f"{scheme}://{p}" for p in fs.glob(f'{args.output_prefix.rstrip("/")}/**'))
        existing_destinations = {p for p in destination_paths if p in possible_existing_destinations}

        console_logger.info(f"Found {len(existing_destinations)} existing files in {args.output_prefix}")

        if len(existing_destinations) >= len(source_paths):
            console_logger.info("No files left to process, exiting")
            return

        source_paths, destination_paths = map(
            lambda t: list(t),
            zip(*[(p, d) for p, d in zip(source_paths, destination_paths) if d not in existing_destinations]),
        )

    console_logger.info(f"After filtering, collectively tagging {len(source_paths)} files")

    # Distribute files across processes
    files_per_process = len(source_paths) / world_size
    start_idx = int(rank * files_per_process)
    end_idx = int((rank + 1) * files_per_process) if rank < world_size - 1 else len(source_paths)
    partition_source_paths = source_paths[start_idx:end_idx]
    partition_destination_paths = destination_paths[start_idx:end_idx]

    console_logger.info(f"Partitioned into {world_size} workers of with avg {files_per_process:.2f} files.")
    console_logger.info(f"GPU {rank}/{world_size} processing {len(partition_source_paths)} files from index {start_idx} to {end_idx}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    chunk_size = 8
    n_chunks = math.ceil(len(partition_source_paths) / chunk_size)
    actual_chunk_size = math.ceil(len(partition_source_paths) / n_chunks)

    source_chunks = [
        partition_source_paths[i:i + actual_chunk_size]
        for i in range(0, len(partition_source_paths), actual_chunk_size)
    ]
    destination_chunks = [
        partition_destination_paths[i:i + actual_chunk_size]
        for i in range(0, len(partition_destination_paths), actual_chunk_size)
    ]
    
    for source_chunk,destination_chunk in zip(source_chunks,destination_chunks):

        process_documents(
            model_name=args.model_name,
            model_dtype=args.model_dtype,
            log_every=args.log_every,
            source_paths=source_chunk,
            destination_paths=destination_chunk,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            text_selector=args.text_key,
            id_selector=args.id_key,
            suffix=args.attribute_suffix,
            model_compile=args.model_compile,
            prefetch_factor=args.prefetch_factor,
            rank=rank
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
    parser.add_argument("--model-compile", action="store_true", help="Compile the model using torch.compile")
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
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor for DataLoader")
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
