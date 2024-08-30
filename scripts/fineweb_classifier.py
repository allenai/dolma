"""
Author: Luca Soldaini (@soldni)
Email:  lucas@allenai.org

To run this script with torchrun, use the following command:
torchrun --nproc_per_node=2 scripts/fineweb_classifier.py \
    --source-prefix s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/documents/dclm-1969.json.zst \
    --output-prefix s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/attributes/fineweb-edu-classifier
    --batch-size 512 # 128 on A6000

Replace <num_gpus> with the number of GPUs you want to use.
"""

import argparse
import multiprocessing as mp
import os
import time
from functools import partial
from itertools import zip_longest
from queue import Queue as QueueType
from typing import TYPE_CHECKING, Generator, List, NamedTuple, Optional, Tuple, Union

import necessary
import smart_open

with necessary.necessary("msgspec") as MSGSPEC_AVAILABLE:
    if TYPE_CHECKING or MSGSPEC_AVAILABLE:
        import msgspec

with necessary.necessary("s3fs") as S3FS_AVAILABLE:
    if TYPE_CHECKING or S3FS_AVAILABLE:
        import s3fs

with necessary.necessary("smart_open>=7.0.4") as SMART_OPEN_AVAILABLE:
    if TYPE_CHECKING or SMART_OPEN_AVAILABLE:
        import smart_open
        from smart_open.compression import _handle_zstd

        smart_open.register_compressor(".zstd", _handle_zstd)

with necessary.necessary("tqdm") as TQDM_AVAILABLE:
    if TYPE_CHECKING or TQDM_AVAILABLE:
        pass

with necessary.necessary("torch") as TORCH_AVAILABLE:
    if TYPE_CHECKING or TORCH_AVAILABLE:
        import torch  # pyright: ignore
        import torch.distributed as dist  # pyright: ignore

with necessary.necessary("transformers") as TRANSFORMERS_AVAILABLE:
    if TYPE_CHECKING or TRANSFORMERS_AVAILABLE:
        from transformers import (  # pyright: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BatchEncoding,
            PreTrainedModel,
            PreTrainedTokenizer,
        )

with necessary.necessary("wandb") as WANDB_AVAILABLE:
    if TYPE_CHECKING or WANDB_AVAILABLE:
        import wandb


LOG_EVERY = 10_000


def get_rank_and_world_size():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def get_local_gpu_rank() -> int:
    """Returns the local GPU rank for the current process using torch.distributed."""
    if dist.is_initialized():
        return dist.get_rank() % torch.cuda.device_count()
    else:
        return 0


def setup() -> Tuple[int, int]:
    if (rank := os.environ.get("RANK")) and (world_size := os.environ.get("WORLD_SIZE")):
        dist.init_process_group("nccl", rank=int(rank), world_size=int(world_size))
    return get_rank_and_world_size()


def cleanup():
    dist.destroy_process_group()


def load_model(model_name: str) -> PreTrainedModel:
    """Loads the model onto the specified GPU."""
    device = torch.device(f"cuda:{get_local_gpu_rank()}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model = torch.compile(model)  # pyright: ignore
    return model  # pyright: ignore


class WandbLogger:
    is_initialized = False
    use_wandb = False
    project = "fineweb-classifier"
    entity = "ai2"
    name = os.environ.get("GANTRY_TASK_NAME", "fineweb-classifier")

    def __new__(cls, *args, **kwargs):
        rank, _ = get_rank_and_world_size()
        if not cls.is_initialized and cls.use_wandb and rank == 0:
            wandb.init(project=cls.project, entity=cls.entity, name=cls.name)
            cls.is_initialized = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        self.rank, self.world_size = get_rank_and_world_size()

    def log(self, **kwargs):
        print(
            "{wandb}{rank}/{world_size}: {kwargs}".format(
                wandb="[wandb]" if (to_wandb := (self.rank == 0) and (self.use_wandb)) else "",
                rank=self.rank,
                world_size=self.world_size,
                kwargs=", ".join(f"{k}={v}" for k, v in kwargs.items()),
            )
        )
        if to_wandb:
            if step := kwargs.pop("step", None):
                wandb.log(kwargs, step=step)
            else:
                wandb.log(kwargs)


def make_prediction(
    tokenizer: PreTrainedTokenizer,
    batch: List["DocSpec"],
    model: PreTrainedModel,
    max_length: Optional[int] = None,
):
    inputs = tokenizer(
        [b.text for b in batch], padding=True, truncation=True, return_tensors="pt", max_length=max_length
    ).to(model.device)
    outputs = model(**inputs)
    scores = outputs.logits.squeeze(-1).float().cpu().tolist()
    return scores


def format_prediction(docs: List["DocSpec"], scores: List[float], model_name: str):
    attributes = []
    for doc, score in zip(docs, scores):
        int_score = int(round(max(0, min(score, 5))))
        attribute = {
            "id": doc.id,
            "attributes": {
                f"{model_name}_score": [[0, len(doc.text), score]],
                f"{model_name}_int_score": [[0, len(doc.text), float(int_score)]],
            },
        }
        attributes.append(attribute)
    return attributes


def format_prediction_batch(batch: "BatchItem", batch_scores: List[float], model_name: str):
    attributes = []
    for id, length, score in zip(batch.ids, batch.lengths, batch_scores):
        int_score = int(round(max(0, min(score, 5))))
        attribute = {
            "id": id,
            "attributes": {
                f"{model_name}_score": [[0, length, score]],
                f"{model_name}_int_score": [[0, length, float(int_score)]],
            },
        }
        attributes.append(attribute)
    return attributes


def async_sync_counts(counts: int) -> int:
    """
    Asynchronously reduces the counts across all GPUs and returns the result.
    """
    # Create an asynchronous CUDA stream
    stream = torch.cuda.Stream()

    # Move counts to the GPU
    tensor_counts = torch.tensor(counts).cuda()

    # Perform all_reduce operation asynchronously
    with torch.cuda.stream(stream):  # pyright: ignore
        dist.all_reduce(tensor_counts, op=dist.ReduceOp.SUM)

    # Stream synchronization ensures that the counts are reduced
    # without blocking other operations
    stream.synchronize()

    return int(tensor_counts.item())


class DocSpec(msgspec.Struct):
    id: str
    text: str


class FileReader:
    def __init__(self, source_path: str):
        self.queue: QueueType[Union[DocSpec, None]] = mp.Queue()  # type: ignore
        self.process = mp.Process(target=self.read_file, args=(source_path, self.queue))

    @staticmethod
    def read_file(source_path: str, queue: QueueType[Union[DocSpec, None]]):
        decoder = msgspec.json.Decoder(DocSpec)
        with smart_open.open(source_path, "rt") as source_file:
            for line in source_file:
                doc = decoder.decode(line)
                queue.put(doc)
        queue.put(None)

    def __enter__(self):
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.join()
        self.process.close()

    def __iter__(self) -> Generator[DocSpec, None, None]:
        if not self.process.is_alive():
            raise RuntimeError("FileReader must be used with a with statement.")
        while True:
            doc = self.queue.get()
            if doc is None:
                break
            yield doc


class BatchItem(NamedTuple):
    ids: List[str]
    lengths: List[int]
    inputs: BatchEncoding


class BatchAndTokenizeReader:
    def __init__(
        self, source_path: str, batch_size: int, tokenizer_name_or_path: str, max_length: Optional[int] = None
    ):
        self.queue: QueueType[Union[BatchItem, None]] = mp.Queue()  # type: ignore
        self.process = mp.Process(
            target=self.read_file,
            kwargs={
                "source_path": source_path,
                "queue": self.queue,
                "batch_size": batch_size,
                "tokenizer_name_or_path": tokenizer_name_or_path,
                "max_length": max_length,
            },
        )

    @classmethod
    def make_item(
        cls, batch: List[DocSpec], tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None
    ) -> BatchItem:
        inputs = tokenizer(
            [d.text for d in batch], padding=True, truncation=True, return_tensors="pt", max_length=max_length
        )
        return BatchItem(ids=[d.id for d in batch], lengths=[len(d.text) for d in batch], inputs=inputs)

    @classmethod
    def read_file(
        cls,
        source_path: str,
        queue: QueueType[Union[BatchItem, None]],
        batch_size: int,
        tokenizer_name_or_path: str,
        max_length: Optional[int] = None,
    ):
        decoder = msgspec.json.Decoder(DocSpec)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        make_item_fn = partial(cls.make_item, tokenizer=tokenizer, max_length=max_length)
        with smart_open.open(source_path, "rt") as source_file:
            batch = []
            for line in source_file:
                batch.append(decoder.decode(line))
                if len(batch) < batch_size:
                    continue

                queue.put(make_item_fn(batch))
                batch = []

        if batch:
            queue.put(make_item_fn(batch))

        queue.put(None)

    def __enter__(self):
        self.process.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.join()
        self.process.close()

    def __iter__(self) -> Generator[BatchItem, None, None]:
        if not self.process.is_alive():
            raise RuntimeError("FileReader must be used with a with statement.")
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item


def process_documents(
    rank: int,
    world_size: int,
    source_paths: List[str],
    destination_paths: List[str],
    batch_size: int,
    model_name: str,
    max_length: Optional[int] = None,
):
    """Processes a batch of files using distributed processing."""
    model = load_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    s3 = s3fs.S3FileSystem()

    step = file_cnt = 0
    model_name_attributes = model.config.name_or_path.replace("/", "_")
    model.eval()
    logger = WandbLogger()
    prev_time = time.time()

    encoder = msgspec.json.Encoder()

    for source_path, destination_path in zip(source_paths, destination_paths):
        file_cnt += 1
        if s3.exists(destination_path):
            print(f"Skipping {source_path} on GPU {rank}/{world_size} because {destination_path} already exists")
            continue

        with torch.no_grad(), smart_open.open(destination_path, "wt") as destination_file, FileReader(
            source_path
        ) as source_file:

            batch: List[DocSpec] = []
            for doc in source_file:
                step += 1
                if step % LOG_EVERY == 0:
                    throughput = LOG_EVERY / -(prev_time - (prev_time := time.time()))
                    logger.log(step=step, throughput=throughput, files=file_cnt, docs=step)

                batch.append(doc)

                if len(batch) < batch_size:
                    continue

                scores = make_prediction(tokenizer, batch, model, max_length)

                attributes = format_prediction(batch, scores, model_name_attributes)
                output = encoder.encode_lines(attributes)
                destination_file.write(output.decode("utf-8"))

                batch = []

            if batch:
                scores = make_prediction(tokenizer, batch, model, max_length)
                attributes = format_prediction(batch, scores, model_name)
                output = encoder.encode_lines(attributes)
                destination_file.write(output.decode("utf-8"))
    cleanup()


def process_documents_batch_and_tokenize(
    rank: int,
    world_size: int,
    source_paths: List[str],
    destination_paths: List[str],
    batch_size: int,
    model_name: str,
    max_length: Optional[int] = None,
):
    """Processes a batch of files using distributed processing."""
    model = load_model(model_name)
    s3 = s3fs.S3FileSystem()

    step = file_cnt = 0
    model_name_attributes = model.config.name_or_path.replace("/", "_")
    model.eval()
    logger = WandbLogger()
    prev_time = time.time()
    log_every = int(round(LOG_EVERY / batch_size) * batch_size)

    encoder = msgspec.json.Encoder()

    for source_path, destination_path in zip(source_paths, destination_paths):
        if s3.exists(destination_path):
            print(f"Skipping {source_path} on GPU {rank}/{world_size} because {destination_path} already exists")
            continue

        with torch.no_grad(), smart_open.open(destination_path, "wt") as destination_file, BatchAndTokenizeReader(
            source_path, batch_size, model_name, max_length
        ) as batches:

            file_cnt += 1

            for batch in batches:
                step += len(batch.ids)

                if step % log_every == 0:
                    throughput = log_every / -(prev_time - (prev_time := time.time()))
                    logger.log(step=step, throughput=throughput, files=file_cnt, docs=step)

                batch.inputs.to(model.device)
                outputs = model(**batch.inputs)
                scores = outputs.logits.squeeze(-1).float().cpu().tolist()

                attributes = format_prediction_batch(batch, scores, model_name_attributes)
                output = encoder.encode_lines(attributes)
                destination_file.write(output.decode("utf-8"))

    cleanup()


def longest_common_sequence(strings):
    # Split each string by "/"
    split_strings = [s.split("/") for s in strings]

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
    rank, world_size = setup()
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    WandbLogger()  # Initialize WandbLogger if use_wandb is True

    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs available, but the script is designed to use multiple GPUs.")

    s3 = s3fs.S3FileSystem()
    source_paths = [f"s3://{p}" for p in s3.glob(args.source_prefix)]

    print(f"Tagging {len(source_paths)} files from {args.source_prefix} to {args.output_prefix}")

    if all("/documents/" in p for p in source_paths):
        source_prefix = longest_common_sequence([p.split("/documents/", 1)[0] for p in source_paths])
        source_prefix = f"{source_prefix}/documents/"
    else:
        source_prefix = longest_common_sequence(source_paths)

    destination_paths = [
        f'{args.output_prefix.rstrip("/")}/{p.replace(source_prefix, "").lstrip("/")}' for p in source_paths
    ]

    # Distribute files across processes
    files_per_process = len(source_paths) // world_size
    partition_source_paths = source_paths[rank * files_per_process : (rank + 1) * files_per_process]
    partition_destination_paths = destination_paths[rank * files_per_process : (rank + 1) * files_per_process]

    print(
        f"Partitioned into {len(partition_source_paths)} of size {len(partition_source_paths[0])}. "
        f"Processing {rank}/{world_size}"
    )
    process_documents(
        rank=rank,
        world_size=world_size,
        model_name=args.model_name,
        source_paths=partition_source_paths,
        destination_paths=partition_destination_paths,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify text from JSONL files on S3 using a Hugging Face model."
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        required=True,
        help="S3 glob pattern for input files (e.g., s3://path/to/docs/*/*.jsonl.gz)",
    )
    parser.add_argument("--output-prefix", type=str, default=None, help="S3 prefix to save the results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceFW/fineweb-edu-classifier",
        help="Hugging Face model name (default: allenai/fineweb-edu-classifier)",
    )
    parser.add_argument(
        "--max-length", type=int, default=None, help="Maximum sequence length for tokenization (default: None)"
    )
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument(
        "--wandb-project", type=str, default="fineweb-classifier", help="Weights & Biases project name"
    )
    parser.add_argument("--wandb-entity", type=str, default="ai2-llm", help="Weights & Biases entity name")
    opts = parser.parse_args()

    if opts.output_prefix is None:
        if "/documents/" not in opts.source_prefix:
            raise ValueError("Output prefix is required unless source prefix contains 'documents'")
        base, _ = opts.source_prefix.split("/documents/", 1)
        opts.output_prefix = f"{base}/attributes/fineweb-edu-classifier"

    if opts.use_wandb:
        WandbLogger.use_wandb = True
        WandbLogger.project = opts.wandb_project
        WandbLogger.entity = opts.wandb_entity

    return opts


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # path = "/Users/lucas/Downloads/0000_dclm_shard_00000065.jsonl.zstd"
    # cnt = 0
    # import tqdm
    # # with FileReader(path) as f:
    # with BatchAndTokenizeReader(path, 10, "bert-base-uncased") as f:
    #     for batch in tqdm.tqdm(f):
    #         cnt += 1
    # print(cnt)
