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
import json
import os
import time
from itertools import zip_longest
from typing import TYPE_CHECKING, List, Optional, Tuple

import smart_open
import necessary

with necessary.necessary("s3fs") as S3FS_AVAILABLE:
    if TYPE_CHECKING or S3FS_AVAILABLE:
        import s3fs

with necessary.necessary("tqdm") as TQDM_AVAILABLE:
    if TYPE_CHECKING or TQDM_AVAILABLE:
        pass

with necessary.necessary("torch") as TORCH_AVAILABLE:
    if TYPE_CHECKING or TORCH_AVAILABLE:
        import torch  # pyright: ignore
        import torch.distributed as dist # pyright: ignore

with necessary.necessary("transformers") as TRANSFORMERS_AVAILABLE:
    if TYPE_CHECKING or TRANSFORMERS_AVAILABLE:
        from transformers import ( # pyright: ignore
            AutoModelForSequenceClassification,
            PreTrainedModel,
            PreTrainedTokenizer,
            AutoTokenizer
        )

with necessary.necessary("wandb") as WANDB_AVAILABLE:
    if TYPE_CHECKING or WANDB_AVAILABLE:
        import wandb


def get_rank_and_world_size():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def setup() -> Tuple[int, int]:
    if (rank := os.environ.get("RANK")) and (world_size := os.environ.get("WORLD_SIZE")):
        dist.init_process_group("nccl", rank=int(rank), world_size=int(world_size))
    return get_rank_and_world_size()

def cleanup():
    dist.destroy_process_group()


def load_model(model_name: str, rank: int) -> PreTrainedModel:
    """Loads the model onto the specified GPU."""
    device = torch.device(f'cuda:{rank}')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model = torch.compile(model) # pyright: ignore
    return model    # pyright: ignore


class WandbLogger:
    is_initialized = False
    use_wandb = False
    project = "fineweb-classifier"
    entity = "ai2"
    name = os.environ.get("GANTRY_TASK_NAME", "fineweb-classifier")

    def __new__(cls, *args, **kwargs):
        if not cls.is_initialized and cls.use_wandb:
            wandb.init(project=cls.project, entity=cls.entity, name=cls.name)
            cls.is_initialized = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        self.rank, self.world_size = get_rank_and_world_size()

    def log(self, **kwargs):
        if self.use_wandb and self.rank == 0:
            if (step := kwargs.pop('step', None)):
                wandb.log(kwargs, step=step)
            else:
                wandb.log(kwargs)

        for k, v in kwargs.items():
            print(f"{self.rank}/{self.world_size} {k}: {v}")


def make_prediction(
    tokenizer: PreTrainedTokenizer,
    batch: List[dict],
    model: PreTrainedModel,
    max_length: Optional[int] = None
):
    inputs = tokenizer(
        [b['text'] for b in batch],
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_length
    ).to(model.device)
    outputs = model(**inputs)
    scores = outputs.logits.squeeze(-1).float().cpu().tolist()
    return scores

def format_prediction(docs: List[dict], scores: List[float], model_name: str):
    attributes = []
    for doc, score in zip(docs, scores):
        int_score = int(round(max(0, min(score, 5))))
        attribute = {
            "id": doc["id"],
            "attributes": {
                f"{model_name}_score": [[0, len(doc['text']), score]],
                f"{model_name}_int_score": [[0, len(doc['text']), float(int_score)]]
            }
        }
        attributes.append(attribute)
    return attributes

def process_file(
    source_path: str,
    destination_path: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: Optional[int] = None,
):
    """Processes a single JSONL file and classifies the text."""

    model.eval()
    model_name = model.config.name_or_path.replace('/', '_')
    start_time = time.time()

    logger = WandbLogger()

    with torch.no_grad():
        with smart_open.open(source_path, 'rt') as source_file, \
            smart_open.open(destination_path, 'wt') as destination_file:
            batch = []

            for i, line in enumerate(source_file):
                if i % 10_000 == 0:
                    throughput = i / (time.time() - start_time)
                    logger.log(step=i, throughput=throughput)

                batch.append(json.loads(line))

                if len(batch) < batch_size:
                    continue

                scores = make_prediction(tokenizer, batch, model, max_length)

                attributes = format_prediction(batch, scores, model_name)
                for attribute in attributes:
                    destination_file.write(json.dumps(attribute) + '\n')

                batch = []

            if batch:
                scores = make_prediction(tokenizer, batch, model, max_length)
                attributes = format_prediction(batch, scores, model_name)
                for attribute in attributes:
                    destination_file.write(json.dumps(attribute) + '\n')


def async_sync_counts(counts: int) -> int:
    """
    Asynchronously reduces the counts across all GPUs and returns the result.
    """
    # Create an asynchronous CUDA stream
    stream = torch.cuda.Stream()

    # Move counts to the GPU
    tensor_counts = torch.tensor(counts).cuda()

    # Perform all_reduce operation asynchronously
    with torch.cuda.stream(stream): # pyright: ignore
        dist.all_reduce(tensor_counts, op=dist.ReduceOp.SUM)

    # Stream synchronization ensures that the counts are reduced
    # without blocking other operations
    stream.synchronize()

    return int(tensor_counts.item())


def process_documents(
    rank: int,
    world_size: int,
    source_paths: List[str] ,
    destination_paths: List[str],
    batch_size: int,
    model_name: str,
    max_length: Optional[int] = None
):
    """Processes a batch of files using distributed processing."""
    model = load_model(model_name, rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    s3 = s3fs.S3FileSystem()

    for source_path, destination_path in zip(source_paths, destination_paths):
        if s3.exists(destination_path):
            print(f"Skipping {source_path} on GPU {rank}/{world_size} because {destination_path} already exists")
            continue

        process_file(
            source_path=source_path,
            destination_path=destination_path,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
        )

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
    WandbLogger()   # Initialize WandbLogger if use_wandb is True

    if not torch.cuda.is_available():
        raise RuntimeError("No GPUs available, but the script is designed to use multiple GPUs.")


    s3 = s3fs.S3FileSystem()
    source_paths = [f's3://{p}' for p in s3.glob(args.source_prefix)]

    print(f"Tagging {len(source_paths)} files from {args.source_prefix} to {args.output_prefix}")

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
    parser = argparse.ArgumentParser(description='Classify text from JSONL files on S3 using a Hugging Face model.')
    parser.add_argument('--source-prefix', type=str, required=True, help='S3 glob pattern for input files (e.g., s3://path/to/docs/*/*.jsonl.gz)')
    parser.add_argument('--output-prefix', type=str, default=None, help='S3 prefix to save the results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--model-name', type=str, default="HuggingFaceFW/fineweb-edu-classifier", help='Hugging Face model name (default: allenai/fineweb-edu-classifier)')
    parser.add_argument('--max-length', type=int, default=None, help='Maximum sequence length for tokenization (default: None)')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default="fineweb-classifier", help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default="ai2-llm", help='Weights & Biases entity name')
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
