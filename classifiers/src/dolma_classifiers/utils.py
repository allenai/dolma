import os
import re
from contextlib import ExitStack
from hashlib import md5
from typing import Any, ContextManager, Dict, Generic, TypeVar

import msgspec
import torch
import torch.distributed as dist
from smart_open.compression import (
    _handle_zstd,
    get_supported_compression_types,
    register_compressor,
)


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


def setup() -> tuple[int, int]:
    if (rank := os.environ.get("RANK")) and (world_size := os.environ.get("WORLD_SIZE")):
        dist.init_process_group("nccl", rank=int(rank), world_size=int(world_size))
    return get_rank_and_world_size()


def cleanup():
    dist.destroy_process_group()


def sanitize_model_name(model_name: str, suffix_data: Any = None) -> str:
    replaced_with_underscores = re.sub("[^a-zA-Z0-9_]", "_", model_name)
    removed_duplicates = re.sub("_{2,}", "_", replaced_with_underscores)
    stripped_trailing_underscores = removed_duplicates.strip("_")

    if suffix_data:
        # encode suffix_data and use first 6 characters of md5 hash as suffix
        encoder = msgspec.json.Encoder()
        stripped_trailing_underscores += f"_{md5(encoder.encode(suffix_data)).hexdigest()[:6]}"

    return stripped_trailing_underscores


T = TypeVar("T")

class KeyedExitStack(Generic[T]):
    """From https://claude.site/artifacts/7150ff45-3cb1-41e5-be5c-0f0890aa332e"""

    def __init__(self):
        self.stack = ExitStack()
        self.resources: Dict[str, T] = {}

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, *exc_details):
        return self.stack.__exit__(*exc_details)

    def push(self, key: str, cm: ContextManager[T]) -> T:
        """Push a context manager onto the stack with an associated key."""
        resource = self.stack.enter_context(cm)
        self.resources[key] = resource
        return resource

    def __contains__(self, key: str) -> bool:
        """Check if a resource with the given key is in the stack."""
        return key in self.resources

    def __getitem__(self, key: str) -> T:
        """Get a resource by key."""
        return self.resources[key]

    def pop(self, key: str) -> None:
        """Close a specific resource and remove it from the stack."""
        if key not in self.resources:
            raise KeyError(f"No resource found with key: {key}")

        resource = self.resources[key]
        # Create a new stack for remaining resources
        new_stack = ExitStack()
        remaining_resources = {k: v for k, v in self.resources.items() if k != key}

        # Transfer all resources except the one being popped
        for k, v in remaining_resources.items():
            new_stack.push(self.stack.pop_all())

        # Close the old stack (which now only contains the resource we want to pop)
        self.stack.close()

        # Update our stack and resources
        self.stack = new_stack
        self.resources = remaining_resources


if ".zstd" not in get_supported_compression_types():
    register_compressor(".zstd", _handle_zstd)
