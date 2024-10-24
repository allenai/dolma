import os
import re
from typing import Any

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


if ".zstd" not in get_supported_compression_types():
    register_compressor(".zstd", _handle_zstd)
