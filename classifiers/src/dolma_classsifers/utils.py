import os


from smart_open.compression import _handle_zstd, get_supported_compression_types, register_compressor
import torch
import torch.distributed as dist


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


if ".zstd" not in get_supported_compression_types():
    register_compressor(".zstd", _handle_zstd)
