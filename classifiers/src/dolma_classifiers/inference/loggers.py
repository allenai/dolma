import logging
import os
import time

import wandb

from .utils import get_rank_and_world_size


def get_logger(logger_name: str):
    rank, world_size = get_rank_and_world_size()

    # Create a custom formatter
    class RankFormatter(logging.Formatter):
        def format(self, record):
            record.rank = rank
            record.world_size = world_size
            return super().format(record)

    # Create a logger with the given name
    logger = logging.getLogger(f'dolma_classifiers.{logger_name}')
    logger.setLevel(logging.INFO)

    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create and set the custom formatter
    formatter = RankFormatter(
        '%(asctime)s [%(rank)d/%(world_size)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


class WandbLogger:
    is_initialized = False
    use_wandb = False
    project = os.environ.get("WANDB_PROJECT", "")
    entity = os.environ.get("WANDB_ENTITY", "")
    name = os.environ.get("GANTRY_TASK_NAME", "")

    def __new__(cls, *args, **kwargs):
        rank, _ = get_rank_and_world_size()
        if not cls.is_initialized and cls.use_wandb and rank == 0:
            assert cls.project, "W&B project name is not set"
            assert cls.entity, "W&B entity name is not set"
            assert cls.name, "W&B run name is not set"
            wandb.init(project=cls.project, entity=cls.entity, name=cls.name)
            cls.is_initialized = True
        return super().__new__(cls, *args, **kwargs)

    def __init__(self):
        self.rank, self.world_size = get_rank_and_world_size()

    def log(self, **kwargs):
        if (self.rank == 0) and (self.use_wandb):
            if step := kwargs.pop("step", None):
                wandb.log(kwargs, step=step)
            else:
                wandb.log(kwargs)


class ProgressLogger:
    def __init__(self, log_every: int = 10_000, wandb_logger: WandbLogger | None = None):
        self.log_every = log_every
        self.logger = get_logger(self.__class__.__name__)
        self.start_time = self.prev_time = time.time()
        self.total_docs = 0
        self.current_docs = 0
        self.current_files = 0
        self.total_files = 0
        self.wandb_logger = wandb_logger

    def increment(self, docs: int = 0, files: int = 0):
        self.current_docs += docs
        self.current_files += files
        self.total_docs += docs
        self.total_files += files

        if self.current_docs >= self.log_every or files > 0:
            current_time = time.time()
            docs_throughput = self.current_docs / (current_time - self.prev_time)
            files_throughput = self.current_files / (current_time - self.prev_time)

            self.logger.info(
                f"Throughput: {docs_throughput:.2f} docs/s, {files_throughput:.2f} files/s " +
                f" ({self.total_docs:.1e} docs; {self.total_files:,} files)"
            )
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    step=self.total_docs,
                    instant_doc_throughput=docs_throughput,
                    total_doc_throughput=self.total_docs / (current_time - self.start_time),
                    instant_file_throughput=files_throughput,
                    total_file_throughput=self.total_files / (current_time - self.start_time),
                    total_files=self.total_files,
                )

            self.prev_time = current_time
            self.current_docs = 0
            self.current_files = 0
