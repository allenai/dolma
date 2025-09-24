import logging
import multiprocessing
from typing import Optional, Union

DOLMA_PREFIX = "dolma"


def get_logger(
    name: str,
    level: Union[int, str] = logging.WARN,
    fmt: str = "[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    if (proc_name := multiprocessing.current_process().name) == "MainProcess":
        proc_name = "main"
    proc_name = proc_name.replace(" ", "_")

    name = f"{proc_name}.dolma.{name}"
    logger = logging.getLogger(name)
    level = level if isinstance(level, int) else getattr(logging, level.upper())
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def reset_level(level: Union[int, str]) -> None:
    """
    Reset the log level for all Dolma loggers.

    Args:
        level (Union[int, str]): The log level to set. It can be either an integer
            representing the log level (e.g., logging.DEBUG) or a string
            representing the log level name (e.g., 'debug').

    Returns:
        None
    """
    if isinstance(level, str):
        if (level_tmp := getattr(logging, level.strip().upper(), None)) is not None:
            level = level_tmp
        else:
            raise ValueError(f"Invalid log level: {level}")

    for logger in logging.Logger.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            if logger.name.startswith("dolma"):
                logger.setLevel(level)
