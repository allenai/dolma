import logging
import multiprocessing


def get_logger(name: str) -> logging.Logger:
    if (proc_name := multiprocessing.current_process().name) == "MainProcess":
        proc_name = "main"
    proc_name = proc_name.replace(" ", "_")

    name = f"{proc_name}.dolma.{name}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARN)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def reset_level_dolma_loggers(level: int) -> None:
    for logger in logging.Logger.manager.loggerDict.values():
        if logger.name.startswith("dolma"):
            logger.setLevel(level)
