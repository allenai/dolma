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
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)

    return logger
