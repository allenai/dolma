import logging


def get_logger(name: str) -> logging.Logger:
    name = f"dolma.{name}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARN)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)

    return logger
