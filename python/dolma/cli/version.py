from dataclasses import dataclass

from dolma.cli import BaseCli
from dolma.core.loggers import get_logger
from dolma.version import __version__


@dataclass
class VersionConfig:
    ...


class VersionCli(BaseCli):
    CONFIG = VersionConfig
    COMMAND = "version"
    DESCRIPTION = "Get the version of the Dolma CLI."

    @classmethod
    def run(cls, parsed_config: VersionConfig):
        logger = get_logger("version", level="INFO")
        logger.info(f"Dolma CLI version: {__version__}")
