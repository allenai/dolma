import copy
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

from dolma.cli import field


@dataclass
class WorkDirConfig:
    input: Optional[str] = field(default=None, help="Path to the input directory.")
    output: Optional[str] = field(default=None, help="Path to the output directory.")


@contextmanager
def make_workdirs(config: WorkDirConfig) -> Generator[WorkDirConfig, None, None]:
    """Create temporary work directories and update the config with their paths."""

    # make a copy of the configuration
    config = copy.deepcopy(config)

    with ExitStack() as stack:
        if config.input is None:
            config.input = stack.enter_context(tempfile.TemporaryDirectory())
        if config.output is None:
            config.output = stack.enter_context(tempfile.TemporaryDirectory())

        yield config
