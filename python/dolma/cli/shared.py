import copy
import os
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

from dolma.cli import field


@dataclass
class WorkDirConfig:
    input: Optional[str] = field(default=None, help="Path to the input directory.")
    output: Optional[str] = field(default=None, help="Path to the output directory.")


@contextmanager
def get_path_to_temp_file(prefix="dolma-", suffix=None) -> Generator[Path, None, None]:
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=True) as f:
        path = Path(f.name)
    yield path

    if path.exists():
        os.remove(path)


@contextmanager
def make_workdirs(config: WorkDirConfig) -> Generator[WorkDirConfig, None, None]:
    """Create temporary work directories and update the config with their paths."""

    # make a copy of the configuration
    config = copy.deepcopy(config)

    with ExitStack() as stack:
        if config.input is None:
            config.input = stack.enter_context(tempfile.TemporaryDirectory(prefix="dolma-input-"))
        if config.output is None:
            config.output = stack.enter_context(tempfile.TemporaryDirectory(prefix="dolma-output-"))

        yield config
