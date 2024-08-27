import json
import warnings
from typing import Any

# warning raised by pkg_resources used in a lot of google packages
warnings.filterwarnings("ignore", message=r".*declare_namespace\(\'.*google.*", category=DeprecationWarning)
# base warning raised when warning above are raised
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*", category=DeprecationWarning)

# must import taggers to register them
# we import the rust extension here and wrap it in a python module
from . import dolma as _dolma  # type: ignore   # noqa: E402
from .core import TaggerRegistry  # noqa: E402
from .core.errors import DolmaRustPipelineError  # noqa: E402
from .core.taggers import BaseTagger  # noqa: E402
from .taggers import DUMMY_INIT_ALL_TAGGERS  # noqa: F401, E402

__all__ = ["add_tagger", "BaseTagger", "run"]

# we create a shortcut to easily add taggers to the registry...
add_tagger = TaggerRegistry.add


def deduper(config: dict):
    """
    Run the deduper with the given configuration.

    Args:
        config (dict): The configuration for the deduper.

    Raises:
        DolmaRustPipelineError: If there is an error running the deduper.

    """
    try:
        _dolma.deduper_entrypoint(json.dumps(config))
    except RuntimeError as e:
        raise DolmaRustPipelineError(f"Error running deduper: {e}") from e


def mixer(config: dict):
    """
    Run the mixer with the given configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters for the mixer.

    Raises:
        DolmaRustPipelineError: If an error occurs while running the mixer.
    """
    try:
        _dolma.mixer_entrypoint(json.dumps(config))
    except RuntimeError as e:
        raise DolmaRustPipelineError(f"Error running mixer: {e}") from e


def run(command: str, config: Any):
    """Run the dolma CLI with the given command and configuration.

    Args:
        command (str): The command to run.
        config (Any): The configuration for the command. It can either be a dictionary
            or a structured config object for the command (e.g., DeduperConfig from dolma.cli.deduper).
    """

    # this avoids a circular import
    from .cli.main import run_from_python  # noqa: E402

    # ...and a shortcut to run the CLI from python
    run_from_python(command=command, config=config)
