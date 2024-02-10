import json
import warnings

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
from .taggers import *  # noqa: E402

__all__ = [
    "add_tagger",
    "BaseTagger",
]

# we create a shortcut to easily add taggers to the registry
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
