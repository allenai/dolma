import json
import warnings

# warning raised by pkg_resources used in a lot of google packages
warnings.filterwarnings("ignore", message=r".*declare_namespace\(\'.*google.*", category=DeprecationWarning)
# base warning raised when warning above are raised
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*", category=DeprecationWarning)

from . import dolma as _dolma  # type: ignore   # noqa: E402


def deduper(config: dict):
    return _dolma.deduper_entrypoint(json.dumps(config))


def mixer(config: dict):
    return _dolma.mixer_entrypoint(json.dumps(config))
