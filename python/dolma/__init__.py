import warnings

# warning raised by pkg_resources used in a lot of google packages
warnings.filterwarnings("ignore", message=r".*declare_namespace\(\'.*google.*", category=DeprecationWarning)
# base warning raised when warning above are raised
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*", category=DeprecationWarning)

# must import taggers to register them
from .core import TaggerRegistry  # noqa: E402
from .core.errors import DolmaRustPipelineError  # noqa: E402
from .core.taggers import BaseTagger  # noqa: E402
from .taggers import *  # noqa: E402

# Import Rust components from the separate package
from dolma_rust_components import deduper, mixer, UrlBlocker  # noqa: E402

__all__ = [
    "add_tagger",
    "BaseTagger",
    "deduper",
    "mixer",
    "UrlBlocker",
]

# we create a shortcut to easily add taggers to the registry
add_tagger = TaggerRegistry.add
