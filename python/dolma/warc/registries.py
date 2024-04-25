from typing import TYPE_CHECKING, Type

from ..core.registry import BaseRegistry

if TYPE_CHECKING:
    from .extractors import BaseExtractor
    from .linearizers import BaseLinearizer


class ExtractorRegistry(BaseRegistry[Type["BaseExtractor"]]):
    pass


class LinearizerRegistry(BaseRegistry[Type["BaseLinearizer"]]):
    pass
