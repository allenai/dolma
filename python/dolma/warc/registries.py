from typing import TYPE_CHECKING, Type

from ..core.registry import BaseRegistry
from ..taggers.language import (
    BaseLanguageTagger,
    Cld2LanguageTagger,
    Cld3LanguageTagger,
    FastTextAllLanguagesDocumentTagger,
    LangdetectTagger,
    LinguaTagger,
)

if TYPE_CHECKING:
    from .html import BaseHtmlExtractor
    from .license import BaseLicenseExtractor


class LanguageTaggerRegistry(BaseRegistry[Type[BaseLanguageTagger]]):
    pass


LanguageTaggerRegistry.add("fasttext")(FastTextAllLanguagesDocumentTagger)
LanguageTaggerRegistry.add("cld3")(Cld3LanguageTagger)
LanguageTaggerRegistry.add("cld2")(Cld2LanguageTagger)
LanguageTaggerRegistry.add("lingua")(LinguaTagger)
LanguageTaggerRegistry.add("langdetect")(LangdetectTagger)
LanguageTaggerRegistry.add("null")(BaseLanguageTagger)


class HtmlExtractorRegistry(BaseRegistry[Type["BaseHtmlExtractor"]]):
    pass


class LicenseExtractorRegistry(BaseRegistry[Type["BaseLicenseExtractor"]]):
    pass
