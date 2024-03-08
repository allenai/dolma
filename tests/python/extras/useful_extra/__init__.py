from dolma import add_tagger
from dolma.taggers.sampling import RandomNumberTagger


@add_tagger("extra_v4")
class ExtraV2Tagger(RandomNumberTagger):
    pass
