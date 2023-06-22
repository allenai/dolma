import json

from . import dolma as _dolma    # type: ignore


def deduper(config: dict):
    return _dolma.deduper_entrypoint(json.dumps(config))


def mixer(config: dict):
    return _dolma.mixer_entrypoint(json.dumps(config))
