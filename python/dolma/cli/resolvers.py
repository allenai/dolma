import multiprocessing
from typing import List, TypeVar

from cached_path import cached_path
from omegaconf.omegaconf import OmegaConf as om
from omegaconf.omegaconf import Resolver

from ..core.paths import glob_path

__all__ = ["cache", "glob", "processes"]


C = TypeVar("C", bound=Resolver)


def resolver(resolver: C) -> C:
    resolver_name = f"d.{resolver.__name__}"
    om.register_new_resolver(resolver_name, resolver, replace=True)
    return resolver


@resolver
def cache(path: str) -> str:
    return str(cached_path(path))


@resolver
def glob(path: str) -> List[str]:
    globbed = list(glob_path(path))
    assert len(globbed) > 0, f"Path {path} does not match any files"
    return globbed


@resolver
def processes(n: int = 0) -> int:
    return max(1, multiprocessing.cpu_count() - n)
