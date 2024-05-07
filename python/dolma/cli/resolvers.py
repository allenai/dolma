import multiprocessing
import sys
from typing import Callable, List, Optional, TypeVar

from cached_path import cached_path
from omegaconf.omegaconf import OmegaConf as om

from ..core.paths import glob_path
from ..core.registry import BaseRegistry


C = TypeVar("C", bound=Callable)


class ResolverRegistry(BaseRegistry[Callable]):
    @classmethod
    def add(cls, name: str, desc: Optional[str] = None) -> Callable[[C], C]:
        _add_fn = super().add(name, desc)

        def _wrapped_add_fn(
            resolver: C,
            base_add_fn: C = _add_fn,    # type: ignore
            resolver_name: str = name,
        ) -> C:
            base_add_fn(resolver)
            resolver_name = f"d.{resolver_name}"
            om.register_new_resolver(resolver_name, resolver, replace=True)
            return resolver
        return _wrapped_add_fn


@ResolverRegistry.add("cache", "Download a file and replace the path with the cached path.")
def cache(path: str) -> str:
    return str(cached_path(path))


@ResolverRegistry.add("glob", "Glob this path and return a list of files.")
def glob(path: str) -> List[str]:
    globbed = list(glob_path(path))
    assert len(globbed) > 0, f"Path {path} does not match any files"
    return globbed


@ResolverRegistry.add("processes", "Return the number of processes available (optionally with buffer).")
def processes(n: int = 0) -> int:
    return max(1, multiprocessing.cpu_count() - n)


@ResolverRegistry.add("stdin", "Read from stdin and return list of paths.")
def stdin() -> List[str]:
    return [line.strip() for line in sys.stdin]
